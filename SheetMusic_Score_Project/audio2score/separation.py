"""Source separation wrappers with a registry-based backend system.

Default backend: Demucs (HTDemucs). For P0, we register stub backends; real
integrations can replace them later without changing the pipeline code.
"""
from __future__ import annotations
from typing import Dict, Union, Optional
from pathlib import Path

import torch
from dataclasses import dataclass
from .utils import Registry
import sys
import subprocess
import tempfile
import logging
import time

log = logging.getLogger(__name__)

try:
    import torchaudio
except Exception as _e:  # pragma: no cover
    torchaudio = None  # type: ignore


# Registry for separation backends
SEPARATOR_REGISTRY = Registry("separator")


@dataclass
class SeparationConfig:
    """Config for separation backends."""
    name: str = "demucs"
    model_name: str = "htdemucs"
    source: str = "hub"                  # hub | local
    checkpoint_path: Optional[str] = None # local checkpoint path (TBD for CLI)
    sample_rate: int = 32000
    # Optional device preference; if None/'auto', pick cuda if available, else cpu.
    device: Optional[str] = None


@SEPARATOR_REGISTRY.register("demucs")
def build_demucs():
    """Factory returning a Demucs separator callable (CLI-based implementation).

    Flow:
      1) Ensure demucs and torchaudio are installed; else raise with guidance.
      2) Prepare input wav: if tensor, save temp wav at cfg.sample_rate.
      3) Choose device: cuda if available and allowed, else cpu.
      4) Run demucs via subprocess: `python -m demucs.separate -n {model} -d {device} -o {tmpdir} {wav}`.
      5) Collect output stems (*.wav), load with torchaudio, resample to cfg.sample_rate, mono, peak-normalize.
    """

    def _pick_device(req: Optional[str]) -> str:
        if req is None or req == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return req

    def _ensure_packages() -> None:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required. Install torchaudio/ffmpeg and retry.")
        try:
            import importlib.util as _iu  # noqa: F401
            if _iu.find_spec("demucs") is None:
                raise ImportError("demucs not installed")
        except Exception as e:
            raise RuntimeError(
                "Demucs is not installed. Please run: pip install demucs"
            ) from e

    def _write_temp_wav(wav: torch.Tensor, sr: int, dirpath: Path) -> Path:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to write temporary WAV files.")
        wav = wav.detach().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        out = dirpath / "input.wav"
        torchaudio.save(str(out), wav.float(), sample_rate=int(sr))
        return out

    def _run_cli(input_wav: Path, model_name: str, device: str, out_dir: Path, checkpoint_path: Optional[str]) -> None:
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", model_name,
            "-d", device,
            "-o", str(out_dir),
            str(input_wav),
        ]
        log.info(f"Running Demucs CLI: {' '.join(cmd)}")
        # TODO: add checkpoint flag when required by a specific demucs version
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            log.error(f"Demucs CLI failed. STDERR: {proc.stderr}")
            raise RuntimeError(
                f"Demucs CLI failed ({proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        log.debug(f"Demucs CLI output: {proc.stdout}")

    def _collect_stems(out_dir: Path, model_name: str, input_base: str) -> Dict[str, Path]:
        # Typical output: {out_dir}/{model_name}/{input_base}/*.wav
        stems_root = out_dir / model_name / input_base
        if stems_root.exists():
            return {p.stem.lower(): p for p in stems_root.glob("*.wav")}
        # Fallback: search recursively
        return {p.stem.lower(): p for p in out_dir.rglob("*.wav")}

    def _load_and_standardize(path: Path, target_sr: int) -> torch.Tensor:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to load stems.")
        wav, sr = torchaudio.load(str(path))
        wav = wav.float()
        if int(sr) != int(target_sr):
            wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        peak = wav.abs().amax().clamp(min=1e-8)
        wav = wav / peak
        return wav

    def _separate(input_audio: Union[str, Path, torch.Tensor], cfg: SeparationConfig) -> Dict[str, torch.Tensor]:
        _ensure_packages()
        device = _pick_device(cfg.device)

        with tempfile.TemporaryDirectory(prefix="demucs_sep_") as td:
            tdir = Path(td)
            # Prepare input file
            if isinstance(input_audio, (str, Path)):
                input_wav = Path(input_audio)
                input_base = input_wav.stem
                # If incoming file SR differs from cfg.sample_rate, Demucs can handle it; we'll standardize outputs later.
            else:
                input_wav = _write_temp_wav(input_audio, cfg.sample_rate, tdir)
                input_base = input_wav.stem

            # Run Demucs
            out_dir = tdir / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            t_cli_start = time.time()
            _run_cli(input_wav, cfg.model_name, device, out_dir, cfg.checkpoint_path)
            t_cli_end = time.time()

            # Collect and load stems
            stem_paths = _collect_stems(out_dir, cfg.model_name, input_base)
            if not stem_paths:
                return {}
            
            t_load_start = time.time()
            stems: Dict[str, torch.Tensor] = {
                stem: _load_and_standardize(p, cfg.sample_rate) for stem, p in stem_paths.items()
            }
            t_load_end = time.time()
            
            log.debug(f"[Demucs] CLI run: {t_cli_end - t_cli_start:.2f}s, Stem loading: {t_load_end - t_load_start:.2f}s")
            return stems

    return _separate


@SEPARATOR_REGISTRY.register("spleeter")
def build_spleeter():
    """Factory returning a Spleeter-like separator callable (stub for now)."""

    def _separate(input_audio: Union[str, Path, torch.Tensor], cfg: SeparationConfig) -> Dict[str, torch.Tensor]:
        # Placeholder identical to demucs stub for now
        if isinstance(input_audio, (str, Path)):
            return {}
        wav = input_audio.detach().clone()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return {"mix": wav}

    return _separate


def build_separator(name: str = "demucs"):
    factory = SEPARATOR_REGISTRY.get(name)
    return factory()


def separate_demucs(
    input_audio: Union[str, Path, torch.Tensor],
    sample_rate: int,
    model_name: str = "htdemucs",
    source: str = "hub",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Separate audio into stems using Demucs (registry-backed).

    This wrapper preserves the previous function name/signature (with added
    source/checkpoint options) to minimize pipeline changes.
    """
    t0 = time.time()
    log.info(f"[Separation] Starting separation with model '{model_name}'")

    sep = build_separator(name="demucs")
    cfg = SeparationConfig(
        name="demucs",
        model_name=model_name,
        source=source,
        checkpoint_path=checkpoint_path,
        sample_rate=sample_rate,
    )
    stems = sep(input_audio, cfg)
    
    duration = time.time() - t0
    log.info(f"[Separation] Finished in {duration:.2f}s. Stems found: {list(stems.keys())}")
    return stems


def map_stems_to_instruments(stems: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Map stem names to instrument groups.

    Args:
        stems: Dict of stem name -> waveform.

    Returns:
        Dict of stem name -> instrument group label.

    Notes:
        Implement mapping based on configs.pipeline.separator.stem_map.
    """
    return {name: name for name in stems.keys()}
