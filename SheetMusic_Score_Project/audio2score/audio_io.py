"""Audio I/O utilities for the audio-to-score pipeline.

All functions return/expect float32 waveforms peak-normalized to [-1, 1].
Each module is responsible for internal resampling to the target sample rate.

Notes:
- Prefer torchaudio for consistency; fallback gracefully with informative errors.
- Keep functions small and reusable; log shapes and sampling rates at call sites.
"""
from __future__ import annotations
from typing import Tuple
from pathlib import Path

import torch

try:
    import torchaudio
    from torchaudio.functional import resample as ta_resample
except Exception as e:  # pragma: no cover
    torchaudio = None  # type: ignore
    ta_resample = None  # type: ignore


def load_audio(path: str | Path, target_sr: int = 16000, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """Load an audio file and return a float32 tensor and the sample rate.

    Args:
        path: Input audio file path.
        target_sr: Desired sample rate for downstream modules.
        mono: When True, convert to mono by averaging channels.

    Returns:
        (waveform, sample_rate): waveform is a float32 tensor shaped (1, T) when mono else (C, T).

    Raises:
        RuntimeError: If torchaudio is not available or loading fails.
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to load audio. Install torchaudio/FFmpeg.")
    waveform, sr = torchaudio.load(str(path))
    waveform = waveform.to(torch.float32)
    # Peak-normalize to [-1, 1]
    peak = waveform.abs().amax().clamp(min=1e-8)
    waveform = waveform / peak
    if mono and waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sr):
        if ta_resample is None:
            raise RuntimeError("torchaudio functional.resample unavailable.")
        waveform = ta_resample(waveform, orig_freq=int(sr), new_freq=int(target_sr))
        sr = int(target_sr)
    return waveform, int(sr)


essential_save_dtype = torch.float32

def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save an audio tensor to file using torchaudio.

    Args:
        path: Destination path (dir created automatically).
        waveform: Tensor (C, T) float32 in [-1, 1]. Mono is (1, T).
        sample_rate: Sampling rate to write.
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to save audio.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu().to(essential_save_dtype)
    torchaudio.save(str(out), wav, int(sample_rate))
