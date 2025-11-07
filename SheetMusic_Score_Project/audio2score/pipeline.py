"""End-to-end audio-to-score pipeline orchestration.

This module ties together detection, separation, transcription, and packaging.
It is designed for Colab-first execution but works locally as well.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import time

import torch

from .audio_io import load_audio
from .detection import detect_instruments
from .separation import separate_demucs, map_stems_to_instruments
from .transcription import pick_transcriber, MidiTrack
from .packaging import merge_tracks, export_midi, export_musicxml

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def run_pipeline(input_audio: str, cfg) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Run the detector → separator → transcriber → packager sequence.

    Args:
        input_audio: Path to input audio file.
        cfg: Hydra config with groups: pipeline(io/detector/separator/transcriber/export) and aug (optional).

    Returns:
        artifacts: Dict of artifact names to file paths (e.g., {'midi': '.../out.mid'})
        stats:     Dict of timings and detected info for MLflow logging.
    """
    pio = cfg.pipeline.io
    pdet = cfg.pipeline.detector
    psep = cfg.pipeline.separator
    ptx = cfg.pipeline.transcriber
    pexp = cfg.pipeline.export

    start_all = time.time()

    # MLflow setup
    ml_enabled = bool(getattr(cfg.pipeline.logging.mlflow, "enabled", False)) if getattr(cfg.pipeline, "logging", None) else False
    if ml_enabled and mlflow is not None:
        mlflow.set_tracking_uri(str(cfg.pipeline.logging.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.pipeline.logging.mlflow.experiment_name))
        ml_ctx = mlflow.start_run(run_name="audio2score-infer")
    else:
        ml_ctx = None

    artifacts: Dict[str, str] = {}
    stats: Dict[str, Any] = {"timings": {}}

    try:
        # 1) Load
        t0 = time.time()
        wav, sr = load_audio(input_audio, target_sr=int(pio.sample_rate), mono=True)
        stats["timings"]["load_s"] = time.time() - t0
        stats["sr"] = sr
        stats["num_samples"] = int(wav.shape[-1])

        # 2) Detect (optional)
        detected = {}
        if bool(getattr(pdet, "enabled", True)) and len(getattr(pdet, "classes", [])) > 0:
            t1 = time.time()
            # Prepare optional mel params from cfg.aug (if present)
            mel_params = None
            if hasattr(cfg, "aug"):
                mel_params = {
                    "sample_rate": int(getattr(cfg.aug, "sample_rate", sr)),
                    "n_fft": int(getattr(cfg.aug, "n_fft", 1024)),
                    "hop_length": int(getattr(cfg.aug, "hop_length", 160)),
                    "n_mels": int(getattr(cfg.aug, "n_mels", 128)),
                    "f_min": float(getattr(cfg.aug, "f_min", 20.0)),
                    "f_max": float(getattr(cfg.aug, "f_max", sr/2)),
                    "log_mel": bool(getattr(cfg.aug, "log_mel", True)),
                    "normalize": bool(getattr(cfg.aug, "normalize", True)),
                }
            detected = detect_instruments(
                wav,
                sr,
                classes=list(pdet.classes),
                name=str(pdet.name),
                threshold=float(pdet.threshold),
                source=str(getattr(pdet, "source", "hub")),
                checkpoint_path=getattr(pdet, "checkpoint_path", None),
                mel_params=mel_params,
            )
            stats["timings"]["detect_s"] = time.time() - t1
            stats["detected"] = detected

        # 3) Separate (optional)
        stems = {"mix": wav}
        if bool(getattr(psep, "enabled", True)):
            t2 = time.time()
            stems = separate_demucs(
                input_audio,
                sample_rate=sr,
                model_name=str(getattr(psep, "model", "htdemucs")),
                source=str(getattr(psep, "source", "hub")),
                checkpoint_path=getattr(psep, "checkpoint_path", None),
            )
            stats["timings"]["separate_s"] = time.time() - t2
        stats["stems"] = list(stems.keys())

        # 4) Transcribe per stem
        t3 = time.time()
        tracks: Dict[str, MidiTrack] = {}

        # Apply mapping from config (e.g., htdemucs: vocals/drums/bass/other)
        stem_map = dict(getattr(psep, "stem_map", {}))
        mapped_stems: Dict[str, torch.Tensor] = {}
        name_count: Dict[str, int] = {}

        for stem_name, stem_wav in stems.items():
            instr = stem_map.get(stem_name, stem_name)
            key = instr
            if key in mapped_stems:
                idx = name_count.get(instr, 1) + 1
                name_count[instr] = idx
                key = f"{instr}_{idx}"
            else:
                name_count[instr] = 1
            mapped_stems[key] = stem_wav

        for instr, stem_wav in mapped_stems.items():
            backend = str(getattr(ptx, "per_instrument", {}).get(instr, ptx.default))
            transcriber = pick_transcriber(instr, default=backend)
            try:
                track = transcriber(stem_wav, sr)  # unified (waveform, sr) interface
            except Exception:
                from .transcription import MidiTrack as _MidiTrack
                track = _MidiTrack(instrument=instr, notes=[])

            tracks[instr] = track
        stats["timings"]["transcribe_s"] = time.time() - t3
        # Basic metric: total note count across tracks
        stats["note_count"] = int(sum(len(t.notes) for t in tracks.values()))

        # 5) Package
        t4 = time.time()
        bundle = merge_tracks(tracks, tempo_bpm=getattr(cfg.pipeline.export, "tempo_bpm", None), metadata={"detected": detected})
        out_dir = Path(pio.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        midi_path = out_dir / (Path(input_audio).stem + ".mid")
        artifacts["midi"] = export_midi(bundle, midi_path)
        if bool(getattr(pexp, "musicxml", False)):
            xml_path = out_dir / (Path(input_audio).stem + ".musicxml")
            artifacts["musicxml"] = export_musicxml(bundle, xml_path)
        stats["timings"]["package_s"] = time.time() - t4

        stats["timings"]["total_s"] = time.time() - start_all

        # MLflow logging
        if ml_ctx is not None and mlflow is not None:
            mlflow.log_params({
                "detector_name": str(pdet.name),
                "separator_name": str(psep.name),
                "transcriber_default": str(ptx.default),
                "sample_rate": int(pio.sample_rate),
            })
            if detected:
                mlflow.log_dict(detected, "detected.json")
            mlflow.log_metrics({k: float(v) for k, v in stats["timings"].items()})
            mlflow.log_metrics({"note_count": float(stats.get("note_count", 0))})
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path="artifacts")

        return artifacts, stats
    finally:
        if ml_ctx is not None and mlflow is not None:
            mlflow.end_run()
