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
            detected = detect_instruments(wav, sr, classes=list(pdet.classes), name=str(pdet.name), threshold=float(pdet.threshold))
            stats["timings"]["detect_s"] = time.time() - t1
            stats["detected"] = detected

        # 3) Separate (optional)
        stems = {"mix": wav}
        if bool(getattr(psep, "enabled", True)):
            t2 = time.time()
            stems = separate_demucs(input_audio, sample_rate=sr, model_name=str(psep.model))
            stats["timings"]["separate_s"] = time.time() - t2
        stats["stems"] = list(stems.keys())

        # 4) Transcribe per stem
        t3 = time.time()
        tracks: Dict[str, MidiTrack] = {}
        for stem_name, stem_wav in stems.items():
            instr = stem_name  # simplistic mapping; refine later via cfg.pipeline.separator.stem_map
            transcriber = pick_transcriber(instr, default=str(ptx.default))
            # For P0: if transcriber expects file path, pass input_audio; if tensor, pass waveform
            try:
                if transcriber.__name__ == "transcribe_basic_pitch":
                    track = transcriber(input_audio)  # type: ignore[arg-type]
                else:
                    track = transcriber(stem_wav, sr)  # type: ignore[misc]
            except Exception:
                # Fallback empty track
                from .transcription import MidiTrack as _MidiTrack
                track = _MidiTrack(instrument=instr, notes=[])
            tracks[instr] = track
        stats["timings"]["transcribe_s"] = time.time() - t3

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
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path="artifacts")

        return artifacts, stats
    finally:
        if ml_ctx is not None and mlflow is not None:
            mlflow.end_run()
