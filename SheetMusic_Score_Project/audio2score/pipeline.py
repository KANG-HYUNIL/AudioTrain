"""End-to-end audio-to-score pipeline orchestration.

This module ties together detection, separation, transcription, and packaging.
It is designed for Colab-first execution but works locally as well.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import time
import importlib
import logging

import torch

from .audio_io import load_audio
from .detection import detect_instruments
from .separation import separate_demucs

log = logging.getLogger(__name__)
from .transcription import pick_transcriber, MidiTrack, transcribe_mix_to_bundle
from .packaging import merge_tracks, export_midi, export_musicxml

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def run_pipeline(input_audio: str, cfg) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Run the audio-to-score pipeline.

    Supports two modes:
    1. Cascading: Detect -> Separate -> Transcribe (per stem) -> Package
    2. End-to-End: Detect (optional) -> Transcribe (mix to bundle) -> Package

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
    
    # Determine pipeline mode (default to cascading for backward compatibility)
    mode = getattr(cfg.pipeline, "mode", "cascading").lower()

    start_all = time.time()

    # MLflow setup
    ml_enabled = bool(getattr(cfg.pipeline.logging.mlflow, "enabled", False)) if getattr(cfg.pipeline, "logging", None) else False
    if ml_enabled and mlflow is not None:
        mlflow.set_tracking_uri(str(cfg.pipeline.logging.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.pipeline.logging.mlflow.experiment_name))
        ml_ctx = mlflow.start_run(run_name=f"audio2score-{mode}")
    else:
        ml_ctx = None

    artifacts: Dict[str, str] = {}
    stats: Dict[str, Any] = {"timings": {}, "mode": mode}

    try:
        # 1) Load
        log.info("=" * 60)
        log.info("[Stage 1/6] Loading Audio")
        log.info("=" * 60)
        log.info(f"Input file: {input_audio}")
        t0 = time.time()
        wav, sr = load_audio(input_audio, target_sr=int(pio.sample_rate), mono=True)
        load_duration = time.time() - t0
        stats["timings"]["load_s"] = load_duration
        stats["sr"] = sr
        stats["num_samples"] = int(wav.shape[-1])
        log.info(f"Audio loaded successfully.")
        log.info(f"  - Sample Rate: {sr} Hz")
        log.info(f"  - Samples: {stats['num_samples']}")
        log.info(f"  - Duration: {stats['num_samples']/sr:.2f} seconds")
        log.info(f"  - Load Time: {load_duration:.4f}s")

        # 2) Detect (optional)
        log.info("=" * 60)
        log.info("[Stage 2/6] Instrument Detection")
        log.info("=" * 60)
        detected = {}
        if bool(getattr(pdet, "enabled", True)) and len(getattr(pdet, "classes", [])) > 0:
            log.info(f"Detector: {pdet.name} (Threshold: {pdet.threshold})")
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
            detect_duration = time.time() - t1
            stats["timings"]["detect_s"] = detect_duration
            stats["detected"] = detected
            log.info(f"Detection Results: {detected}")
            log.info(f"  - Detection Time: {detect_duration:.4f}s")
        else:
            log.info("Instrument detection skipped (disabled in config).")

        # --- Branching Logic ---
        tracks: Dict[str, MidiTrack] = {}
        
        if mode == "cascading":
            # 3) Separate (optional)
            log.info("=" * 60)
            log.info("[Stage 3/6] Source Separation (Cascading Mode)")
            log.info("=" * 60)
            stems = {"mix": wav}
            if bool(getattr(psep, "enabled", True)):
                model_name = str(getattr(psep, "model", "htdemucs"))
                log.info(f"Separator Model: {model_name}")
                t2 = time.time()
                stems = separate_demucs(
                    input_audio,
                    sample_rate=sr,
                    model_name=model_name,
                    source=str(getattr(psep, "source", "hub")),
                    checkpoint_path=getattr(psep, "checkpoint_path", None),
                )
                sep_duration = time.time() - t2
                stats["timings"]["separate_s"] = sep_duration
                log.info(f"Separation complete.")
                log.info(f"  - Stems Created: {list(stems.keys())}")
                log.info(f"  - Separation Time: {sep_duration:.4f}s")
            else:
                log.info("Source separation skipped (disabled in config).")
            stats["stems"] = list(stems.keys())

            # 4) Transcribe per stem
            log.info("=" * 60)
            log.info("[Stage 4/6] Transcription (Cascading Mode)")
            log.info("=" * 60)
            t3 = time.time()
            
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
                log.info(f"Processing Instrument: {instr}")
                log.info(f"  - Backend: {backend}")
                
                instr_t0 = time.time()
                transcriber = pick_transcriber(instr, default=backend)
                try:
                    track = transcriber(stem_wav, sr)  # unified (waveform, sr) interface
                    instr_duration = time.time() - instr_t0
                    log.info(f"  - Notes Found: {len(track.notes)}")
                    log.info(f"  - Time: {instr_duration:.4f}s")
                except Exception as e:
                    log.warning(f"  - Transcription failed for {instr}: {e}. Using empty track.")
                    from .transcription import MidiTrack as _MidiTrack
                    track = _MidiTrack(instrument=instr, notes=[])

                tracks[instr] = track
            
            transcribe_duration = time.time() - t3
            stats["timings"]["transcribe_s"] = transcribe_duration

        elif mode == "end2end":
            # Skip Separation
            log.info("=" * 60)
            log.info("[Stage 3/6] Source Separation (Skipped for End-to-End Mode)")
            log.info("=" * 60)
            stats["timings"]["separate_s"] = 0.0
            stats["stems"] = ["mix"]

            # 4) Transcribe Mix directly
            log.info("=" * 60)
            log.info("[Stage 4/6] Transcription (End-to-End Mode)")
            log.info("=" * 60)
            t3 = time.time()
            
            e2e_backend = str(getattr(ptx, "e2e_backend", "mt3"))
            e2e_model_path = getattr(ptx, "e2e_model_path", None)
            e2e_vocab_path = getattr(ptx, "e2e_vocab_path", None)
            
            log.info(f"Processing Mix with End-to-End Backend: {e2e_backend}")
            
            try:
                # Call the mix transcriber which returns a MidiBundle directly
                # We need to unpack it into our local 'tracks' dict
                bundle = transcribe_mix_to_bundle(
                    wav, 
                    sr, 
                    backend=e2e_backend,
                    model_path=e2e_model_path,
                    vocab_path=e2e_vocab_path
                )
                tracks = bundle.tracks
                log.info(f"  - Tracks Found: {list(tracks.keys())}")
            except Exception as e:
                log.error(f"End-to-End transcription failed: {e}")
                # Fallback to empty tracks if needed or re-raise
                tracks = {}

            transcribe_duration = time.time() - t3
            stats["timings"]["transcribe_s"] = transcribe_duration

        else:
            raise ValueError(f"Unknown pipeline mode: {mode}. Use 'cascading' or 'end2end'.")

        # Basic metric: total note count across tracks
        stats["note_count"] = int(sum(len(t.notes) for t in tracks.values()))
        log.info(f"Transcription Stage Complete.")
        log.info(f"  - Total Notes: {stats['note_count']}")
        log.info(f"  - Total Transcription Time: {stats['timings']['transcribe_s']:.4f}s")

        # 5) Tempo estimation (if not provided) before packaging
        log.info("=" * 60)
        log.info("[Stage 5/6] Tempo Estimation")
        log.info("=" * 60)
        tempo_cfg = getattr(cfg.pipeline.export, "tempo_bpm", None)
        tempo_bpm = None
        if tempo_cfg is not None:
            tempo_bpm = float(tempo_cfg)
            log.info(f"Using configured tempo: {tempo_bpm} BPM")
        else:
            log.info("Estimating tempo from audio...")
            t_tempo = time.time()
            try:
                lb = importlib.import_module("librosa")
                # Downsample for speed if very long; use a small slice if > 120s
                wav_np = wav.detach().cpu().numpy()
                max_samples = int(sr * 120)
                if wav_np.shape[-1] > max_samples:
                    wav_np = wav_np[:max_samples]
                onset_env = lb.onset.onset_strength(y=wav_np, sr=sr)
                tempo_bpm = float(lb.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None).mean())
                log.info(f"  - Estimated Tempo: {tempo_bpm:.2f} BPM")
                log.info(f"  - Estimation Time: {time.time() - t_tempo:.4f}s")
            except Exception as e:
                log.warning(f"Tempo estimation failed: {e}")
                tempo_bpm = None
        stats["estimated_tempo_bpm"] = tempo_bpm if tempo_bpm is not None else "unknown"

        # 6) Package
        log.info("=" * 60)
        log.info("[Stage 6/6] Packaging & Export")
        log.info("=" * 60)
        t4 = time.time()
        # Pass time signature if configured (e.g., "4/4")
        time_sig = getattr(cfg.pipeline.export, "time_signature", None)
        bundle = merge_tracks(tracks, tempo_bpm=tempo_bpm, metadata={"detected": detected}, time_signature=time_sig)
        out_dir = Path(pio.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        midi_path = out_dir / (Path(input_audio).stem + ".mid")
        artifacts["midi"] = export_midi(bundle, midi_path)
        log.info(f"Exported MIDI: {midi_path}")
        if bool(getattr(pexp, "musicxml", False)):
            xml_path = out_dir / (Path(input_audio).stem + ".musicxml")
            artifacts["musicxml"] = export_musicxml(bundle, xml_path)
            log.info(f"Exported MusicXML: {xml_path}")
        
        package_duration = time.time() - t4
        stats["timings"]["package_s"] = package_duration
        log.info(f"Packaging Time: {package_duration:.4f}s")

        total_duration = time.time() - start_all
        stats["timings"]["total_s"] = total_duration
        
        log.info("=" * 60)
        log.info("PIPELINE COMPLETED SUCCESSFULLY")
        log.info("=" * 60)
        log.info(f"Total Execution Time: {total_duration:.2f}s")
        log.info("=" * 60)

        # MLflow logging
        if ml_ctx is not None and mlflow is not None:
            mlflow.log_params({
                "mode": mode,
                "detector_name": str(pdet.name),
                "separator_name": str(psep.name) if mode == "cascading" else "N/A",
                "transcriber_default": str(ptx.default) if mode == "cascading" else str(getattr(ptx, "e2e_backend", "mt3")),
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
