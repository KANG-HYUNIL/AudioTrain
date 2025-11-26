# Audio-to-Score Project

This project implements an audio-to-score transcription pipeline that converts audio files into multi-track MIDI and MusicXML scores. It supports two architectural modes: **Cascading** (Stem-based) and **End-to-End** (Mix-based).

## Features

- **Instrument Detection**: Identifies instruments present in the audio using PaSST.
- **Source Separation**: Separates audio into stems (vocals, drums, bass, other) using Demucs (Cascading mode).
- **Transcription**:
  - **Cascading Mode**: Transcribes each stem individually using Basic Pitch, Onsets & Frames (Piano), or CREPE (Monophonic).
  - **End-to-End Mode**: Transcribes the full mix directly using Transformer-based models (MT3, YourMT3, Perceiver TF).
- **Packaging**: Merges transcribed tracks into a single MIDI/MusicXML file with tempo estimation.
- **Experiment Tracking**: Logs parameters, metrics, and artifacts to MLflow.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some backends like `mt3` or `piano_transcription_inference` may require additional installation.*

## Usage

The main entry point is `scripts/infer_pipeline.py`. You can configure the pipeline using Hydra command-line overrides.

### 1. Cascading Mode (Default)

This mode separates the audio into stems and transcribes each stem.

```bash
python scripts/infer_pipeline.py \
    pipeline.io.input_path=./inputs/my_song.wav \
    pipeline.io.output_dir=./outputs \
    pipeline.mode=cascading
```

### 2. End-to-End Mode

This mode skips separation and uses a monolithic model to transcribe the mix.

```bash
python scripts/infer_pipeline.py \
    pipeline.io.input_path=./inputs/my_song.wav \
    pipeline.io.output_dir=./outputs \
    pipeline.mode=end2end \
    pipeline.transcriber.e2e_backend=mt3
```

**Supported E2E Backends:**
- `mt3`: Music Transformer (Google)
- `yourmt3`: Custom T5-based model
- `perceiver_tf`: Perceiver IO (TensorFlow)

## Configuration

Configuration is managed by Hydra in `configs/pipeline/base.yaml`.

Key parameters:
- `pipeline.mode`: `cascading` or `end2end`
- `pipeline.detector.enabled`: Enable/disable instrument detection.
- `pipeline.separator.model`: Model for Demucs (e.g., `htdemucs`).
- `pipeline.transcriber.e2e_backend`: Backend for E2E mode.

## Project Structure

- `audio2score/`: Core package source.
  - `pipeline.py`: Main orchestration logic.
  - `transcription.py`: Transcription backends and adapters.
  - `separation.py`: Source separation wrappers.
  - `detection.py`: Instrument detection wrappers.
- `configs/`: Hydra configuration files.
- `scripts/`: Execution scripts.
- `notebooks/`: Colab notebooks for experimentation.
