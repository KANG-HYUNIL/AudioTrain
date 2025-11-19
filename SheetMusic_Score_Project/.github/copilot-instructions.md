# Audio-to-Score Pipeline AI-Agent Instructions

This document provides essential guidance for AI agents working on this audio-to-score codebase.

## 1. Big Picture: The Inference Pipeline

The core of this project is an audio-to-score inference pipeline orchestrated by Hydra. The main entry point is `scripts/infer_pipeline.py`.

The pipeline executes the following sequence:
1.  **Load Audio**: Loads an audio file.
2.  **Instrument Detection**: A `PaSST`-based model identifies instruments present in the audio (`audio2score/detection.py`).
3.  **Source Separation**: `Demucs` is used via a command-line wrapper to separate the audio into different instrument stems (e.g., `drums`, `bass`, `other`) (`audio2score/separation.py`).
4.  **Transcription**: Each stem is transcribed into musical notes. The backend is chosen based on the instrument (`audio2score/transcription.py`):
    -   `Basic Pitch` for most melodic instruments.
    -   `Onsets & Frames` for piano.
    -   `CREPE` for monophonic transcription.
5.  **Packaging**: The transcribed notes from all stems are merged into a single multi-track MIDI (`pretty_midi`) and optionally a MusicXML (`music21`) file. This step also handles tempo estimation (`librosa`), time signature, and velocity scaling (`audio2score/packaging.py`).
6.  **Logging**: Key parameters, metrics (e.g., processing time, note count), and output artifacts (MIDI/XML files) are logged to MLflow.

The entire process is defined and configured via Hydra YAML files located in `configs/pipeline/`. The main configuration is `configs/pipeline/base.yaml`.

## 2. Core Developer Workflow

The primary workflow is running the inference pipeline.

**To run the pipeline:**
Use `scripts/infer_pipeline.py` with Hydra overrides. The two most important parameters are the input and output paths.

```bash
# Example: Run pipeline on a sample audio file
python scripts/infer_pipeline.py pipeline.io.input_path=./inputs/sample.wav pipeline.io.output_dir=./outputs
```

- **Configuration**: To change pipeline behavior (e.g., swap models, change thresholds), modify the YAML files in `configs/pipeline/` or override them from the command line.
- **Dependencies**: Key external dependencies are `torch`, `torchaudio`, `demucs`, `basic-pitch`, `music21`, `pretty_midi`, `librosa`, `hydra-core`, `mlflow`, and `transformers`. These are installed via `pip`.
- **Debugging**: The pipeline logs extensively to `stdout`. For deeper debugging, check the intermediate artifacts saved in the output directory and the MLflow run artifacts.

## 3. Project Conventions & Patterns

- **Hydra for Everything**: All configuration, from file paths to model parameters, is managed by Hydra. When adding a new feature, expose its parameters in a new or existing YAML file under `configs/`.
- **Lazy Imports**: To keep the application lightweight and avoid hard dependencies, external libraries like `transformers`, `basic_pitch`, etc., are often imported lazily inside the functions that use them. Follow this pattern for new, heavy dependencies.
- **Registry for Backends**: The transcription module (`audio2score/transcription.py`) uses a dictionary-based registry to manage different transcription backends. This allows for easy swapping and extension.
- **CLI Wrappers for Stability**: Tools like `Demucs` are wrapped as `subprocess` calls. This isolates their environment and improves the main application's stability.
- **English Docstrings and Comments**: All code documentation must be in English, following Google-style or NumPy-style for docstrings.

## 4. Key Files and Directories

- `scripts/infer_pipeline.py`: Main entry point for the inference pipeline.
- `configs/pipeline/base.yaml`: The primary configuration file for the pipeline. This is the first place to look to understand the default setup.
- `audio2score/pipeline.py`: Contains the main `run_pipeline` function that orchestrates the entire process.
- `audio2score/{detection.py, separation.py, transcription.py, packaging.py}`: Each file corresponds to a major stage in the pipeline.
- `notebooks/colab_entry.ipynb`: A self-contained notebook for running the entire pipeline on Google Colab, including setup and execution.
- `mlruns/`: Default output directory for MLflow logs and artifacts.
