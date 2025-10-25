<!-- Copilot / AI agent instructions for working with this repo -->
# Quick orientation for AI coding agents

This file explains the concrete, discoverable patterns and workflows in the DCASE2024 Task 1 baseline repository so an AI agent can be productive immediately.

1) Big picture (what runs where)
- Entry point: `run_training.py`. It contains two top-level flows: `train(config)` (default) and `evaluate(config)` (when `--evaluate` is passed).
- Data loading: `dataset/dcase24.py` exposes `get_training_set`, `get_test_set`, `get_eval_set`. Important: the module expects a global `dataset_dir` to be set in `dataset/dcase24.py` (it is currently `None` by default). This must be configured before running training or evaluation.
- Model & preprocessing: the pipeline is raw waveform -> Resample -> MelSpectrogram -> optional augmentation -> model. See `PLModule.mel_forward` in `run_training.py` and `models/mel.py` for mel preprocessing details.
- Model architecture: implemented in `models/baseline.py` (inverted-bottleneck blocks, `get_model(...)`). Model complexity checks are done via `helpers/nessi.py`.

2) Key developer workflows and commands (concrete)
- Environment: recommended conda workflow is in `README.md` (python 3.10). Install PyTorch separately (see README) and then:
  - `pip install -r requirements.txt`
- Train baseline (example):
  - `python run_training.py --subset=100`  # trains on full training split; subset can be 5,10,25,50,100
- Evaluate / create evaluation predictions (example):
  - `python run_training.py --evaluate --ckpt_id=<wandb_id>`
  - Checkpoint lookup: `evaluate()` expects a checkpoint at `os.path.join(project_name, ckpt_id, "checkpoints", "last.ckpt")`.
  - Output folder: predictions + info are written to `predictions/<ckpt_id>/` (see `evaluate()`).
- Split downloads: when you run training, split CSV files (e.g., `split100.csv`, `test.csv`) are auto-downloaded into `split_setup` from the URL configured in `dataset/dcase24.py`.

3) Project-specific conventions and gotchas
- dataset_dir is a compile-time/constant variable in `dataset/dcase24.py`. The repository uses that single variable instead of a CLI flag or env var. Set it before running (or modify code to read an env var). Failure to do so triggers an assert at import time.
- Weights & Biases logging: `run_training.py` constructs a `WandbLogger` and uses the `project_name` and `experiment_name` from CLI args. Checkpoints are saved by Lightning's `ModelCheckpoint` callback under the wandb-run folder: `<project_name>/<wandb_id>/checkpoints/`.
- Precision handling: CLI `--precision` is parsed as a string (default `"32"`) and passed directly to `pl.Trainer(precision=config.precision)`. When changing precision prefer values Lightning accepts (`32`, `16`, `bf16`, or the string Lightning expects). If debugging on CPU, update the `accelerator` argument in Trainer (default is `'gpu'`).
- Time-rolling augmentation: `RollDataset` in `dataset/dcase24.py` uses `numpy.random.random_integers` (deprecated); it's the project's current behavior for `roll` augmentation.
- MixStyle augmentation: frequency MixStyle is implemented in `helpers/utils.py` as `mixstyle(x, p, alpha)` and is applied during training when `--mixstyle_p` > 0.

4) Integration points and checks
- Complexity checks: `helpers/nessi.py` exposes `MAX_PARAMS_MEMORY = 128_000` and `MAX_MACS = 30_000_000`. `evaluate()` asserts the model meets these limits before generating evaluation predictions.
- Mel preprocessing: `models/mel.py` contains `AugmentMelSTFT`; `PLModule` uses torchaudio transforms to build the mel pipeline. Pay attention to `sample_rate`, `n_mels`, `n_fft`, `hop_length`, `window_length` CLI flags in `run_training.py`.

5) Files to inspect when making changes (high signal)
- `run_training.py` — top-level training/eval logic, CLI args, PLModule, WandB wiring.
- `dataset/dcase24.py` — dataset entrypoints, where `dataset_dir` must be set; split downloads behavior.
- `models/baseline.py` — network architecture and `get_model()` parameters (base_channels, multiplier, expansion rate).
- `helpers/nessi.py` — complexity checks (MAX_PARAMS_MEMORY, MAX_MACS).
- `helpers/utils.py` — MixStyle augmentation implementation used in training.
- `models/mel.py` — mel spectrogram transform and augmentation logic.

6) Concrete examples for common agent tasks
- To change the default batch size and debug on CPU for a fast local smoke test:
  - run: `python run_training.py --subset=5 --batch_size=8 --num_workers=0` and edit `run_training.py` trainer creation to `accelerator='cpu', devices=1` if GPU is not available.
- To run evaluation after a successful training run with wandb id `abcd1234` and save predictions:
  - `python run_training.py --evaluate --ckpt_id=abcd1234 --project_name=DCASE24_Task1`

7) Safety for automated edits
- Avoid changing the `dataset_dir` handling lightly; many modules import it at module-load time (the code asserts `dataset_dir is not None`). If you propose making it an env variable or CLI argument, update `dataset/dcase24.py` and any module that imports `dataset_config` accordingly.

If any section above is unclear or you want examples in Korean, tell me which parts to expand or which files you want me to annotate inline; I can iterate the file.
