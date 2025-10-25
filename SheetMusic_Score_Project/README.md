# Audio KD + Pruning Mini Project (NSynth-first)

End-to-end scaffold to compare Knowledge Distillation (KD) and pruning strategies for audio instrument classification using log-mel spectrograms. Focus is on: student (MobileNet) vs KD (with a pretrained CNN teacher), single vs progressive pruning, and complete MLflow logging for Params/MACs/metrics.

This project currently prioritizes NSynth (via small local subset creation); other dataset files are out of scope and may be removed or ignored during experiments.

## Project structure

```
configs/           # Hydra YAML (data, aug, model, train)
dataloaders/       # Log-Mel pipeline, SpecAug, collate
dataset_prepare/   # nsynth subset builder + folder dataset
models/            # Student (MobileNet), Teacher (pretrained CNN factory)
training/          # KD loss, loops, pruning, metrics
tools/             # MACs profiler, ONNX (optional)
scripts/           # train_kd, prune_and_finetune, run_experiments, GPU check
notebooks/         # (optional) EDA
requirements.txt   # Dependencies
```

## Requirements

- Windows (PowerShell) or cross-platform
- Python 3.10+
- FFmpeg installed (for torchaudio backend) recommended

Install deps (prefer a virtual environment):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start MLflow UI locally:

```powershell
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

## Dataset (NSynth subset, ≤1GB target)

We stream NSynth from Hugging Face and materialize a small, class-balanced subset into a folder-of-wavs layout:

```
data/nsynth/<family>/<index>.wav
```

Families/classes are configured in `configs/data/nsynth.yaml`. By default, `prepare: true` will create the subset on first run.

## How the pipeline works

1) Hydra composes configs from `configs/config.yaml` defaults:
   - `data: nsynth`, `aug: mel_16k`, `model: student`, `train: base` => merged into `cfg.data/cfg.aug/cfg.model/cfg.train`.
2) `scripts/train_kd.py` builds the dataset with `FolderAudioDataset` and applies `LogMelSpectrogram` (+ optional SpecAug for train).
3) Batches are collated to a fixed time length via `dataloaders.collate.collate_fixed_length` (shape: B, C=1, F, T).
4) Student model is built from `models.student_mobilenet.build_student_model` (MobileNet V2/V3 adapted to 1ch).
5) If `train.kd.enabled=true`, a teacher is built from `models.teacher_passt.build_teacher_model` (ImageNet-pretrained CNN adapted to 1ch) and KD loss is used.
6) Training/eval happens in `training.loops.fit` with AMP and MLflow logging per epoch.
7) Params/MACs are profiled (thop) and logged.
8) For pruning, `scripts/prune_and_finetune.py` loads a baseline checkpoint, applies single or progressive pruning via `training.pruning`, and fine-tunes (with or without KD).

Metrics: accuracy, macro-F1; Artifacts: config snapshot, checkpoints, history JSON; Scalars: train loss, val acc/F1, KD components (kl/ce), Params, MACs.

## Quickstart (PowerShell)

1) Check CUDA availability and optionally install GPU wheels:

```powershell
python -m scripts.check_and_setup_gpu          # show torch/versions and CUDA availability
python -m scripts.check_and_setup_gpu --install  # optional: attempt installing CUDA wheels (cu124)
```

---

## Scenario-based Experiment Execution

Select Yaml for each Scenario

1. **Non-KD, No Pruning**
   - `configs/train/train_nonKD_nonPruning.yaml` 사용
   - run: 
     ```powershell
     python -m scripts.train_kd
     ```

2. **Non-KD, Single Pruning**
   - `configs/train/train_nonKD_singlePruning.yaml` 사용
   - run: 
     ```powershell
     python -m scripts.prune_and_finetune
     ```

3. **Non-KD, Progressive Pruning**
   - `configs/train/train_nonKD_progressivePruning.yaml` 사용
   - run: 
     ```powershell
     python -m scripts.prune_and_finetune
     ```

4. **KD, No Pruning**
   - `configs/train/train_KD_nonPruning.yaml` 사용
   - run: 
     ```powershell
     python -m scripts.train_kd
     ```

5. **KD, Single Pruning**
   - `configs/train/train_KD_singlePruning.yaml` 사용
   - run: 
     ```powershell
     python -m scripts.prune_and_finetune
     ```

6. **KD, Progressive Pruning**
   - `configs/train/train_KD_progressivePruning.yaml` 사용
   - run: 
     ```powershell
     python -m scripts.prune_and_finetune
     ```

**실행 방법:**
1) `configs/config.yaml`,  `defaults.train` -> change scenario by setting yaml
   예시:
   ```yaml
   defaults:
     - data: nsynth
     - aug: mel_16k
     - model: student
     - train: train_KD_singlePruning  # 원하는 시나리오로 변경
   ```
2) 

**MLflow UI**: http://127.0.0.1:5000 

## Hyperparameter tuning

- Log-mel: `configs/aug/mel_16k.yaml` (n_mels, n_fft, hop_length, SpecAug masks and percentages)
- Student: `configs/model/student.yaml` (arch, width_mult)
- KD: `configs/train/base.yaml` → `kd.alpha`, `kd.temperature`; Teacher spec: `configs/model/student.yaml` → `teacher.*`
- Pruning: `configs/train/base.yaml` → `pruning.*` (mode, amounts, steps)

Interleaved options:

- `pruning.interleaved_finetune`: when true and mode=progressive, runs short fine-tunes between pruning steps
- `pruning.step_epochs`: epochs per interleaved fine-tune step
- `pruning.remove_after_each`: call prune.remove() after each step (recommended)

Override via CLI for quick sweeps. Example:

```powershell
python -m scripts.train_kd aug.n_mels=64 aug.hop_length=160 train.kd.enabled=true train.kd.alpha=0.6
```

## Reproducibility & outputs

- Checkpoints: `./checkpoints/<checkpoint_name>`
- MACs/Params: logged to MLflow metrics; per-run config snapshot in artifacts
- History: `training/history.json` artifact (per-epoch metrics)

## Notes on datasets

This project runs NSynth-first. Other dataset modules (IRMAS/OpenMIC) are not used in the current experiment plan. If they remain in the tree, ignore them; they will not be invoked by the scripts here.

## License & usage

This repository is for research/education. Respect all third-party licenses (datasets, pretrained models, separation tools). Do not upload copyrighted audio.
