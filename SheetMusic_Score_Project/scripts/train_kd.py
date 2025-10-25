"""
Hydra entrypoint for wiring configs and data pipeline.

This initial version composes Hydra configs, optionally prepares an NSynth subset
via Hugging Face Datasets streaming, builds a folder-based dataset, and runs a
single-batch smoke test with the Log-Mel transform. Model and training loops
will be added in subsequent iterations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

# Local utilities
from dataset_prepare.folder_audio import FolderAudioDataset
from dataloaders.collate import collate_fixed_length
from training.loops import fit
from models import build_teacher_model, build_student_model
from tools.profile_macs import profile_model

try:
    # Optional: NSynth subset creator (uses HF datasets)
    from dataset_prepare.nsynth import prepare_nsynth_subset
except Exception:
    prepare_nsynth_subset = None  # type: ignore[assignment]
from dataset_prepare import prepare_dataset

def _resolve_device(name: str) -> torch.device:
    """Resolve device by config string: "auto" | "cpu" | "cuda"."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_mel_transform(aug_cfg, *, train_mode: bool = True) -> Optional[Any]:
    """Construct a Log-Mel transform, optionally composed with SpecAug.

    Args:
        aug_cfg: Config node under cfg.aug with mel/specaug parameters.
        train_mode: When True and specaug.enable is True, apply SpecAug after mel.

    Returns:
        A callable transform or None.
    """
    if not getattr(aug_cfg, "enable", True):
        return None
    try:
        from dataloaders.mel_pipeline import LogMelSpectrogram
    except Exception as e:
        print(f"[WARN] Failed to import LogMelSpectrogram: {e}. Proceeding without mel transform.")
        return None

    mel = LogMelSpectrogram(
        sample_rate=aug_cfg.sample_rate,
        n_mels=aug_cfg.n_mels,
        n_fft=aug_cfg.n_fft,
        hop_length=aug_cfg.hop_length,
        f_min=aug_cfg.f_min,
        f_max=aug_cfg.f_max,
        log_mel=aug_cfg.log_mel,
        normalize=aug_cfg.normalize,
    )

    # Optionally wrap with SpecAug during training only
    try:
        from dataloaders.augment import SpecAugment, ComposeMelAndAug
        if train_mode and getattr(aug_cfg, "specaug", None) and aug_cfg.specaug.enable:
            specaug = SpecAugment(
                max_time_mask_pct=float(getattr(aug_cfg.specaug, "time_mask_pct", 0.10)),
                max_freq_mask_pct=float(getattr(aug_cfg.specaug, "freq_mask_pct", 0.15)),
                num_time_masks=int(getattr(aug_cfg.specaug, "time_mask", 2)),
                num_freq_masks=int(getattr(aug_cfg.specaug, "freq_mask", 2)),
                mask_value=0.0,
            )
            return ComposeMelAndAug(mel, specaug)
    except Exception as e:
        # If augment module is missing or fails, just return mel
        print(f"[WARN] SpecAug not applied: {e}")
    return mel

def collate_train_fn(batch, target_frames):
    return collate_fixed_length(batch, target_frames=target_frames, random_crop=True)
def collate_val_fn(batch, target_frames):
    return collate_fixed_length(batch, target_frames=target_frames, random_crop=False)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Compose Hydra configs and run a minimal data pipeline smoke test.

    Notes:
        - With `defaults: [data: nsynth, aug: mel_16k, model: student, train: base]`,
          the content of each group option is merged directly under its group node.
          Example: cfg.data.root, cfg.data.dataset, cfg.aug.sample_rate, cfg.train.batch_size.
        - There is no `cfg.data.nsynth` unless you explicitly nest your YAML like:
          `nsynth: { root: ... }` or set `# @package data.nsynth`.
    """
    print("==== Composed Config ====")
    print(OmegaConf.to_yaml(cfg))

    # Seed & device (flat access)
    torch.manual_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)
    print(f"[Device] Using: {device}")

    # Data config (flat)
    data_cfg = cfg.data
    data_root = Path(data_cfg.root)

    data_dir = None

    if getattr(data_cfg, "prepare", False):
        # Download or subset the dataset using the registered preparer
        data_dir = prepare_dataset(
            data_cfg.dataset,
            root=data_root,
            split=data_cfg.split,
            families=list(getattr(data_cfg, "families", [])),
            max_per_family=int(getattr(data_cfg, "max_per_family", 100)),
            target_sr=int(cfg.aug.sample_rate),
            # 기타 데이터셋별 인자 추가 가능
        )
    else:
        # Use existing local folder structure
        # 예: data/nsynth, data/tinysol 등
        data_dir = data_root / data_cfg.dataset.lower()

    # 이후 데이터셋 로딩
    classes = list(getattr(data_cfg, "families", []))

    # Build Log-Mel transforms for train/val
    train_transform = _build_mel_transform(cfg.aug, train_mode=True)
    val_transform = _build_mel_transform(cfg.aug, train_mode=False)

    # Dataset & split indices (deterministic)
    max_per_class = int(getattr(data_cfg, "max_per_family", getattr(data_cfg, "max_files_per_class", 0)) or 0)
    base_ds_for_split = FolderAudioDataset(
        data_dir=data_dir,
        classes=classes,
        max_files_per_class=max_per_class,
        transform=None,
        return_path=False,
    )
    N = len(base_ds_for_split)
    n_val = int(N * float(cfg.train.val_ratio))
    n_train = max(1, N - n_val)
    # Build deterministic indices based on seed
    g = torch.Generator().manual_seed(int(cfg.train.seed))
    perm = torch.randperm(N, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val: n_val + n_train]

    # Build actual train/val datasets with their own transforms
    train_full = FolderAudioDataset(
        data_dir=data_dir,
        classes=classes,
        max_files_per_class=max_per_class,
        transform=train_transform,
        return_path=True,
    )
    val_full = FolderAudioDataset(
        data_dir=data_dir,
        classes=classes,
        max_files_per_class=max_per_class,
        transform=val_transform,
        return_path=True,
    )
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    num_classes = len(train_full.class_to_idx)
    print(f"[Data] num_classes={num_classes} classes={list(train_full.class_to_idx.keys())}")

    # Collate: fix time length to target_frames for consistent batch shapes
    target_frames = int(getattr(cfg.aug, "target_frames", 0) or 0)
    # Collate functions are now defined at module level for multiprocessing compatibility
 
    from functools import partial
    collate_train = partial(collate_train_fn, target_frames=target_frames) if target_frames > 0 else None
    collate_val = partial(collate_val_fn, target_frames=target_frames) if target_frames > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_val,
    )

    # Smoke test: one batch from train
    xb, yb, pb = next(iter(train_loader))
    if isinstance(xb, torch.Tensor):
        print(f"[Batch] x={tuple(xb.shape)} y={tuple(yb.shape)} example_path={pb[0]}")
    else:
        print(f"[Batch] features type={type(xb)} count={len(xb)}")

    # Build student model and run a forward pass for shape verification
    model_cfg = cfg.model
    arch = getattr(model_cfg, "arch", "mobilenet_v3_small")
    width_mult = float(getattr(model_cfg, "width_mult", 0.75))
    pretrained = bool(getattr(model_cfg, "pretrained", False))

    student = cast(nn.Module, build_student_model(
        arch=arch,
        width_mult=width_mult,
        num_classes=num_classes,
        in_channels=1,
        pretrained=pretrained,
    )).to(device)

    # Optional teacher model for KD
    kd_cfg = cfg.train.kd
    teacher = None
    if bool(getattr(kd_cfg, "enabled", False)):
        tconf = getattr(model_cfg, "teacher", None)
        teacher_name = str(getattr(tconf, "name", "cnn_resnet18")) if tconf is not None else "cnn_resnet18"
        teacher_pretrained = bool(getattr(tconf, "pretrained", True)) if tconf is not None else True
        teacher_freeze = bool(getattr(tconf, "freeze", True)) if tconf is not None else True
        teacher = cast(nn.Module, build_teacher_model(
            name=teacher_name, num_classes=num_classes, pretrained=teacher_pretrained, freeze=teacher_freeze
        )).to(device)
        # Optional: load a fine-tuned teacher checkpoint if provided
        teacher_ckpt = str(getattr(tconf, "checkpoint", "")) if tconf is not None else ""
        if teacher_ckpt:
            # Resolve as <checkpoint_dir>/<filename>
            ckpt_dir_t = Path(getattr(cfg.train, "checkpoint_dir", "checkpoints"))
            ckpt_path = ckpt_dir_t / Path(teacher_ckpt).name
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=device)
                model_state = state.get("model_state", state)
                missing, unexpected = teacher.load_state_dict(model_state, strict=False)
                if missing or unexpected:
                    print(f"[Teacher-CKPT] Loaded with missing={len(missing)} unexpected={len(unexpected)} keys")
            else:
                print(f"[WARN] Teacher checkpoint not found: {ckpt_path}")

    # Train end-to-end
    ckpt_dir = Path(getattr(cfg.train, "checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(getattr(cfg.train, "checkpoint_name", "student_best.pt"))
    ckpt_path = str(ckpt_dir / ckpt_name)

    # MLflow configuration
    mlcfg = cfg.train.logging.mlflow
    ml_enabled = bool(getattr(mlcfg, "enabled", False))

    if ml_enabled:
        # Use absolute tracking URI rooted at original CWD to avoid Hydra run dir nesting
        # try:
        #     orig_cwd = Path(hydra.utils.get_original_cwd())
        # except Exception:
        #     orig_cwd = Path.cwd()
        #tracking_uri = str((orig_cwd / str(mlcfg.tracking_uri)).resolve())
        tracking_uri = str(mlcfg.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(mlcfg.experiment_name))

        run_name = str(getattr(cfg, "experiment_name", "run"))
        with mlflow.start_run(run_name=run_name):
            # Log composed config
            mlflow.log_text(OmegaConf.to_yaml(cfg), artifact_file="configs/composed.yaml")

            # Log key params
            params = {
                # Data
                "data.dataset": str(cfg.data.dataset),
                "data.split": str(getattr(cfg.data, "split", "")),
                "data.classes": ",".join(classes),
                "data.max_per_class": int(max_per_class),
                "data.num_classes": int(num_classes),
                "data.n_train": int(len(train_ds)),
                "data.n_val": int(len(val_ds)),
                # Aug
                "aug.sample_rate": int(cfg.aug.sample_rate),
                "aug.n_mels": int(cfg.aug.n_mels),
                "aug.n_fft": int(cfg.aug.n_fft),
                "aug.hop_length": int(cfg.aug.hop_length),
                "aug.target_frames": int(getattr(cfg.aug, "target_frames", 0) or 0),
                "aug.specaug": bool(getattr(cfg.aug.specaug, "enable", False)),
                # Model
                "model.arch": str(arch),
                "model.width_mult": float(width_mult),
                "model.pretrained": bool(pretrained),
                # Teacher
                "kd.enabled": bool(getattr(kd_cfg, "enabled", False)),
                "kd.alpha": float(getattr(kd_cfg, "alpha", 0.7)),
                "kd.temperature": float(getattr(kd_cfg, "temperature", 4.0)),
                "teacher.name": str(getattr(model_cfg, "teacher", {}).get("name", "resnet18")),
                "teacher.pretrained": bool(getattr(model_cfg, "teacher", {}).get("pretrained", True)),
                "teacher.freeze": bool(getattr(model_cfg, "teacher", {}).get("freeze", True)),
                # Train
                "train.batch_size": int(cfg.train.batch_size),
                "train.epochs": int(cfg.train.epochs),
                "train.lr": float(cfg.train.lr),
                "train.weight_decay": float(cfg.train.weight_decay),
                "train.amp": bool(cfg.train.amp),
                "train.val_ratio": float(cfg.train.val_ratio),
                # System
                "device": str(device),
                "seed": int(cfg.train.seed),
            }
            mlflow.log_params(params)

            # Model size (params)
            num_params = sum(p.numel() for p in student.parameters())
            mlflow.log_metric("model_params", int(num_params))

            # Profile MACs with a nominal input (B=1, C=1, F=n_mels, T=target_frames)
            try:
                tf = int(getattr(cfg.aug, "target_frames", 128) or 128)
                prof = profile_model(student, input_shape=(1, 1, int(cfg.aug.n_mels), tf))
                if prof.get("macs", -1) != -1:
                    mlflow.log_metric("model_macs", int(prof["macs"]))
            except Exception as e:
                print(f"[WARN] Profile MACs failed: {e}")

            history = fit(
                student,
                train_loader,
                val_loader,
                device,
                epochs=int(cfg.train.epochs),
                lr=float(cfg.train.lr),
                weight_decay=float(cfg.train.weight_decay),
                amp=bool(cfg.train.amp),
                ckpt_path=ckpt_path,
                num_classes=num_classes,
                mlflow_enabled=True,
                kd_enabled=bool(getattr(kd_cfg, "enabled", False)),
                teacher=teacher,
                kd_alpha=float(getattr(kd_cfg, "alpha", 0.7)),
                kd_temperature=float(getattr(kd_cfg, "temperature", 4.0)),
                early_stopping=bool(getattr(cfg.train.early_stopping, "enabled", False)),
                es_patience=int(getattr(cfg.train.early_stopping, "patience", 5)),
                es_min_delta=float(getattr(cfg.train.early_stopping, "min_delta", 0.0)),
            )

            # Log history as an artifact (JSON)
            import json, tempfile

            with tempfile.TemporaryDirectory() as td:
                hist_path = Path(td) / "history.json"
                hist_path.write_text(json.dumps(history, indent=2))
                mlflow.log_artifact(str(hist_path), artifact_path="training")

            # Log best checkpoint
            if Path(ckpt_path).exists():
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    else:
        history = fit(
            student,
            train_loader,
            val_loader,
            device,
            epochs=int(cfg.train.epochs),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            amp=bool(cfg.train.amp),
            ckpt_path=ckpt_path,
            num_classes=num_classes,
            mlflow_enabled=False,
            kd_enabled=bool(getattr(kd_cfg, "enabled", False)),
            teacher=teacher,
            kd_alpha=float(getattr(kd_cfg, "alpha", 0.7)),
            kd_temperature=float(getattr(kd_cfg, "temperature", 4.0)),
        )

    # Reparameterization (if enabled in model config)
    if hasattr(model_cfg, "reparameterize") and bool(getattr(model_cfg, "reparameterize", False)):
        if hasattr(student, "reparameterize"):
            print("[Model] Running reparameterization...")
            student.reparameterize()
            print("[Model] Reparameterization complete.")
        else:
            print("[Model] reparameterize() not implemented for this model.")
    print(f"[Done] Training finished. Best checkpoint: {ckpt_path}")
# ...existing code...

if __name__ == "__main__":
    main()
