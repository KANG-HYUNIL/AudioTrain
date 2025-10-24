"""
Fine-tune a Teacher model on the current dataset (NSynth-first) and save a checkpoint.

This script mirrors data wiring from train_kd.py but trains the TEACHER model
as a normal classifier (CrossEntropy). The saved checkpoint can then be loaded
in train_kd.py via model.teacher.checkpoint for stronger KD.
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

from dataset_prepare.folder_audio import FolderAudioDataset
from dataloaders.collate import collate_fixed_length
from dataloaders.mel_pipeline import LogMelSpectrogram
from dataloaders.augment import SpecAugment, ComposeMelAndAug
from models import build_teacher_model
from training.loops import fit
from tools.profile_macs import profile_model


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_mel(aug_cfg, train_mode: bool):
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
    if train_mode and getattr(aug_cfg, "specaug", None) and aug_cfg.specaug.enable:
        spec = SpecAugment(
            max_time_mask_pct=float(getattr(aug_cfg.specaug, "time_mask_pct", 0.10)),
            max_freq_mask_pct=float(getattr(aug_cfg.specaug, "freq_mask_pct", 0.15)),
            num_time_masks=int(getattr(aug_cfg.specaug, "time_mask", 2)),
            num_freq_masks=int(getattr(aug_cfg.specaug, "freq_mask", 2)),
        )
        return ComposeMelAndAug(mel, spec)
    return mel


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("==== Teacher Fine-tune Config ====")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)

    # Data (NSynth-first)
    data_cfg = cfg.data
    if data_cfg.dataset != "nsynth":
        raise ValueError("This script currently supports nsynth-only.")
    data_root = Path(data_cfg.root)
    data_dir = data_root / "nsynth"
    classes = list(data_cfg.families)

    # Split indices (deterministic)
    base_ds = FolderAudioDataset(
        data_dir=data_dir,
        classes=classes,
        max_files_per_class=int(getattr(data_cfg, "max_per_family", 0) or 0),
        transform=None,
        return_path=False,
    )
    N = len(base_ds)
    n_val = int(N * float(cfg.train.val_ratio))
    n_train = max(1, N - n_val)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(int(cfg.train.seed))).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val : n_val + n_train]

    # Datasets with transforms
    train_tf = _build_mel(cfg.aug, True)
    val_tf = _build_mel(cfg.aug, False)
    train_full = FolderAudioDataset(data_dir=data_dir, classes=classes,
                                    max_files_per_class=int(getattr(data_cfg, "max_per_family", 0) or 0),
                                    transform=train_tf, return_path=True)
    val_full = FolderAudioDataset(data_dir=data_dir, classes=classes,
                                  max_files_per_class=int(getattr(data_cfg, "max_per_family", 0) or 0),
                                  transform=val_tf, return_path=True)
    train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)

    num_classes = len(train_full.class_to_idx)
    target_frames = int(getattr(cfg.aug, "target_frames", 0) or 0)
    collate_train = (lambda b: collate_fixed_length(b, target_frames=target_frames, random_crop=True)) if target_frames > 0 else None
    collate_val = (lambda b: collate_fixed_length(b, target_frames=target_frames, random_crop=False)) if target_frames > 0 else None

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_train)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_val)

    # Build Teacher (for finetune, do NOT freeze)
    mcfg = cfg.model
    tconf = getattr(mcfg, "teacher", {})
    tname = str(getattr(tconf, "name", "cnn_resnet18"))
    tpre = bool(getattr(tconf, "pretrained", True))
    # Force freeze False for finetuning unless explicitly set otherwise
    tfrz = bool(getattr(tconf, "freeze", False))
    teacher = cast(nn.Module, build_teacher_model(
        name=tname,
        num_classes=num_classes,
        pretrained=tpre,
        freeze=tfrz,
    )).to(device)

    # MLflow setup
    mlcfg = cfg.train.logging.mlflow
    ml_enabled = bool(getattr(mlcfg, "enabled", False))
    if ml_enabled:
        try:
            orig_cwd = Path(hydra.utils.get_original_cwd())
        except Exception:
            orig_cwd = Path.cwd()
        tracking_uri = str((orig_cwd / str(mlcfg.tracking_uri)).resolve())
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("teacher-finetune")

    ckpt_dir = Path(getattr(cfg.train, "checkpoint_dir", "checkpoints"))
    # Always resolve teacher checkpoint as <checkpoint_dir>/<filename>
    # If model.teacher.checkpoint is empty, fallback to teacher_best.pt
    t_ckpt_name = str(getattr(tconf, "checkpoint", "")).strip() if tconf is not None else ""
    if not t_ckpt_name:
        t_ckpt_name = "teacher_best.pt"
    # Normalize to filename only in case a path was mistakenly provided
    t_ckpt_name = Path(t_ckpt_name).name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / t_ckpt_name)

    run_name = str(getattr(cfg, "experiment_name", "teacher-ft"))
    ctx = mlflow.start_run(run_name=run_name) if ml_enabled else None
    try:
        if ml_enabled:
            mlflow.log_text(OmegaConf.to_yaml(cfg), artifact_file="configs/teacher_finetune.yaml")
            # Params & MACs
            try:
                params = sum(p.numel() for p in teacher.parameters())
                mlflow.log_metric("teacher_params", int(params))
                tf = int(getattr(cfg.aug, "target_frames", 128) or 128)
                prof = profile_model(teacher, input_shape=(1, 1, int(cfg.aug.n_mels), tf))
                if prof.get("macs", -1) != -1:
                    mlflow.log_metric("teacher_macs", int(prof["macs"]))
            except Exception as e:
                print(f"[WARN] MACs profiling failed: {e}")

        # Train teacher (CE only)
        _ = fit(
            teacher,
            train_loader,
            val_loader,
            device,
            epochs=int(cfg.train.epochs),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            amp=bool(cfg.train.amp),
            ckpt_path=ckpt_path,
            num_classes=num_classes,
            mlflow_enabled=ml_enabled,
            kd_enabled=False,
            teacher=None,
            early_stopping=bool(getattr(cfg.train.early_stopping, "enabled", False)),
            es_patience=int(getattr(cfg.train.early_stopping, "patience", 5)),
            es_min_delta=float(getattr(cfg.train.early_stopping, "min_delta", 0.0)),
        )

        if ml_enabled and Path(ckpt_path).exists():
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
    finally:
        if ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
