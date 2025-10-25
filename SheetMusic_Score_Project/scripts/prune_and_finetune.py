 
"""
Prune model and run (optional) re-KD fine-tuning with MLflow logging.

This script mirrors the dataset/model wiring from train_kd.py, then:
1) Loads a baseline student checkpoint
2) Applies pruning (single/progressive)
3) Fine-tunes for a few epochs (with or without KD)
4) Logs metrics, Params, MACs, and saves the pruned checkpoint
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
from models import build_student_model, build_teacher_model
from training.loops import fit
from training.pruning import (
    apply_pruning,
    count_parameters,
    global_unstructured_l1,
    structured_ln_on_convs,
    remove_all_pruning,
)
from tools.profile_macs import profile_model
from dataset_prepare import prepare_dataset

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

def collate_train_fn(batch, target_frames):
    from dataloaders.collate import collate_fixed_length
    return collate_fixed_length(batch, target_frames=target_frames, random_crop=True)

def collate_val_fn(batch, target_frames):
    from dataloaders.collate import collate_fixed_length
    return collate_fixed_length(batch, target_frames=target_frames, random_crop=False)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("==== Prune & Fine-tune Config ====")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)

    # Data prep (same as train)
    data_cfg = cfg.data
    data_root = Path(data_cfg.root)

    if getattr(data_cfg, "prepare", False):
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
        data_dir = data_root / data_cfg.dataset.lower()
    classes = list(getattr(data_cfg, "families", []))

    target_frames = int(getattr(cfg.aug, "target_frames", 0) or 0)
    train_transform = _build_mel(cfg.aug, True)
    val_transform = _build_mel(cfg.aug, False)

    base_ds_for_split = FolderAudioDataset(
        data_dir=data_dir,
        classes=classes,
        max_files_per_class=int(getattr(data_cfg, "max_per_family", 0) or 0),
        transform=None,
        return_path=False,
    )
    N = len(base_ds_for_split)
    n_val = int(N * float(cfg.train.val_ratio))
    n_train = max(1, N - n_val)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(int(cfg.train.seed))).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val : n_val + n_train]

    train_full = FolderAudioDataset(data_dir=data_dir, classes=classes,
                                    max_files_per_class=int(getattr(data_cfg, "max_per_family", 0) or 0),
                                    transform=train_transform, return_path=True)
    val_full = FolderAudioDataset(data_dir=data_dir, classes=classes,
                                  max_files_per_class=int(getattr(data_cfg, "max_per_family", 0) or 0),
                                  transform=val_transform, return_path=True)
    train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)

    num_classes = len(train_full.class_to_idx)

    # collate_fn에 lambda를 사용하지 않고, partial을 이용해 target_frames 인자를 고정
    from functools import partial
    collate_train = partial(collate_train_fn, target_frames=target_frames) if target_frames > 0 else None
    collate_val = partial(collate_val_fn, target_frames=target_frames) if target_frames > 0 else None
    
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_train)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_val)

    # Build student and load checkpoint
    mcfg = cfg.model
    student = cast(nn.Module, build_student_model(
        arch=str(getattr(mcfg, "arch", "mobilenet_v3_small")),
        width_mult=float(getattr(mcfg, "width_mult", 0.75)),
        num_classes=num_classes,
        in_channels=1,
        pretrained=bool(getattr(mcfg, "pretrained", False)),
    )).to(device)

    ckpt_in = Path(getattr(cfg.train, "checkpoint_dir", "checkpoints")) / str(getattr(cfg.train, "checkpoint_name", "student_best.pt"))
    if not ckpt_in.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_in}")
    state = torch.load(ckpt_in, map_location=device)
    student.load_state_dict(state.get("model_state", state))

    # KD teacher optional during fine-tune
    kd_cfg = cfg.train.kd
    teacher = None
    if bool(getattr(kd_cfg, "enabled", False)):
        tconf = getattr(mcfg, "teacher", None)
        tname = str(getattr(tconf, "name", "cnn_resnet18")) if tconf is not None else "cnn_resnet18"
        tpre = bool(getattr(tconf, "pretrained", True)) if tconf is not None else True
        tfrz = bool(getattr(tconf, "freeze", True)) if tconf is not None else True
        teacher = cast(nn.Module, build_teacher_model(
            name=tname,
            num_classes=num_classes,
            pretrained=tpre,
            freeze=tfrz,
        )).to(device)
        # Optional: load fine-tuned teacher checkpoint from <checkpoint_dir>/<filename>
        t_ckpt_name = str(getattr(tconf, "checkpoint", "")) if tconf is not None else ""
        if t_ckpt_name:
            t_ckpt_resolved = Path(getattr(cfg.train, "checkpoint_dir", "checkpoints")) / Path(t_ckpt_name).name
            if t_ckpt_resolved.exists():
                state = torch.load(str(t_ckpt_resolved), map_location=device)
                model_state = state.get("model_state", state)
                try:
                    teacher.load_state_dict(model_state, strict=False)
                except Exception:
                    missing, unexpected = teacher.load_state_dict(model_state, strict=False)
                    if missing or unexpected:
                        print(f"[Teacher-CKPT] Loaded with missing={len(missing)} unexpected={len(unexpected)} keys")
            else:
                print(f"[WARN] Teacher checkpoint not found: {t_ckpt_resolved}")

    # MLflow logging (initialize early so interleaved steps can log)
    mlcfg = cfg.train.logging.mlflow
    ml_enabled = bool(getattr(mlcfg, "enabled", False))
    if ml_enabled:
        # try:
        #     orig_cwd = Path(hydra.utils.get_original_cwd())
        # except Exception:
        #     orig_cwd = Path.cwd()
        #tracking_uri = str((orig_cwd / str(mlcfg.tracking_uri)).resolve())
        tracking_uri = str(mlcfg.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(mlcfg.experiment_name))

    run_name = str(getattr(cfg, "experiment_name", "prune-ft"))
    ctx = mlflow.start_run(run_name=run_name) if ml_enabled else None

    # Apply pruning
    pr_cfg = cfg.train.pruning
    if bool(getattr(pr_cfg, "enabled", False)):
        mode = str(getattr(pr_cfg, "mode", "single")).lower()
        if mode == "progressive" and bool(getattr(pr_cfg, "interleaved_finetune", False)):
            # Interleaved progressive pruning: prune -> short fine-tune -> prune -> ...
            steps = int(getattr(pr_cfg, "progressive_steps", 3))
            ga_total = float(getattr(pr_cfg, "global_unstructured_amount", 0.3))
            sa_total = float(getattr(pr_cfg, "structured_ln_amount", 0.2))
            ga_step = max(0.0, ga_total / max(1, steps))
            sa_step = max(0.0, sa_total / max(1, steps))
            step_epochs = int(getattr(pr_cfg, "step_epochs", 2))
            remove_each = bool(getattr(pr_cfg, "remove_after_each", True))

            for s in range(1, steps + 1):
                print(f"[Progressive Pruning] Step {s}/{steps}: ga={ga_step}, sa={sa_step}")
                if ga_step > 0:
                    global_unstructured_l1(student, amount=ga_step)
                if sa_step > 0:
                    structured_ln_on_convs(student, amount=sa_step)
                if remove_each:
                    remove_all_pruning(student)

                # Short fine-tune after each step
                _ = fit(
                    student,
                    train_loader,
                    val_loader,
                    device,
                    epochs=step_epochs,
                    lr=float(cfg.train.lr),
                    weight_decay=float(cfg.train.weight_decay),
                    amp=bool(cfg.train.amp),
                    ckpt_path=None,
                    num_classes=num_classes,
                    mlflow_enabled=ml_enabled,
                    kd_enabled=bool(getattr(kd_cfg, "enabled", False)),
                    teacher=teacher,
                    kd_alpha=float(getattr(kd_cfg, "alpha", 0.7)),
                    kd_temperature=float(getattr(kd_cfg, "temperature", 4.0)),
                )
        else:
            # Previous behavior: single or progressive (no interleaved FT), then one longer fine-tune below
            student = apply_pruning(
                student,
                mode=mode,
                global_unstructured_amount=float(getattr(pr_cfg, "global_unstructured_amount", 0.3)),
                structured_ln_amount=float(getattr(pr_cfg, "structured_ln_amount", 0.2)),
                progressive_steps=int(getattr(pr_cfg, "progressive_steps", 3)),
                remove=True,
            )

    try:
        if ml_enabled:
            mlflow.log_text(OmegaConf.to_yaml(cfg), artifact_file="configs/prune_finetune.yaml")
            # Params & MACs
            mlflow.log_metric("student_params", int(count_parameters(student)))
            try:
                tf = int(getattr(cfg.aug, "target_frames", 128) or 128)
                prof = profile_model(student, input_shape=(1, 1, int(cfg.aug.n_mels), tf))
                if prof.get("macs", -1) != -1:
                    mlflow.log_metric("student_macs", int(prof["macs"]))
            except Exception as e:
                print(f"[WARN] MACs profiling failed: {e}")

        # Fine-tune
        out_dir = Path(getattr(cfg.train, "checkpoint_dir", "checkpoints"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ckpt = str(out_dir / ("pruned_" + str(getattr(cfg.train, "checkpoint_name", "student_best.pt"))))
        _ = fit(
            student,
            train_loader,
            val_loader,
            device,
            # If we interleaved, we still run a final fine-tune for cfg.train.epochs epochs
            epochs=int(cfg.train.epochs),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            amp=bool(cfg.train.amp),
            ckpt_path=out_ckpt,
            num_classes=num_classes,
            mlflow_enabled=ml_enabled,
            kd_enabled=bool(getattr(kd_cfg, "enabled", False)),
            teacher=teacher,
            kd_alpha=float(getattr(kd_cfg, "alpha", 0.7)),
            kd_temperature=float(getattr(kd_cfg, "temperature", 4.0)),
            early_stopping=bool(getattr(cfg.train.early_stopping, "enabled", False)),
            es_patience=int(getattr(cfg.train.early_stopping, "patience", 5)),
            es_min_delta=float(getattr(cfg.train.early_stopping, "min_delta", 0.0)),
        )

        # Reparameterization (if enabled in model config)
        mcfg = cfg.model
        if hasattr(mcfg, "reparameterize") and bool(getattr(mcfg, "reparameterize", False)):
            if hasattr(student, "reparameterize"):
                print("[Model] Running reparameterization...")
                student.reparameterize()
                print("[Model] Reparameterization complete.")
            else:
                print("[Model] reparameterize() not implemented for this model.")
        if ml_enabled and Path(out_ckpt).exists():
            mlflow.log_artifact(out_ckpt, artifact_path="checkpoints")
    finally:
        if ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
