"""
Hydra entrypoint for wiring configs and data pipeline.

This initial version composes Hydra configs, optionally prepares an NSynth subset
via Hugging Face Datasets streaming, builds a folder-based dataset, and runs a
single-batch smoke test with the Log-Mel transform. Model and training loops
will be added in subsequent iterations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Subset

import hydra
from omegaconf import DictConfig, OmegaConf

# Local utilities
from dataset_prepare.folder_audio import FolderAudioDataset
from dataloaders.collate import collate_fixed_length
from training.loops import fit
from models.student_mobilenet import build_student_model

try:
    # Optional: NSynth subset creator (uses HF datasets)
    from dataset_prepare.nsynth import prepare_nsynth_subset
except Exception:
    prepare_nsynth_subset = None  # type: ignore[assignment]


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
            specaug = SpecAugment()
            return ComposeMelAndAug(mel, specaug)
    except Exception as e:
        # If augment module is missing or fails, just return mel
        print(f"[WARN] SpecAug not applied: {e}")
    return mel



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

    # Optional NSynth subset preparation
    if data_cfg.dataset == "nsynth":
        data_dir = data_root / "nsynth"
        if getattr(data_cfg, "prepare", False):
            if prepare_nsynth_subset is None:
                raise RuntimeError("dataset_prepare.nsynth not available. Install 'datasets' and check import path.")
            data_dir = prepare_nsynth_subset(
                root=data_root,
                split=data_cfg.split,                  # 'train' | 'valid' | 'test'
                families=list(data_cfg.families),
                max_per_family=int(data_cfg.max_per_family),
                target_sr=int(cfg.aug.sample_rate),
            )
        classes = list(data_cfg.families)

    else:
        raise ValueError(f"Unsupported dataset selector: {data_cfg.dataset}")

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
    collate_train = None
    collate_val = None
    if target_frames > 0:
        collate_train = lambda b: collate_fixed_length(b, target_frames=target_frames, random_crop=True)
        collate_val = lambda b: collate_fixed_length(b, target_frames=target_frames, random_crop=False)

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

    student = build_student_model(
        arch=arch,
        width_mult=width_mult,
        num_classes=num_classes,
        in_channels=1,
        pretrained=pretrained,
    ).to(device)

    # Train end-to-end (student-only for now)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "student_best.pt")

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
    )
    print(f"[Done] Training finished. Best checkpoint: {ckpt_path}")
# ...existing code...

if __name__ == "__main__":
    main()
