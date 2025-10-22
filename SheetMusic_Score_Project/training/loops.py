"""
Minimal training/evaluation loops for student-only training.

This module provides:
- train_one_epoch: single pass over the train loader with optional AMP.
- evaluate: run model on validation loader and compute simple metrics.
- fit: orchestrate epochs, save best checkpoint, and return history.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from .metrics import summarize_classification_metrics


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = True,
) -> float:
    """Train the model for one epoch and return average loss.

    Uses CrossEntropyLoss for multi-class classification.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=amp)
    total_loss = 0.0
    total_samples = 0

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp:
            with autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        bs = yb.size(0)
        total_loss += loss.detach().item() * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int) -> Dict[str, float]:
    """Evaluate the model and compute accuracy/macro-F1 on the whole loader."""
    model.eval()
    all_logits = []
    all_targets = []
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_targets.append(yb)

    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, num_classes)
    targets_cat = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, dtype=torch.long)
    return summarize_classification_metrics(logits_cat, targets_cat, num_classes)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    amp: bool = True,
    ckpt_path: Optional[str] = "checkpoint.pt",
    num_classes: Optional[int] = None,
) -> Dict[str, list]:
    """Run a minimal training loop with evaluation and checkpointing.

    Returns a history dict with train_loss and val_{acc,macro_f1} lists.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1 = -1.0

    if num_classes is None:
        # Infer from a forward pass on a batch
        xb, _, _ = next(iter(val_loader))
        with torch.no_grad():
            logits = model(xb.to(device))
        num_classes = int(logits.shape[-1])

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, amp=amp)
        metrics = evaluate(model, val_loader, device, int(num_classes))

        history["train_loss"].append(train_loss)
        history["val_acc"].append(metrics["acc"])
        history["val_macro_f1"].append(metrics["macro_f1"])

        print(f"[Epoch {epoch:02d}] loss={train_loss:.4f} val_acc={metrics['acc']:.4f} val_f1={metrics['macro_f1']:.4f}")

        # Save best on macro-F1
        if metrics["macro_f1"] > best_f1 and ckpt_path is not None:
            best_f1 = metrics["macro_f1"]
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": metrics,
            }, ckpt_path)
            print(f"[Checkpoint] Saved best model to {ckpt_path}")

    return history
