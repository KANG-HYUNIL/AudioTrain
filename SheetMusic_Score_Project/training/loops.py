"""
Training/evaluation loops supporting student-only and KD (teacher-student) modes.

Provided APIs:
- train_one_epoch: single pass over the train loader with optional KD and AMP.
- evaluate: run model on validation loader and compute simple metrics.
- fit: orchestrate epochs, save best checkpoint, and return history.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from .metrics import summarize_classification_metrics
from .kd_loss import kd_loss as compute_kd_loss


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = True,
    # KD options
    kd_enabled: bool = False,
    teacher: Optional[nn.Module] = None,
    kd_alpha: float = 0.7,
    kd_temperature: float = 4.0,
) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch and return (avg_loss, aux_metrics).

    When kd_enabled is True and a teacher is provided, use KD loss:
    loss = alpha * T^2 * KL(s/T || t/T) + (1-alpha) * CE(hard)
    Otherwise, use standard CrossEntropyLoss.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=amp)
    total_loss = 0.0
    total_samples = 0
    kd_running = {"kl": 0.0, "ce": 0.0}

    if kd_enabled and teacher is not None:
        teacher.eval()

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        parts = {"kl": torch.tensor(0.0, device=device), "ce": torch.tensor(0.0, device=device)} if (kd_enabled and teacher is not None) else {}

        optimizer.zero_grad(set_to_none=True)
        if amp:
            with autocast():
                logits = model(xb)
                if kd_enabled and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(xb)
                    loss, parts = compute_kd_loss(
                        logits, t_logits, yb, alpha=kd_alpha, temperature=kd_temperature
                    )
                else:
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            if kd_enabled and teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(xb)
                loss, parts = compute_kd_loss(
                    logits, t_logits, yb, alpha=kd_alpha, temperature=kd_temperature
                )
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        bs = yb.size(0)
        total_loss += loss.detach().item() * bs
        total_samples += bs
        if kd_enabled and teacher is not None:
            kd_running["kl"] += float(parts["kl"]) * bs
            kd_running["ce"] += float(parts["ce"]) * bs
    avg_loss = total_loss / max(1, total_samples)
    aux = {}
    if kd_enabled and teacher is not None and total_samples > 0:
        aux = {
            "kd_kl": kd_running["kl"] / total_samples,
            "kd_ce": kd_running["ce"] / total_samples,
        }
    return avg_loss, aux


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
    mlflow_enabled: bool = False,
    # KD options
    kd_enabled: bool = False,
    teacher: Optional[nn.Module] = None,
    kd_alpha: float = 0.7,
    kd_temperature: float = 4.0,
    # Early stopping options
    early_stopping: bool = False,
    es_patience: int = 5,
    es_min_delta: float = 0.0,
) -> Dict[str, list]:
    """Run a minimal training loop with evaluation and checkpointing.

    Returns a history dict with train_loss and val_{acc,macro_f1} lists.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1 = -1.0
    no_improve = 0

    if num_classes is None:
        # Infer from a forward pass on a batch
        xb, _, _ = next(iter(val_loader))
        with torch.no_grad():
            logits = model(xb.to(device))
        num_classes = int(logits.shape[-1])

    for epoch in range(1, epochs + 1):
        train_loss, aux = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp,
            kd_enabled=kd_enabled,
            teacher=teacher,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
        )
        metrics = evaluate(model, val_loader, device, int(num_classes))

        history["train_loss"].append(train_loss)
        history["val_acc"].append(metrics["acc"])
        history["val_macro_f1"].append(metrics["macro_f1"])

        print(f"[Epoch {epoch:02d}] loss={train_loss:.4f} val_acc={metrics['acc']:.4f} val_f1={metrics['macro_f1']:.4f}")

        # Optional MLflow logging per epoch
        if mlflow_enabled:
            try:
                import mlflow

                log_dict = {
                    "train_loss": float(train_loss),
                    "val_acc": float(metrics["acc"]),
                    "val_macro_f1": float(metrics["macro_f1"]),
                }
                if kd_enabled and aux:
                    log_dict.update({
                        "kd_kl": float(aux.get("kd_kl", 0.0)),
                        "kd_ce": float(aux.get("kd_ce", 0.0)),
                    })
                mlflow.log_metrics(log_dict, step=epoch)
            except Exception as e:
                print(f"[MLflow] Logging failed: {e}")

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

        # Early stopping check (monitor macro-F1)
        if early_stopping:
            improved = metrics["macro_f1"] > (best_f1 + es_min_delta)
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(1, int(es_patience)):
                    print(f"[EarlyStopping] No improvement for {no_improve} epochs. Stopping early.")
                    break

    return history
