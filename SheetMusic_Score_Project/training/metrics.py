"""
Metrics utilities for multi-class classification (single label).

This module implements lightweight accuracy and macro-F1 without external
dependencies. It assumes logits of shape (B, K) and integer targets of shape (B,).
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch


@torch.no_grad()
def multiclass_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Top-1 accuracy for multi-class classification.

    Args:
        logits: Tensor of shape (B, K)
        targets: Long tensor of shape (B,)

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(max(1, total))


@torch.no_grad()
def multiclass_macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Compute macro-averaged F1 score across classes.

    Args:
        logits: Tensor of shape (B, K)
        targets: Long tensor of shape (B,)
        num_classes: Number of classes K

    Returns:
        Macro-F1 as a float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    f1_sum = 0.0
    eps = 1e-8
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(eps, (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1
    return f1_sum / max(1, num_classes)


@torch.no_grad()
def summarize_classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Compute a dictionary of metrics for convenience.

    Returns keys: {"acc": ..., "macro_f1": ...}
    """
    return {
        "acc": multiclass_top1_accuracy(logits, targets),
        "macro_f1": multiclass_macro_f1(logits, targets, num_classes),
    }
