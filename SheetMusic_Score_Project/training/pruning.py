"""
Pruning utilities: global L1 unstructured, structured Ln pruning, and progressive pruning.

This module wraps torch.nn.utils.prune to support:
- Global unstructured pruning across Conv2d/Linear by L1 magnitude
- Structured Ln pruning on Conv2d output channels
- prune.remove to make pruning permanent
- Progressive pruning + optional fine-tuning cycles
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.nn.utils import prune


def _prunable_modules(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    targets: List[Tuple[nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight is not None:
            targets.append((m, "weight"))
        if isinstance(m, nn.Linear) and m.weight is not None:
            targets.append((m, "weight"))
    return targets


def global_unstructured_l1(model: nn.Module, amount: float = 0.3) -> None:
    """Apply global unstructured L1 pruning across Conv2d/Linear weights."""
    parameters_to_prune = _prunable_modules(model)
    if not parameters_to_prune:
        return
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=float(amount)
    )


def structured_ln_on_convs(model: nn.Module, amount: float = 0.2, n: int = 2, dim: int = 0) -> None:
    """Apply structured Ln pruning on Conv2d modules along output channel dimension."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight is not None:
            prune.ln_structured(m, name="weight", amount=float(amount), n=int(n), dim=int(dim))


def remove_all_pruning(model: nn.Module) -> None:
    """Make all pruning permanent (remove reparametrization wrappers)."""
    for m in model.modules():
        for name, _ in list(m.named_parameters(recurse=False)):
            # If module has param with name 'weight' and 'weight_orig' attribute exists => pruned
            if hasattr(m, name + "_orig") and hasattr(m, name + "_mask"):
                try:
                    prune.remove(m, name)
                except Exception:
                    pass


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def progressive_prune(
    model: nn.Module,
    *,
    steps: int = 3,
    global_amount_per_step: float = 0.1,
    structured_amount_per_step: float = 0.05,
    remove_after_each: bool = False,
) -> None:
    """Progressively apply pruning in multiple steps (unstructured + structured).

    Args:
        steps: Number of pruning rounds.
        global_amount_per_step: L1 amount per round.
        structured_amount_per_step: Ln amount per round on convs.
        remove_after_each: If True, remove pruning reparam after each round.
    """
    for _ in range(max(1, int(steps))):
        if global_amount_per_step > 0:
            global_unstructured_l1(model, amount=global_amount_per_step)
        if structured_amount_per_step > 0:
            structured_ln_on_convs(model, amount=structured_amount_per_step)
        if remove_after_each:
            remove_all_pruning(model)


def apply_pruning(
    model: nn.Module,
    *,
    mode: str = "single",  # "single" | "progressive"
    global_unstructured_amount: float = 0.3,
    structured_ln_amount: float = 0.2,
    progressive_steps: int = 3,
    remove: bool = True,
) -> nn.Module:
    """High-level API to apply pruning according to the specified mode."""
    mode = mode.lower()
    if mode == "single":
        if global_unstructured_amount > 0:
            global_unstructured_l1(model, amount=global_unstructured_amount)
        if structured_ln_amount > 0:
            structured_ln_on_convs(model, amount=structured_ln_amount)
    elif mode == "progressive":
        progressive_prune(
            model,
            steps=progressive_steps,
            global_amount_per_step=max(0.0, global_unstructured_amount / max(1, progressive_steps)),
            structured_amount_per_step=max(0.0, structured_ln_amount / max(1, progressive_steps)),
            remove_after_each=False,
        )
    else:
        raise ValueError(f"Unknown pruning mode: {mode}")

    if remove:
        remove_all_pruning(model)
    return model
