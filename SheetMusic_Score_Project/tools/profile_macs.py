"""
Profile model MACs and Params using thop.
"""

from __future__ import annotations
from typing import Tuple, Dict

import torch


def profile_model(model: torch.nn.Module, input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128)) -> Dict[str, int]:
    """Return MACs and Params for a model given an example input shape.

    Args:
        model: torch.nn.Module
        input_shape: (B, C, F, T) example shape
    """
    try:
        from thop import profile
    except Exception:
        return {"macs": -1, "params": sum(p.numel() for p in model.parameters())}

    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)
    out = profile(model, inputs=(dummy,), verbose=False)
    if isinstance(out, tuple):
        macs = out[0]
        params = out[1]
    else:
        macs = out.macs if hasattr(out, "macs") else -1
        params = out.params if hasattr(out, "params") else sum(p.numel() for p in model.parameters())
    return {"macs": int(macs), "params": int(params)}
