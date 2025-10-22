"""
Student model factory for MobileNet backbones (V3-Small / V2) adapted for audio spectrograms.

This module provides utilities to:
- build a MobileNet model with 1-channel input (for log-mel spectrograms),
- replace the classifier head to match the number of classes,
- optionally start from torchvision pretrained weights (for 3ch models) and
  adapt the first convolution weights to mono by RGB-average.

Input shape contract
- Expected: (B, C=1, F, T), where F is number of mel bins and T is time frames.
- MobileNet models include AdaptiveAvgPool2d(1) before the classifier, so variable
  (F, T) is supported without changes.

Usage
    model = build_student_model(
        arch="mobilenet_v3_small", width_mult=0.75, num_classes=K,
        in_channels=1, pretrained=False
    )
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models as tvm
import math


def _replace_first_conv_to_mono(model: nn.Module, *, in_channels: int = 1, pretrained: bool = False) -> None:
    """
    Replace the first Conv2d layer to accept `in_channels` (default mono=1).

    If `pretrained` is True and the original conv had 3 input channels, the new
    conv's weights are initialized by averaging the RGB weights to a single channel.
    Otherwise, Kaiming initialization is used by default for the new conv.
    """
    # MobileNetV3/V2: first conv is typically model.features[0][0]
    # We search for the first Conv2d encountered in features.
    first_conv = None
    parent = None
    parent_key = None

    # Try common path for MobileNetV3/V2
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        seq: nn.Sequential = model.features
        for idx in range(len(seq)):
            m = seq[idx]
            if isinstance(m, nn.Sequential):
                for inner_idx in range(len(m)):
                    mm = m[inner_idx]
                    if isinstance(mm, nn.Conv2d):
                        first_conv = mm
                        parent = m
                        parent_key = inner_idx
                        break
            elif isinstance(m, nn.Conv2d):
                first_conv = m
                parent = seq
                parent_key = idx
            if first_conv is not None:
                break

    if first_conv is None:
        # Fallback: scan all modules
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                # We cannot easily set by name here; prefer known backbones.
                break

    if first_conv is None:
        raise RuntimeError("Could not locate the first Conv2d in the model to adapt input channels.")

    if first_conv.in_channels == in_channels:
        return  # already matches

    def _to_2tuple(v):
        if isinstance(v, tuple):
            if len(v) >= 2:
                return (int(v[0]), int(v[1]))
            if len(v) == 1:
                return (int(v[0]), int(v[0]))
            # Defensive default for empty tuple (should not occur in practice)
            return (1, 1)
        return (int(v), int(v))

    padding = first_conv.padding if isinstance(first_conv.padding, str) else _to_2tuple(first_conv.padding)
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=_to_2tuple(first_conv.kernel_size),
        stride=_to_2tuple(first_conv.stride),
        padding=padding,
        dilation=_to_2tuple(first_conv.dilation),
        groups=first_conv.groups,
        bias=(first_conv.bias is not None),
        padding_mode=first_conv.padding_mode,
    )

    # Initialize weights
    with torch.no_grad():
        if pretrained and first_conv.in_channels == 3 and in_channels == 1:
            # Average RGB filters to mono
            w = first_conv.weight.clone()  # (out_c, 3, kh, kw)
            w_mono = w.mean(dim=1, keepdim=True)  # (out_c, 1, kh, kw)
            new_conv.weight.copy_(w_mono)
            if new_conv.bias is not None and first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        else:
            # Default init (Kaiming uniform)
            nn.init.kaiming_uniform_(new_conv.weight, a=math.sqrt(5))
            if new_conv.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_conv.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(new_conv.bias, -bound, bound)

    # Replace in model
    if parent is not None and parent_key is not None and isinstance(parent, nn.Sequential):
        parent[parent_key] = new_conv
    else:
        # Best-effort: not expected for MobileNetV3/V2 paths
        raise RuntimeError("Failed to set the new first conv in the model.")


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
    """
    Replace the final classifier to output `num_classes` logits.

    MobileNetV3-Small: model.classifier[-1] is nn.Linear(1024 -> 1000)
    MobileNetV2:       model.classifier[-1] is nn.Linear(1280 -> 1000) (width_mult affects dims)
    """
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return
    # Fallback: try to find a linear layer named 'fc' or last linear
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            # Set via attribute assignment if directly on the model
            if hasattr(model, name):
                setattr(model, name, nn.Linear(in_features, num_classes))
                return
    raise RuntimeError("Could not replace classifier head automatically.")


def build_student_model(
    arch: str = "mobilenet_v3_small",
    *,
    width_mult: float = 0.75,
    num_classes: int = 3,
    in_channels: int = 1,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build a MobileNet student model adapted for mono spectrogram inputs.

    Args:
        arch: Backbone name: "mobilenet_v3_small" or "mobilenet_v2".
        width_mult: Width multiplier for channels (0.5 ~ 1.0 typically).
        num_classes: Number of output classes (logits).
        in_channels: Input channels; for log-mel use 1.
        pretrained: Use torchvision's pretrained ImageNet weights (3ch) and adapt
                    first conv to mono by averaging RGB weights.

    Returns:
        nn.Module: Configured MobileNet model.
    """
    if arch == "mobilenet_v3_small":
        # torchvision 0.14+: weights API; keep compatibility
        try:
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = tvm.mobilenet_v3_small(weights=weights)
        except Exception:
            model = tvm.mobilenet_v3_small(pretrained=pretrained)
        # Apply width multiplier if available (torchvision's mobilenet_v3_small does not take width_mult directly)
        # For simplicity, we keep default channels; if strict width_mult is needed, switch to timm.
    elif arch == "mobilenet_v2":
        try:
            weights = tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = tvm.mobilenet_v2(weights=weights, width_mult=width_mult)
        except Exception:
            model = tvm.mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
    else:
        raise ValueError(f"Unsupported student arch: {arch}")

    # Adapt first conv to mono (and handle pretrained weight adaptation)
    _replace_first_conv_to_mono(model, in_channels=in_channels, pretrained=pretrained)

    # Replace classifier head
    _replace_classifier(model, num_classes=num_classes)

    return model
"""
Student model: MobileNetV2/V3 small for multi-label classification.
TODO: Use torchvision mobilenet and replace classifier head with Sigmoid activation at inference.
"""

from typing import Optional

import torch
import torch.nn as nn


class StudentMobileNet(nn.Module):
    def __init__(self, num_classes: int = 14, width_mult: float = 0.75, pretrained: bool = False):
        super().__init__()
        # TODO: load torchvision.models.mobilenet_v2 and adapt
        self.backbone = nn.Identity()
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.backbone(x))
        return logits  # BCEWithLogitsLoss during training
