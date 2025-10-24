"""
Teacher model builders (CNN family) and registration.

Registered names:
- cnn_resnet18
- cnn_efficientnet_b0
- cnn_mobilenet_v3_large
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models as tvm
from . import register_teacher


def _to_2tuple(v):
    if isinstance(v, tuple):
        if len(v) >= 2:
            return (int(v[0]), int(v[1]))
        if len(v) == 1:
            return (int(v[0]), int(v[0]))
        return (1, 1)
    return (int(v), int(v))


def _adapt_first_conv_to_mono(model: nn.Module) -> None:
    """Replace first Conv2d to accept 1 input channel, RGB-avg if possible."""
    first_conv = None
    parent = None
    key = None

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        seq: nn.Sequential = model.features
        for i, m in enumerate(seq):
            if isinstance(m, nn.Conv2d):
                first_conv = m
                parent, key = seq, i
                break
            if isinstance(m, nn.Sequential):
                for j, mm in enumerate(m):
                    if isinstance(mm, nn.Conv2d):
                        first_conv = mm
                        parent, key = m, j
                        break
            if first_conv is not None:
                break
    if first_conv is None:
        # Try common ResNet style
        if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
            first_conv = model.conv1
            parent, key = model, "conv1"

    if first_conv is None:
        raise RuntimeError("Could not find first Conv2d to adapt to mono.")

    if first_conv.in_channels == 1:
        return

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=_to_2tuple(first_conv.kernel_size),
        stride=_to_2tuple(first_conv.stride),
        padding=_to_2tuple(first_conv.padding),
        dilation=_to_2tuple(first_conv.dilation),
        groups=first_conv.groups if first_conv.groups == 1 else 1,
        bias=(first_conv.bias is not None),
        padding_mode=first_conv.padding_mode,
    )
    with torch.no_grad():
        if first_conv.weight.shape[1] == 3:
            w = first_conv.weight.clone()
            w_mono = w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(w_mono)
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        else:
            nn.init.kaiming_uniform_(new_conv.weight, a=5 ** 0.5)
            if new_conv.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_conv.weight)
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(new_conv.bias, -bound, bound)

    if isinstance(parent, nn.Sequential) and isinstance(key, int):
        parent[key] = new_conv
    elif isinstance(parent, nn.Module) and isinstance(key, str):
        setattr(parent, key, new_conv)
    else:
        raise RuntimeError("Failed to set adapted first conv.")


@register_teacher("cnn_resnet18")
def build_cnn_resnet18(*, num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    try:
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
    except Exception:
        model = tvm.resnet18(pretrained=pretrained)
    _adapt_first_conv_to_mono(model)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    return model


@register_teacher("cnn_efficientnet_b0")
def build_cnn_efficientnet_b0(*, num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    try:
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
    except Exception:
        model = tvm.efficientnet_b0(pretrained=pretrained)
    _adapt_first_conv_to_mono(model)
    in_feats = int(getattr(model.classifier[-1], "in_features", 1280))
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
    return model


@register_teacher("cnn_mobilenet_v3_large")
def build_cnn_mobilenet_v3_large(*, num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    try:
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
    except Exception:
        model = tvm.mobilenet_v3_large(pretrained=pretrained)
    _adapt_first_conv_to_mono(model)
    in_feats = int(getattr(model.classifier[-1], "in_features", 1280))
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
    return model
