"""
Transformer-based teacher builders using timm ViT.

Registered names:
- vit_b16  (vit_base_patch16_224)
- vit_s16  (vit_small_patch16_224)

Notes:
- We adapt in_chans=1 to consume (B, 1, F, T). Features are resized to 224x224
  via adaptive pooling to match ViT expected spatial size.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from . import register_teacher


class _ResizeWrapper(nn.Module):
    """Wrap a backbone expecting (B,C,224,224) to accept arbitrary (B,C,F,T)."""
    def __init__(self, backbone: nn.Module, img_size: Tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=1, F, T) -> resize to (B,C,224,224)
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return self.backbone(x)


def _make_headed(model: nn.Module, num_classes: int) -> nn.Module:
    # timm ViT has model.head or model.get_classifier()
    in_feats = None
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_feats = model.head.in_features
        model.head = nn.Linear(in_feats, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, num_classes)
    else:
        # Use timm helper
        in_feats = model.get_classifier().in_features if hasattr(model.get_classifier(), "in_features") else 768
        model.reset_classifier(num_classes)
    return model


@register_teacher("vit_b16")
def build_vit_b16(*, num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, in_chans=1, num_classes=num_classes)
    model = _make_headed(model, num_classes)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze head
        for m in model.modules():
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    p.requires_grad = True
    return _ResizeWrapper(model)


@register_teacher("vit_s16")
def build_vit_s16(*, num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    model = timm.create_model("vit_small_patch16_224", pretrained=pretrained, in_chans=1, num_classes=num_classes)
    model = _make_headed(model, num_classes)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        for m in model.modules():
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    p.requires_grad = True
    return _ResizeWrapper(model)
