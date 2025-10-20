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
