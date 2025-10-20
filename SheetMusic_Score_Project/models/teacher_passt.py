"""
Teacher model (PaSST or AST placeholder).
TODO: Load pretrained checkpoint via timm/transformers or local.
"""

from typing import Optional

import torch
import torch.nn as nn


class TeacherBackbone(nn.Module):
    def __init__(self, num_classes: int, checkpoint: Optional[str] = None):
        super().__init__()
        # TODO: replace with real PaSST/AST backbone
        self.backbone = nn.Identity()
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
