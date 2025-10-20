"""
Knowledge Distillation loss implementation (Hinton).
KD = alpha * T^2 * KL(student_T || teacher_T) + (1-alpha) * CE(hard)
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor,
            alpha: float = 0.7, temperature: float = 4.0) -> Tuple[torch.Tensor, dict]:
    T = temperature
    # soft targets with temperature
    p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
    ce = F.cross_entropy(student_logits, targets)
    loss = alpha * kl + (1 - alpha) * ce
    return loss, {"kl": kl.detach(), "ce": ce.detach()}
