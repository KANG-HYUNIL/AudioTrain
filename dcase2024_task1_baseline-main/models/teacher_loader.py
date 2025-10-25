# Teacher model registry and loader
import torch
from typing import Dict

TEACHER_REGISTRY = {}

def register_teacher(name):
    def decorator(cls):
        TEACHER_REGISTRY[name] = cls
        return cls
    return decorator

# Example: DeiT-B (HuggingFace)
@register_teacher("deit_b")
class DeiTBTeacher(torch.nn.Module):
    def __init__(self, pretrained=True, input_size=224, n_classes=10, checkpoint_path=None, use_local_checkpoint=False):
        super().__init__()
        from transformers import DeiTForImageClassification
        self.model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224") if pretrained else DeiTForImageClassification(config={"num_labels": n_classes})
        self.input_size = input_size
        if use_local_checkpoint and checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))
    def forward(self, x):
        return self.model(x).logits

# Add more teacher models here using @register_teacher

def get_teacher_model(cfg: Dict):
    name = cfg.get("name")
    if name not in TEACHER_REGISTRY:
        raise ValueError(f"Unknown teacher model: {name}")
    return TEACHER_REGISTRY[name](**cfg)
