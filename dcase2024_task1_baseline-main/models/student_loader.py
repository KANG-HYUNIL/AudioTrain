# Student model registry and loader
import torch
from typing import Dict

STUDENT_REGISTRY = {}

def register_student(name):
    def decorator(cls):
        STUDENT_REGISTRY[name] = cls
        return cls
    return decorator

# Example: EfficientNet-B0
@register_student("efficientnet_b0")
class EfficientNetB0Student(torch.nn.Module):
    def __init__(self, pretrained=True, input_size=224, dropout=0.2, n_classes=10, checkpoint_path=None, use_local_checkpoint=False):
        super().__init__()
        from torchvision.models import efficientnet_b0
        self.model = efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, n_classes)
        self.input_size = input_size
        self.dropout = torch.nn.Dropout(dropout)
        if use_local_checkpoint and checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))
    def forward(self, x):
        return self.model(x)


# RepVGG-A0 (RepConv student)
from models.rep_conv import RepConvBlock
@register_student("repvgg_a0")
class RepVGG_A0_Student(torch.nn.Module):
    def __init__(self, input_shape=128, num_classes=10, repconv=True, ckpt_path=None):
        super().__init__()
        # repconv 옵션에 따라 branches 설정
        if repconv:
            branches = {"1x1": True, "3x1": True, "1x3": True, "identity": False}
        else:
            branches = {"1x1": False, "3x1": False, "1x3": False, "identity": False}
        self.features = torch.nn.Sequential(
            RepConvBlock(1, 32, kernel_size=3, stride=1, padding=1, branches=branches),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Linear(32, num_classes)
        if ckpt_path:
            self.load_state_dict(torch.load(ckpt_path))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_student_model(cfg: Dict):
    name = cfg.get("name")
    if name not in STUDENT_REGISTRY:
        raise ValueError(f"Unknown student model: {name}")
    return STUDENT_REGISTRY[name](**cfg)
