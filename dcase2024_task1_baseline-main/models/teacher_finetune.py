# Teacher 모델 다운로드 및 finetune 모듈
import torch
from transformers import DeiTForImageClassification, ViTForImageClassification
from torch.utils.data import DataLoader

TEACHER_MODEL_MAP = {
    "deit_base_patch16_224": DeiTForImageClassification,
    "vit_base_patch16_224": ViTForImageClassification,
}

def download_teacher_model(name, pretrained=True, num_classes=10):
    if name not in TEACHER_MODEL_MAP:
        raise ValueError(f"Unknown teacher model: {name}")
    if pretrained:
        model = TEACHER_MODEL_MAP[name].from_pretrained(f"facebook/{name}")
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = TEACHER_MODEL_MAP[name](config={"num_labels": num_classes})
    return model

# Finetune teacher 모델 (별도 진행)
def finetune_teacher(model, train_loader, val_loader, epochs=5, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).logits
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        # validation loop 생략(필요시 추가)
    return model
