# KD, Pruning, Progressive Pruning & Iterative Re-study 모듈
import torch
import torch.nn.functional as F

# Knowledge Distillation Loss
class KDLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = torch.nn.CrossEntropyLoss()
    def forward(self, student_logits, teacher_logits, labels):
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        ce_loss = self.ce(student_logits, labels)
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss

# Pruning util (magnitude-based)
def prune_model(model, prune_ratio=0.3):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    import torch.nn.utils.prune as prune
    for module, param in parameters_to_prune:
        prune.l1_unstructured(module, name=param, amount=prune_ratio)
    return model

# Progressive Pruning & Iterative Re-study
def progressive_prune_and_retrain(model, train_fn, train_loader, val_loader, prune_steps, prune_ratios, retrain_epochs):
    history = []
    for step in range(prune_steps):
        model = prune_model(model, prune_ratio=prune_ratios[step])
        model = train_fn(model, train_loader, val_loader, epochs=retrain_epochs[step])
        # 성능, sparsity 등 기록
        sparsity = sum([torch.sum(m.weight == 0).item() for m in model.modules() if hasattr(m, 'weight')])
        total = sum([m.weight.numel() for m in model.modules() if hasattr(m, 'weight')])
        history.append({
            'step': step,
            'sparsity': sparsity / total,
            'model_state_dict': model.state_dict()
        })
    return history
