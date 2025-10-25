# Progressive Pruning 자동 파이프라인 스크립트
import torch
import yaml
from helpers.kd_prune import progressive_prune_and_retrain
from models.student_loader import get_student_model

# config yaml 예시: prune_steps, prune_ratios, retrain_epochs, student_cfg

def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    student_cfg = cfg['student_cfg']
    model = get_student_model(student_cfg)
    # train_loader, val_loader는 사용자 환경에 맞게 준비 필요
    train_loader = ...
    val_loader = ...
    prune_steps = cfg['prune_steps']
    prune_ratios = cfg['prune_ratios']
    retrain_epochs = cfg['retrain_epochs']
    def train_fn(model, train_loader, val_loader, epochs):
        # 사용자 정의 학습 함수 연결
        return model
    history = progressive_prune_and_retrain(model, train_fn, train_loader, val_loader, prune_steps, prune_ratios, retrain_epochs)
    # 결과 저장
    torch.save(history, 'prune_history.pt')

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
