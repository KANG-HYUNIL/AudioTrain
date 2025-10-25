## 목표

이 문서는 사용자가 참고한 논문(https://arxiv.org/pdf/2410.20775)을 바탕으로, 현재 레포지토리(`run_training.py` 기반 DCASE2024 Task1 baseline)를 확장·재현하고 추가 실험을 진행하기 위한 단계별 명령문(실행 계획)을 제공합니다. 핵심 목표는 논문에서 사용한 기법들(병렬 kernel + re-parameterize, Transformer ensemble → knowledge distillation, progressive pruning + 재학습 등)을 직접 구현·체험하고, 그 기반에서 다양한 변형/하이퍼파라미터 실험을 수행하여 비교·평가하는 것입니다.

요구사항(간단)
- baseline 코드를 크게 바꾸지 않고 확장 가능할 것
- 다양한 모델/하이퍼파라미터를 편리하게 조정할 수 있을 것 (권장: 최소한의 config 체계 또는 Hydra 채택)
- 모든 실험 결과는 비교 가능하게 기록되어야 함 (성능, 파라미터 수, MACs, epoch별 로그, 체크포인트)
- 모델 복잡도(파라미터 수, MACs)는 `helpers/nessi.py`를 재사용해 자동 측정할 것

## 핵심 구성 요소와 파일 위치(참고)
- 진입점: `run_training.py` (train / evaluate 분기)
- 데이터 로더: `dataset/dcase24.py` (중요: `dataset_dir` 전역 변수 설정 필요)
- 모델: `models/baseline.py` (get_model 인터페이스) — 여기에 Rep-style 병렬 conv/재매개변수화(RepVGG 계열)를 연결
- Mel 전처리: `models/mel.py` 또는 `PLModule.mel_forward` (run_training.py)
- MixStyle: `helpers/utils.py::mixstyle`
- 복잡도 계산: `helpers/nessi.py::get_torch_size`

## 제안하는 단계별 작업(우선순위 순)

1) 준비 — baseline을 빠르게 돌려보기
  - README대로 환경 구성(권장: conda python=3.10, PyTorch 별도 설치)
  - `dataset/dcase24.py` 에서 `dataset_dir`를 실제 경로로 설정(또는 임시로 small subset으로 테스트).
  - 빠른 smoke test (CPU 디버그 가능):

    ```powershell
    python run_training.py --subset=5 --batch_size=8 --num_workers=0
    ```

  - 목적: baseline이 정상 동작하고 WandB(있다면)를 통해 로그가 남는지 확인.

2) 재현(핵심) — 논문 핵심 기법 중 하나씩 추가하여 검증
  A. 병렬 kernel + re-parameterization (training: multi-branch conv, inference: fusion)
    - 구현 전략: `models/rep_conv.py` 같은 모듈을 추가하여 병렬 브랜치(1x1, 3x1, 1x3 등) 형태로 forward를 구현하되, inference 시점엔 weight fusion API(fuse_conv)를 제공.
    - 통합 위치: `models/baseline.py`의 Block 또는 Network에서 옵션으로 대체 가능한 블록으로 교체.
    - 검증: training/inference 결과가 동일(±tiny)한지, inference 모델에서 MACs/파라미터 수가 줄어드는지 확인.

  B. Transformer 기반 teacher ensemble → Knowledge Distillation
    - 전략: 여러 개의 pre-trained Transformer teacher 모델(외부 또는 직접 학습)을 준비한 뒤, soft-target(temperature T) + CE 조합으로 student를 학습.
    - 구성: `run_training.py`에 KD 옵션(예: --kd_alpha, --kd_temp, --teacher_ckpts)을 추가. KD loss: L = alpha * CE(student, label) + (1-alpha) * KD_loss(student_logits, teacher_logits, T)
    - teacher ensemble 구현: predict teacher logits per-batch, 평균 logits 후 softmax, 또는 logits sum.
    - 리소스: Transformer teacher들은 크기가 크므로 별도 GPU/사전학습 필요. 작은 실험은 1~2개 teacher로 시작.

  C. Progressive pruning + iterative retrain
    - 전략: pruning schedule을 설계(예: 전체 반복 수 R, 매 반복 prune ratio r_i). 각 단계:
      1. 현재 모델에서 중요도 기준(예: magnitude)으로 파라미터 제거(비율 r_i)
      2. 남은 모델 재학습(retrain) — 논문처럼 KD를 결합할 수도 있음
    - 자동화: `scripts/progressive_prune.py`를 추가해 반복 파이프라인(프루닝 → 체크포인트 → 재학습)을 구현.
    - 체크포인트 정책: 각 프루닝 단계마다 결과(accuracy, params, MACs, sparsity 등)를 CSV/JSON으로 기록.

3) 실험 인프라 및 설정
  - Config 시스템: 기존 CLI를 유지하되, 많은 하이퍼파라미터를 깔끔하게 관리하기 위해 다음 둘 중 하나 권장:
    1) 가벼운 방법: `configs/experiments/*.yaml` (단순 YAML 로드 + argparse와 병합). 기존 코드를 크게 변경하지 않음.
    2) (선택) Hydra 도입: `hydra-core` 설치 후 `configs/` 디렉토리로 전환. 이 경우 `run_training.py` 진입부만 최소 수정.
  - 권장 필드: model variant, repconv on/off, kd_alpha, kd_temp, teacher_ckpts, prune_schedule, batch_size, lr, sample_rate 등.

4) 실험 기록(로그/추적)
  - 메인: Weights & Biases 권장(이미 사용 중). config, MACs, 파라미터 수, train/val/test metric, pruning 단계별 메타데이터를 wandb에 저장.
  - 로컬 보조: `experiments/results.csv` 또는 `experiments/<exp_id>/info.json` 형태로 요약 저장 (columns: exp_id, timestamp, config path, macro_avg_acc, params, MACs, pruning_step, notes)
  - 자동 측정: training 전에 `nessi.get_torch_size(model, input_size)` 호출해 MACs/params를 wandb와 로컬 요약에 기록.

5) 재현성/검증
  - 각 실험은 (1) config/seed, (2) 체크포인트, (3) info.json(output metrics + MACs)을 함께 저장.
  - 테스트: `run_training.py --evaluate --ckpt_id=<wandb_id>` 로 예측 재현 가능.

6) 안전/비파괴 변경 원칙
  - `dataset_dir`는 현재 `dataset/dcase24.py`에서 import 시점에 assert로 검사됩니다. 실험 자동화 전까지는 이 동작을 유지하세요.
  - 새로운 옵션(예: dataset_dir를 env로 덮어쓰기)은 `dataset/dcase24.py`에 작은 분기 추가로 구현하되(예: os.getenv('TAU24_PATH')가 있으면 override), 기본 동작은 변경하지 마세요.

## 구현 세부 가이드(개발자 작업 목록)
1. 파일 추가/수정
   - `models/rep_conv.py` — Rep-style 병렬 conv 블록(훈련용 브랜치 + inference fuse 함수)
   - `models/baseline.py` — Block을 필요시 교체하도록 옵션화
   - `run_training.py` — KD 관련 CLI 옵션 추가, teacher logits 로딩/ensemble, KD loss 계산 hook 추가, pruning 파이프라인 호출 지점 추가
   - `scripts/progressive_prune.py` — pruning 스케줄러/도구 (prune + retrain 반복 자동화)
   - `configs/` (선택) — 실험별 YAML 모음(또는 Hydra)

2. 핵심 하이퍼파라미터(초기권장값)
   - KD: kd_alpha=0.5, kd_temp=4.0
   - Rep-conv: enable=True during training, fuse at export
   - Progressive Pruning: steps=3~5, prune_ratios=[0.3,0.3,0.3], retrain_epochs=[30,30,30]

3. 실험 예시 명령
   - baseline 학습(빠른):

     ```powershell
     python run_training.py --subset=5 --batch_size=16 --num_workers=0
     ```

   - RepConv 활성화(예: CLI 플래그 구현 시):

     ```powershell
     python run_training.py --subset=100 --repconv True --batch_size=128
     ```

   - KD로 학생학습(teacher ckpt 리스트 사용 가정):

     ```powershell
     python run_training.py --subset=100 --kd_alpha=0.6 --kd_temp=3.0 --teacher_ckpts="t1.ckpt,t2.ckpt"
     ```

   - Progressive pruning 자동 파이프라인(초안):

     ```powershell
     python scripts/progressive_prune.py --config configs/prune_schedule.yaml --base_ckpt last.ckpt
     ```

## 실험 결과 비교 항목(항상 기록할 것)
- macro_avg_acc (validation/test)
- class-wise acc (가능하면)
- device-wise acc (run_training.py가 이미 측정)
- 파라미터 수
- MACs
- 모델 사이즈(bytes) — inference 시 quantized/half-precision 시 값
- pruning 단계별 sparsity
- 학습 곡선(학습/검증 손실 및 정확도)

## 엔지니어링·테스트 품질 게이트
- 모든 변경은 빠른 smoke-test(1 epoch 또는 subset=5)로 동작 확인
- 변경 후 `nessi.get_torch_size`로 MACs/params 계산이 실패하면 즉시 중단

## 요청사항(사용자에게)
1) Hydra 도입 허용 여부: (예/아니오). 허용 시 `run_training.py`를 최소 수정해 바로 Hydra를 사용하도록 scaffold를 추가합니다.
2) WandB 사용 계속할지 여부(현재 기본값은 WandB). 로컬 CSV로만 기록할 경우 워크플로우를 약간 조정합니다.
3) Transformer teacher들은 직접 학습할지, 공개된 체크포인트를 사용할지 결정해 주세요.

---
이 명령문은 기존 `.github/copilot-instructions.md`와 충돌하지 않도록 설계되었습니다(그 파일은 에이전트 일반 워크플로우를 설명). 원하시면 위 작업들 중 하나를 선택해 바로 코드 패치를 만들어 드리겠습니다 (예: `models/rep_conv.py` 초안 구현, `scripts/progressive_prune.py` 초안 만들기, 또는 Hydra scaffold 추가). 어떤 작업을 먼저 해 드릴까요?

## 프로젝트 결정(영구 기록)
이 프로젝트에서 이후 모든 에이전트가 따를 기본 선택들을 아래에 기록합니다. 다른 변경을 원하면 이 섹션을 업데이트하세요.

- Hydra 도입: 예 — `run_training.py`는 최소한으로 수정하여 Hydra를 통한 `configs/*.yaml` 적용을 기본 실험 인터페이스로 사용합니다. (추가 변경 시 호환 모드로 기존 argparse 파라미터를 유지)
- 실험 기록/로그: Weights & Biases(기본) — WandB를 주 로깅/실험 추적 도구로 사용합니다. 로컬 보조로 CSV/JSON 요약(`experiments/`)을 병행합니다.
- Transformer teachers: 공개된 pre-trained Transformer 모델을 가져와서 타겟 데이터셋(예: TAU Urban Acoustic Scenes)으로 파인튜닝(fine-tuning)하여 teacher ensemble을 구성합니다. 학생(student) 모델은 KD(student <- ensemble) 방식으로 학습합니다.

위 세 가지 결정은 RESEARCH_PLAN.md와 `.github/copilot-instructions.md` 모두에 반영되어야 하며, 다른 에이전트가 자동화 작업을 할 때 우선적으로 따릅니다.
