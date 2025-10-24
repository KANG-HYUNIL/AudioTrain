---
applyTo: '**'
---
Step 1·2 작업을 위한 “명령문” 사양(다른 에이전트에게 전달용)
공통 목표
P0를 위해 NSynth로 학생 모델 단독 → Teacher 통합 KD → 프루닝 전단계까지 가동 가능한 엔드투엔드 최소 파이프라인 확립.
모든 실험은 MLflow에 기록. 설정은 Hydra YAML로 관리.
Step 0. 준비(전제 조건)

[실행] uv 가상환경 생성/활성화. torch 3종은 GPU 빌드(또는 CPU)로 설치. 나머지는 requirements.txt로 설치.
[실행] ffmpeg 설치 및 torchaudio 로딩 테스트 통과.
[확인] mlflow ui 실행 확인(mlflow ui) 후 브라우저 접근.
완료 기준

python -c "import torch;print(torch.cuda.is_available())"가 True(GPU) 또는 False(CPU)로 명확.
mlflow ui가 열리고 빈 실험 페이지가 보임.
Step 1. NSynth + 학생 단독 파이프라인 구축
목표

NSynth 서브셋(≤1GB) 다운로드→Mel 변환→학생(MobileNet) 단독 학습/검증→MLflow 로깅까지 한 번에 실행.
작업 항목
[코드] NSynth.py
download_NSynth(root, classes, max_total_size_mb): Zenodo에서 필요한 파일만 받도록 부분 다운로드 구현.
get_items(split="train/val"): (path, label) 목록 생성. 클래스 인덱스 매핑 저장.
[코드] mel_pipeline.py
파이프라인: load with torchaudio → resample(16k~32k) → mono → MelSpectrogram → Log-Mel → 정규화(Tensor[C, F, T]).
파라미터는 Hydra로(from aug.yaml) 주입.
[코드] augment.py
최소: SpecAug(Time/Freq mask) on Log-Mel. 이후 Noise/Gain/RIR/코덱는 비활성 기본값으로 스위치만 준비.
[코드] student_mobilenet.py
torchvision MobileNetV2 또는 MobileNetV3-Small 로드, width_mult 설정.
최종 classifier를 num_classes에 맞게 교체. 출력은 로짓. NSynth은 멀티클래스 기준으로 CE 사용.
[코드] metrics.py
멀티클래스 Acc, macro-F1 구현.
[코드] loops.py
학생 단독 학습 루프: AMP 지원, 평가 루프, 체크포인트(.pt) 저장.
MLflow 로깅 훅: 파라미터(config 스냅샷), 학습/검증 손실/지표, 체크포인트 업로드.
[코드] train_kd.py
Hydra @hydra.main로 configs/{data,model,train,aug}.yaml 읽고 위 요소 조립.
[설정] configs/*
data.yaml: dataset=NSynth, root=./data, classes=[소수 악기], max_total_size_mb=1000, split 비율.
model.yaml: student={arch: mobilenet_v3_small, width_mult: 0.75, num_classes: K}.
train.yaml: epochs=5~20, batch_size=32, lr=3e-4, optimizer=adamw, amp=true, num_workers=2, seed, device=auto.
aug.yaml: sr=16000, n_fft=1024, hop=160, n_mels=128, fmin=20, fmax=8000, specaug={time_mask:02, freq_mask:02}.
[검증] 샌티 체크
작은 서브셋(예: 클래스 3개×각 50샘플)으로 1~3epoch 오버핏되는지 확인.
[로깅] MLflow
실험명: NSynth-Student-Baseline. 파라미터/지표/ckpt 아티팩트 기록.
실행 커맨드(예시)

uv 환경 활성화 후:
python -m scripts.train_kd data.dataset=NSynth data.root=./data data.classes="[violin,flute,clarinet]" train.epochs=5
완료 기준(DoD)
학습/검증 루프가 에러 없이 돌고, MLflow에 런이 생성되며 ckpt가 업로드.
작은 서브셋에서 train acc/f1이 빠르게 상승(오버핏 신호).
Step 2. Teacher 통합 + KD 학습
목표

PaSST 또는 AST를 Teacher로 불러와 KD 적용. KD가 학생 단독 대비 검증 지표를 향상시키는지 확인.
작업 항목
[코드] teacher_passt.py 또는 teacher_ast.py
HF/timm에서 사전학습 체크포인트 로드.
입력: Log-Mel 또는 모델 요구 스펙(채널/패치 크기) 정합. feature_dim 확인 후 프로젝트 클래스 수로 헤드 교체.
기본 freeze=True, 필요 시 마지막 몇 층만 unfreeze 옵션.
[코드] kd_loss.py
KD 손실: kd = (T^2)KLDiv(log_softmax(s/T), softmax(t/T)); total = alphakd + (1-alpha)*CE(hard).
하이퍼: T=2~4, alpha=0.7 기본.
[코드] loops.py
kd_mode 플래그 추가: forward에서 teacher(same input) 로짓 t 얻어 KD 혼합 손실 계산.
MLflow에 KD 관련 파라미터(T, alpha, freeze 여부) 기록.
[코드] profile_macs.py
thop으로 학생 모델 Params/MACs 측정. 결과를 MLflow 아티팩트/메트릭으로 저장.
[설정] model.yaml
teacher={name: passt_base|ast_base, checkpoint: <hf-id-or-path>, freeze: true}
kd={enabled: true, T: 3.0, alpha: 0.7}
[실험] 비교
동일 NSynth 서브셋에서 학생 단독 vs KD 학습 비교(epochs/seed 동일).
최적 threshold(멀티클래스는 불필요, 멀티라벨 전환 시만) 규칙 명시.
실행 커맨드(예시)

python -m scripts.train_kd model.teacher.name=passt_base model.kd.enabled=true model.kd.T=3.0 model.kd.alpha=0.7
python -m tools.profile_macs model.student.arch=mobilenet_v3_small model.student.width_mult=0.75
완료 기준(DoD)

KD 모드에서 학습/검증이 정상 수행되고 MLflow에 KD 파라미터/지표가 기록.
학생 단독 대비 검증 Acc/F1이 동일 또는 향상(소규모 NSynth 기준 소폭↑ 기대).
MACs/Params 리포트가 MLflow에 아티팩트로 저장.
추가 메모

이후 Step 3로 프루닝(pruning.py + prune_and_fineturn.py)과 재-KD 파인튜닝을 진행. 구조적 채널 프루닝 후 prune.remove 적용, MACs/Params 개선과 지표 유지 여부를 MLflow로 확인.