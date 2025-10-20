---
applyTo: '**'
---
역할: 너는 VS Code 안에서 동작하는 내 AI 페어프로그래머야.
목표: 4주 내 제출 가능한 오디오 기반 악기 인식·분리 미니 프로젝트를 설계·구현·문서화한다. 석사 지원 포트폴리오용이며, 교사–학생(KD) + 프루닝을 실제로 적용해 경량화·일반화를 경험하는 것이 핵심이다.

0) 필수 제약

데이터셋: 정상 링크/라이브러리로 내려받을 수 있어야 하며, 최초 다운로드 총량 ≤ 1GB를 우선한다. (필요시 부분 다운로드/부분 사용으로 제한)

오디오 파이프라인: 입력은 wav/mp3 등 일반 오디오. torchaudio로 로드 → 리샘플/채널정규화 → MelSpectrogram/Log-Mel → 증강(RIR/코덱/노이즈·Time/Freq mask 등) → Tensor. 
PyTorch Documentation
+2
PyTorch Documentation
+2

모델 전략: Teacher(대형, 공개 사전학습) + Student(경량 CNN, MobileNet류) → 지식 증류(KD) → 프루닝 → 재-KD. (KD는 Hinton 정석 사용: temperature/soft-target + CE 혼합) 
Universitat Pompeu Fabra

문서화·로그: MLflow(또는 W&B 중 MLflow 우선)로 매 실험의 하이퍼파라미터, 정확도, F1/mAP, 손실, Params(개수), 모델 크기(MB), MACs를 기록. 설정은 Hydra로 YAML 관리. 
Hydra
+4
mlflow.org
+4
mlflow.org
+4

오픈소스 모델을 사용하고, 저작권/라이선스를 준수한다. (임의 상용 음원 업로드/배포 금지. 분리는 연구/개인용만. 필요시 Spleeter/Demucs 사전학습 모델 사용) 
GitHub
+1

1) 데이터 소스(1GB 이내 우선)

TinySOL(작은 솔로 음표, 14종 악기·모노포닉, 수백 MB급) → 파이프라인/증강/다중클래스 분류 빠른 성공용. 
Zenodo
+1

IRMAS(지배적 악기 인식, 3초 클립·다중 장르) → 서브셋 다운로드 스크립트로 클래스 3–5개/파일 수 제한하여 ≤1GB 유지. 
Universitat Pompeu Fabra
+2
Zenodo
+2

OpenMIC-2018(다중 라벨·10초, FMA CC 음원) → 역시 메타데이터 기반 서브셋 추출로 ≤1GB. (완전체는 큼) 
Zenodo
+1

(참고) ESC-50는 600–770MB 수준이지만 환경음이라 악기 중심과는 다름—디버그/파이프라인 검증용 백업. 
GitHub
+1

요청 사항:

datasets/ 모듈에 다운로더(Zenodo/HF/직접 링크), 부분 샘플러(클래스·파일 수·길이 제한) 포함.

dataloaders/에 torchaudio 로더 + MelSpectrogram 변환 + 증강(torch-audiomentations/audiomentations, torchaudio RIR/코덱) 파이프라인 구현. 
GitHub
+2
iver56.github.io
+2

2) 모델·학습 구성

Teacher(택1+)

PaSST(Audio Transformer, Patchout) 또는 AST(HuggingFace hub 모델) 사전학습 체크포인트 활용. 
GitHub
+2
Hugging Face
+2

대체: PANNs(Cnn14) 사전학습(오디오 태깅 기반). 
GitHub

Student(경량 Backbone)

torchvision.models.mobilenet_v2(width_mult=0.5~1.0) 또는 MobileNetV3-Small. (Depthwise Separable Conv) 
GitHub

학습 방식

손실: KD_loss*T^2*alpha + CE*(1-alpha)(KL-div on softened probs). 기본값 T=2~4, alpha=0.7. 
Universitat Pompeu Fabra

프루닝: PyTorch torch.nn.utils.prune로 글로벌 L1 비구조 + 채널 구조적 혼합, 이후 prune.remove로 영구 적용. 
Zenodo
+1

재-KD: 프루닝 후 성능 보정.

지표

분류: multi-label(IRMAS/OpenMIC)일 때 mAP/micro-F1, multi-class면 Acc/F1.

복잡도: Params, 모델 파일 크기(MB), MACs(thop 등) 기록.

증강

Time/Freq mask(SpecAug), mixup(멀티라벨 시 label mix), RIR/코덱(torchaudio 튜토리얼), Gain/Noise 등. 
PyTorch Documentation

3) 2단계(분리) 파이프라인

사전학습 분리기로 Demucs(HTDemucs v4) 또는 Spleeter(4/5 stems) 사용. 로컬 오디오 입력 → stems 출력. 훈련 불필요.

매핑: 예) 예측 클래스에 drums/bass/piano가 있으면 해당 stem 추출, guitar는 4-stem 기준 ‘other’에 포함될 수 있음을 로그에 명시. 
GitHub
+1

4) 리포·문서화·실행

프로젝트 구조(예시)

audio-kd-prune/
  configs/ (Hydra YAML: data.yaml, model.yaml, train.yaml, aug.yaml)
  datasets/ (tinysol.py, irmas.py, openmic_subset.py, download_utils.py)
  dataloaders/ (mel_pipeline.py, augment.py)
  models/ (teacher_passt.py, teacher_ast.py, student_mobilenet.py)
  training/ (loops.py, kd_loss.py, pruning.py, metrics.py)
  tools/ (profile_macs.py, export_onnx.py)
  scripts/ (train_kd.py, prune_and_fineturn.py, infer_classify.py, separate_stems.py)
  notebooks/ (EDA, sanity-check)
  README.md  LICENSE  requirements.txt
etc..

MLflow 로깅: 실험명/런명·하이퍼파라미터·지표·아티팩트(혼동행렬/PR 곡선·모델 ckpt·config 스냅샷)를 자동 기록. (PyTorch Quickstart/Best practices 준수) 
mlflow.org
+1

Hydra: 모든 하이퍼/경로/증강옵션을 YAML로 관리하고 CLI override 지원. 
Hydra
+1

5) 수용 기준(완료 정의, Definition of Done)

Stage-1(필수):

(a) 교사-학생 KD 학습이 성공적으로 완료되고,

(b) 프루닝→재-KD 후에도 베이스라인 대비 Params/MACs↓ & 성능 손실 최소

(c) MLflow에 실험 로그와 결과 표(Acc/F1/mAP + Params + MACs)가 남는다.

Stage-2(필수):

입력 오디오에 대해 stems 파일을 생성하고, 예측 클래스에 맞는 stem만 개별 wav로 제공. (매핑/한계 기록)

Stage-3(선택):

각 stem에 대해 사보/전사는 시간·우선순위상 옵션으로 남김.

문서화:

README에 데이터 준비(≤1GB 서브셋)·학습·추론·분리·지표·제약·한계 명시.

결과는 표·그림으로 요약.

윤리·법적 고지:

저작권 콘텐츠는 공개 배포 금지. 분리 도구 사용은 연구/개인 범위로 제한. (Spleeter 관련 공개 기사 참고) 
Pitchfork