# Audio KD + Pruning Mini Project

이 리포는 악기 인식(멀티라벨) + 소스 분리(stems) 엔드투엔드를 4주 내 완주하기 위한 스캐폴드입니다. 핵심은 Teacher–Student 지식 증류(KD), 프루닝, 경량 백본(MobileNet)로 Params/MACs를 줄이며 성능을 유지하는 것입니다.

## 목표

- Stage-1: TinySOL로 전처리→KD→프루닝→재-KD→로그→추론까지 완료
- Stage-2: 예측 클래스 기반 stem 분리(Demucs/Spleeter 중 하나 연결)
- Stage-3(선택): 문서/결과 표·그림 정리

## 폴더 구조

```
audio-kd-prune/
  configs/              # Hydra YAML: data.yaml, model.yaml, train.yaml, aug.yaml
  datasets/             # TinySOL/IRMAS/OpenMIC 서브셋 다운로드/로더
  dataloaders/          # torchaudio Log-Mel, 증강 파이프라인
  models/               # Teacher(PaSST/AST), Student(MobileNet)
  training/             # KD loss, loops, pruning, metrics
  tools/                # MACs/Params, ONNX export
  scripts/              # train_kd, prune_and_fineturn, infer_classify, separate_stems
  notebooks/            # EDA/검증 노트북
  README.md  LICENSE  requirements.txt
```

## 우선순위

- P0: TinySOL 엔드투엔드
- P1: IRMAS/OpenMIC 부분셋 멀티라벨 1회 실험
- P2: 분리 스크립트 연결(Demucs/Spleeter)
- P3: 문서/결과 도표 정리 및 리팩토링

## 설치 안내(지금은 설치하지 마세요)

의존성 목록은 `requirements.txt`에 정리되어 있습니다. 추후 uv/venv 등 가상환경에서 설치합니다.

## 데이터

- TinySOL(Zenodo) — 소용량, 단일 음표 솔로 악기
- IRMAS(UPF/Zenodo) — 지배적 악기 인식(멀티라벨)
- OpenMIC-2018(Zenodo) — 다중 라벨, 10초 클립

각 데이터셋은 `datasets/`의 스크립트로 부분 다운로드(≤1GB) 옵션을 제공합니다.

## 저작권/윤리 고지

- 공개 배포가 제한된 음원/모델 가중치를 저장소에 업로드하지 마세요.
- Spleeter/Demucs 사용은 연구/개인 범위로 제한합니다. 각 라이선스 준수 필수.
