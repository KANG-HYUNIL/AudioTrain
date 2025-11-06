아래 블록을 그대로 복사해 “명령문”으로 사용하세요. (인용/근거는 블록 뒤쪽에 첨부)

---

applyTo: "**"

역할: 너는 VS Code 안에서 동작하는 내 AI 페어프로그래머다.
목표: 4주 내 제출 가능한 “음원 → 악보(멀티트랙 MIDI/MusicXML)” 미니 프로젝트를 설계·구현·문서화한다. 석사 지원 포트폴리오용이며, 이후 KD/프루닝 등 경량화 실험은 별도 브랜치에서 진행한다.

공통 규범

* 모든 소스 주석과 Docstring은 영어로 작성한다(모든 함수/클래스에 Google-style 또는 NumPy-style Docstring 필수).
* 설정은 Hydra(YAML)로 구성하고, 모든 실행/산출물/메트릭은 MLflow로 기록한다.
* 오디오 내부 표현 규약: `float32`, peak-normalized `[-1, 1]`, 기본 mono. 모듈별 요구 SR이 다르면 “모듈 내부에서” 책임지고 리샘플한다(Detector, Separator, Transcriber 각각의 target_sr은 설정으로 명시).
* 라이선스 준수: 공개 사전학습 모델(Demucs/Basic Pitch/music21 등)만 사용. 업로드 음원은 연구/개인 용도로만 처리한다.

폴더 구조(권장)

```
audio2score/
  configs/
    pipeline.yaml
    detector/*.yaml
    separator/*.yaml
    transcriber/*.yaml
    post/*.yaml
  src/
    audio_io.py
    detection.py
    separation.py
    transcription.py
    packaging.py
    pipeline.py
  ui/
    app_gradio.py
  requirements.txt
  README.md
```

Step 0. 환경 준비 (Colab/Local 공통)

* FFmpeg / PyTorch / torchaudio / demucs / basic-pitch / music21 / librosa / hydra-core / mlflow 설치.
* GPU 가용성 및 FFmpeg 버전, torchaudio 로더 동작을 프린트로 확인·로그한다.
* MLflow: Colab에서는 `mlflow.set_tracking_uri("file:./mlruns")`로 파일 기반 트래킹 사용. Local은 `mlflow ui`(선택).
* 완료 기준(DoD)

  * `python -c "import torch;print(torch.cuda.is_available())"` 결과 확인.
  * `import demucs, basic_pitch, music21, librosa` 성공.
  * MLflow run 생성 및 params/metrics/artifacts 기록 확인.

Step 1. 엔드투엔드 추론 파이프라인(P0)
목표: 단일 오디오 입력에 대해 “악기 감지 → 분리 → 전사 → 멀티트랙 MIDI(+선택 MusicXML)”를 생성하고 MLflow에 로그/아티팩트를 기록한다.

[코드] src/audio_io.py

* `load_audio(path, target_sr: int | None, mono: bool) -> (tensor, sr)`
* `save_audio(path, tensor, sr) -> None`
* SR/채널 정규화 유틸 포함(내부 포맷 유지).

[코드] src/detection.py

* `build_detector(name="panns|ast|passt", pretrained=True)`

  * 기본값: `panns`(AudioSet 라벨 기반 clipwise 확률).
* `detect_instruments(waveform, sr, cfg) -> dict[str, float]`

  * threshold + top-k 동시 적용, 라벨 서브셋(예: piano/guitar/violin/cello/flute/saxophone/trumpet/organ/harp/bass/drums/voice).
* 설정 예시

  ```
  detector:
    name: panns
    threshold: 0.2
    max_classes: 3
    classes: [piano, guitar, violin, cello, flute, saxophone, trumpet, organ, harp, bass, drums, voice]
  ```

[코드] src/separation.py

* Demucs는 **CLI 래퍼**로 호출(안정성 우선).

  * `separate_demucs(path_or_waveform, model_name="htdemucs") -> dict[stem_name, path]`
  * 실행 시간, 출력 경로, 파일 크기 로깅.
* `map_stems_to_instruments(stems: dict) -> dict[instrument_group, stem_path]`

  * 기본 매핑: `drums→drums`, `bass→bass`, `vocals→voice(옵션)`, `other→melodic`.
  * 필요 시 Spleeter(5 stems) 옵션 제공(설정 스위치).

[코드] src/transcription.py

* 백엔드

  * `Basic Pitch`: 범용 피치 악기 전사(빠르고 가벼움).
  * `Onsets & Frames`: 피아노 특화.
* I/O 정책: 각 백엔드 내부에서 자체 요구 SR로 리샘플.
* 함수

  * `transcribe_basic_pitch(stem_path) -> midi.MidiFile`
  * `transcribe_onsets_frames(waveform, sr) -> midi.MidiFile`
  * `pick_transcriber(instrument) -> callable` (policy: piano→OnF, 그 외→Basic Pitch; drums는 P1에서 처리 혹은 skip)

[코드] src/packaging.py

* `merge_tracks(track_dict, tempo=None, metadata=None) -> music21.stream.Score`
* `export_midi(score, output_path) -> Path`
* `export_musicxml(score, output_path) -> Path`
* GM Program 매핑 테이블 포함(피아노=1, 베이스=33 등), 드럼은 채널 10 사용. 트랙명/순서/메타데이터(title, detector summary)를 포함.

[코드] src/pipeline.py

* `run_pipeline(input_audio, cfg) -> (artifacts: dict, stats: dict)`

  * 단계별 타이밍(ms), note_count, 파일 크기, 유효 노트 비율 등 수집.
  * MLflow: `experiment=audio2score`, `run_name=<detector-separator-transcriber>`

    * `log_params`: 모델/버전/threshold/SR/장치/그리드
    * `log_metrics`: step_time, total_time, note_count, filesizes
    * `log_artifacts`: 원본/분리 WAV, per-track MIDI/XML, 로그 JSON
* 실패 대비(fallback)

  * 분리 실패: 원본 단일 트랙을 그대로 전사.
  * 감지 결과 없음: 기본 `piano`로 전사해 산출물 보장.
  * 전사 실패: 백엔드 페일오버(OnF→Basic Pitch 등).

[설정] configs/pipeline.yaml (핵심)

```
io:
  input_path: /content/input.wav
  output_dir: /content/outputs
  device: cuda
  default_sr: 44100

detector:
  name: panns
  threshold: 0.2
  max_classes: 3
  classes: [piano, guitar, violin, cello, flute, saxophone, trumpet, organ, harp, bass, drums, voice]

separator:
  name: demucs
  model: htdemucs
  stems: [drums, bass, vocals, other]
  map_stems:
    piano: other
    guitar: other
    bass: bass
    drums: drums
    voice: vocals
    strings: other

transcriber:
  default_backend: basic_pitch
  piano_backend: onsets_frames
  output: [midi, musicxml]

post:
  tempo: librosa           # librosa | essentia
  quantize: "1/16"
  velocity: 64
  gm_table: default
```

[검증] 샘플 오디오(30–60초) 2–3개 실행 → 멀티트랙 MIDI 생성 및 MLflow 기록 확인.

Step 2. 품질 개선(P1)

* 템포/박자 추정: `librosa.beat.beat_track` 기반 tempo & beat 이벤트 생성.
* 양자화: grid(1/8, 1/16)로 노트 시작/길이 스냅, 최소 길이 임계치로 유실 방지.
* 벨로시티: 전역 노멀라이즈 후 악기군별 커브 스케일링(피아노는 다이내믹 넓게 등).
* GM Program/트랙 메타데이터: detector 확률과 stem 매핑 충돌 시 우선순위(예: detector confidence 우선) 규칙 정의.
* Gradio UI(간단 데모)

  * 입력: `gr.Audio(type="filepath")`
  * 출력: `gr.File` 또는 `gr.DownloadButton`으로 MIDI/XML 다운로드 링크 제공
  * 예제 샘플은 `gr.Examples`로 노출
* DoD

  * 임의 업로드 오디오 → 브라우저 UI에서 멀티트랙 MIDI 다운로드 가능
  * MLflow에 P1 관련 params/metrics 추가 기록(tempo_estimator, grid, velocity_policy 등)

Step 3. (선택) 학습/미세튜닝

* Detector 임계값 조정/미세튜닝(소규모 데이터셋).
* 특정 악기 전사 모델 파인튜닝(데이터 준비 후).
* KD/프루닝/최적화는 별도 브랜치에서 프로젝트 범위 밖으로 분리 관리.

실행 예(개념)

* Colab: `from src.pipeline import run_pipeline; run_pipeline("song.wav", cfg)`
* Local(옵션): `python -m scripts.infer_pipeline io.input_path=... io.output_dir=...`

UI/문서/로그 규범

* UI: 결과 파일은 고정된 출력 폴더에 저장하고, UI는 경로를 직접 다운로드 가능하도록 반환.
* 문서: README에 “요구 환경, 빠른 시작, 한계(드럼 전사는 P1 이후), 라이선스”를 명확히 기재.
* 로그: 모든 실행은 동일한 MLflow Experiment(`audio2score`) 하위에 남기고, run_name은 `<detector>-<separator>-<transcriber>` 규칙을 따른다.

---

참고·근거

* Demucs: 음악 소스 분리(드럼/베이스/보컬/기타 등), `htdemucs` 모델 제공과 CLI 사용 예시. ([GitHub][1])
* Basic Pitch: 스포티파이의 오디오→MIDI(피치 벤드 포함) 오픈소스 전사 라이브러리/데모. ([GitHub][2])
* PANNs/panns-inference: AudioSet 사전학습 오디오 태깅 모델과 쉬운 추론 API. ([GitHub][3])
* music21: MIDI/MusicXML import/export 지원, MusicXML 변환/쓰기 관련 문서. ([music21.org][4])
* librosa: `beat_track`로 비트·템포 추정(동적 프로그래밍 기반). ([librosa.org][5])
* Gradio: `Audio`, `File`, `DownloadButton`, `Interface` 컴포넌트로 업/다운로드 UI 구성. ([Gradio][6])
* MLflow: Tracking API로 params/metrics/artifacts 로깅, 파일 기반 tracking_uri 예시. ([mlflow.org][7])
* Hydra 1.3: Defaults List/구성 합성으로 프로파일링 가능한 설정 관리. ([hydra.cc][8])

필요 시, 위 명령문에 포함된 YAML/함수 시그니처/로그 키 이름은 그대로 구현 표준으로 간주한다.

[1]: https://github.com/facebookresearch/demucs?utm_source=chatgpt.com "facebookresearch/demucs: Code for the paper Hybrid ..."
[2]: https://github.com/spotify/basic-pitch?utm_source=chatgpt.com "GitHub - basic-pitch: Audio-to-MIDI Converter w/ Pitch Bend"
[3]: https://github.com/qiuqiangkong/panns_inference?utm_source=chatgpt.com "qiuqiangkong/panns_inference"
[4]: https://www.music21.org/music21docs/usersGuide/usersGuide_08_installingMusicXML.html?utm_source=chatgpt.com "Installing MusicXML Readers and File Formats (1) — music21 ..."
[5]: https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html?utm_source=chatgpt.com "librosa.beat.beat_track — librosa 0.11.0 documentation"
[6]: https://www.gradio.app/docs/gradio/audio?utm_source=chatgpt.com "Gradio Audio Docs"
[7]: https://mlflow.org/docs/latest/ml/tracking/?utm_source=chatgpt.com "MLflow Tracking"
[8]: https://hydra.cc/docs/1.3/tutorials/basic/your_first_app/defaults/?utm_source=chatgpt.com "Selecting default configs"
