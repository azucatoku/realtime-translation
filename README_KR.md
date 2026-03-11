# 실시간 음성 채팅 번역기

**녹음 간 화자 식별** 기능을 갖춘 실시간 음성 채팅 번역 애플리케이션.  
각 사람이 따로 녹음하면, 시스템이 음성 지문 비교를 통해 여러 녹음에 걸쳐 누가 말하고 있는지 자동으로 인식합니다.

---

## 주요 기능

- **녹음 간 화자 식별** -- pyannote 임베딩으로 음성 지문을 추출하고, 코사인 유사도 매칭(임계값 0.75)으로 동일인 여부를 판별합니다.
- **자동 음성 인식** -- WhisperX (large-v3)를 GPU 가속으로 구동하며, 정밀한 타임스탬프 정렬을 수행합니다.
- **양방향 자동 번역** -- 한국어 또는 영어를 자동 감지하여 상대 언어로 번역합니다 (Google Translate).
- **음성 합성 (TTS)** -- 화자별 다른 목소리로 번역된 음성을 생성합니다 (Microsoft Edge TTS).
- **바우하우스 스타일 채팅 UI** -- 화자별 색상(파랑/노랑)으로 구분되는 말풍선, 굵은 타이포그래피, 그리드 배경.

---

## 아키텍처

```
브라우저 마이크 (WebM)
      |
      v
FastAPI 백엔드 (/chat/)
      |
      +-- [1] pydub: WebM -> WAV 변환
      +-- [2] WhisperX: 음성 인식 + 언어 감지
      +-- [3] WhisperX: 타임스탬프 정밀 정렬
      +-- [4] pyannote/embedding: 음성 지문 추출
      |       -> 기존 화자와 코사인 유사도 비교 (>= 0.75)
      |       -> 기존 화자 ID 배정 또는 신규 등록
      +-- [5] Google Translate + Edge TTS -> base64 오디오
      |
      v
JSON 응답 -> 번역 텍스트 + 오디오 재생이 포함된 채팅 말풍선
```

---

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| 백엔드 | FastAPI + Uvicorn |
| 음성 인식 | WhisperX (large-v3, GPU) |
| 화자 식별 | pyannote/embedding (코사인 유사도) |
| 번역 | Google Translate (deep-translator) |
| 음성 합성 | Microsoft Edge TTS |
| 프론트엔드 | Vanilla HTML / CSS / JavaScript |

---

## 프로젝트 구조

```
translation_app/
├── .gitignore
├── README.md               # 영문 README
├── README_KR.md            # 한글 README
├── backend/
│   ├── .env                # HF_AUTH_TOKEN (git 추적 제외)
│   ├── main.py             # FastAPI 백엔드 서버
│   ├── translator_cli.py   # 독립 실행형 CLI 번역기 (마이크 -> 번역 -> 스피커)
│   ├── test_diarization.py # 화자 분리 검증 스크립트
│   ├── make_test.py        # 테스트 오디오 생성기 (2화자 시뮬레이션)
│   └── requirements.txt    # Python 의존성 목록
└── frontend/
    ├── index.html          # 채팅 UI 구조
    ├── style.css           # 바우하우스 디자인 시스템
    └── app.js              # 녹음, API 호출, 채팅 렌더링
```

---

## 설치 방법

### 사전 요구 사항

- Python 3.10
- CUDA 지원 NVIDIA GPU
- Conda (권장)
- 시스템에 FFmpeg 설치 (Windows: `winget install Gyan.FFmpeg`)

### 설치

```bash
# 1. Conda 가상환경 생성 및 활성화
conda create -n trans_env python=3.10 -y
conda activate trans_env

# 2. Python 의존성 설치
cd backend
pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# 3. HuggingFace 토큰을 담은 .env 파일 생성
echo HF_AUTH_TOKEN=본인의_허깅페이스_토큰 > .env
```

### HuggingFace 모델 접근 권한

아래 모델들의 이용약관에 동의해야 합니다:

- [pyannote/embedding](https://huggingface.co/pyannote/embedding)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### 실행

**터미널 1 -- 백엔드:**
```bash
conda activate trans_env
cd backend
uvicorn main:app --reload
```

**터미널 2 -- 프론트엔드:**
```bash
cd frontend
python -m http.server 8080
```

브라우저에서 `http://localhost:8080` 으로 접속합니다.

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/chat/` | 오디오 업로드 -> STT + 화자 식별 + 번역 + TTS |
| POST | `/reset_speakers/` | 서버의 화자 기억 전체 초기화 |

### POST /chat/

**요청:** `multipart/form-data` 형식, `file` 필드에 `.webm` 오디오 Blob 첨부

**응답:**
```json
{
  "dialogues": [
    {
      "speaker": "SPEAKER_00",
      "original": "안녕하세요",
      "translated": "Hello",
      "language": "ko",
      "audio_b64": "base64-인코딩된-mp3..."
    }
  ]
}
```

---

## 화자 식별 작동 원리

1. 각 녹음의 전체 오디오를 `pyannote/embedding` 모델에 통과시켜 256차원 음성 지문 벡터를 추출합니다.
2. 이 벡터를 이전에 등록된 모든 화자의 벡터와 **코사인 유사도**로 비교합니다.
3. 최고 유사도 점수가 **0.75 이상**이면 해당 기존 화자로 배정합니다. 저장된 임베딩은 이동평균(기존 80% + 신규 20%)으로 업데이트되어 시간이 지날수록 정확도가 향상됩니다.
4. 임계값을 초과하는 매칭이 없으면 새로운 화자 ID를 등록합니다.
5. `RESET SPEAKERS` 버튼을 누르면 저장된 모든 임베딩이 삭제되어 처음부터 다시 시작합니다.

---

## 한계점

- **화자 메모리 휘발성.** 모든 화자 임베딩은 서버 메모리(RAM)에 저장됩니다. 백엔드 서버를 재시작하면 모든 화자 데이터가 사라집니다. 영구 저장소가 없습니다.
- **녹음당 1인 가정.** 각 녹음은 한 사람이 말하는 것으로 처리됩니다. 한 녹음에 두 사람이 말하면 전체 클립에 하나의 화자 ID만 할당됩니다.
- **짧은 오디오 정확도 저하.** 2~3초 미만의 녹음은 음성 인식과 화자 임베딩의 신뢰도가 떨어질 수 있습니다. WhisperX 역시 30초 미만 오디오에서는 언어 감지 정확도가 낮아진다고 경고합니다.
- **한국어와 영어만 지원.** 현재 번역 파이프라인은 한국어-영어 간 번역만 지원합니다. Whisper가 감지한 다른 언어는 기본적으로 영어 번역으로 처리됩니다.
- **코사인 유사도 임계값 고정.** 화자 매칭 임계값(0.75)이 하드코딩되어 있습니다. 소음이 많은 환경이나 비슷한 목소리에서는 오인식이 발생할 수 있으며, 사용 환경에 따라 조정이 필요할 수 있습니다.
- **동시 사용자 미지원.** 화자 메모리가 전역(단일 딕셔너리)입니다. 여러 사용자가 동시에 서버에 접속하면 동일한 화자 풀을 공유하게 되어, 화자 식별이 되섞일 수 있습니다.
- **GPU 메모리 요구량.** 세 개의 AI 모델(WhisperX large-v3, pyannote embedding, pyannote diarization)이 동시에 로드되어 약 6~8GB의 VRAM이 필요합니다.
- **정렬 모델 매 요청 로드.** WhisperX 정렬 모델이 매 요청마다 로드/삭제되어 지연이 발생합니다. 대상 언어별 정렬 모델을 사전 캐싱하면 성능이 개선될 수 있습니다.
- **스트리밍 미지원.** 오디오가 완전히 녹음된 후에만 처리가 가능합니다. 실시간 스트리밍 전사 및 번역은 구현되어 있지 않습니다.

---

## 라이선스

이 프로젝트는 교육 및 개인 용도로 사용됩니다.
