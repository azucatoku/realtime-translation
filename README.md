# Real-Time Voice Chat Translator

A real-time voice chat translation application with **cross-recording speaker identification**.  
Each person records separately, and the system automatically recognizes who is speaking across multiple recordings using voice fingerprint comparison.

---

## Features

- **Cross-Recording Speaker Identification** -- Recognizes the same speaker across separate recordings using pyannote speaker embeddings and cosine similarity matching.
- **Automatic Speech Recognition** -- WhisperX (large-v3) with GPU acceleration and precise timestamp alignment.
- **Bidirectional Translation** -- Auto-detects Korean or English and translates to the other language via Google Translate.
- **Text-to-Speech** -- Generates translated audio with different voices per speaker using Microsoft Edge TTS.
- **Bauhaus-Style Chat UI** -- Speaker-colored chat bubbles (blue/yellow), bold typography, and a grid-pattern background.

---

## Architecture

```
Browser Mic (WebM)
      |
      v
FastAPI Backend (/chat/)
      |
      +-- [1] pydub: WebM -> WAV conversion
      +-- [2] WhisperX: Speech-to-Text + language detection
      +-- [3] WhisperX: Timestamp alignment
      +-- [4] pyannote/embedding: Extract voice fingerprint
      |       -> Compare with known speakers (cosine similarity >= 0.75)
      |       -> Assign existing or new speaker ID
      +-- [5] Google Translate + Edge TTS -> base64 audio
      |
      v
JSON Response -> Chat Bubbles with translated text + audio playback
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Speech-to-Text | WhisperX (large-v3, GPU) |
| Speaker Identification | pyannote/embedding (cosine similarity) |
| Translation | Google Translate (deep-translator) |
| Text-to-Speech | Microsoft Edge TTS |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## Project Structure

```
translation_app/
├── .gitignore
├── README.md
├── backend/
│   ├── .env                  # HF_AUTH_TOKEN (not tracked by git)
│   ├── main.py               # FastAPI backend server
│   ├── translator_cli.py     # Standalone CLI translator (mic -> translate -> speaker)
│   ├── test_diarization.py   # Speaker diarization verification script
│   ├── make_test.py          # Test audio generator (2-speaker simulation)
│   └── requirements.txt      # Python dependencies
└── frontend/
    ├── index.html            # Chat UI structure
    ├── style.css             # Bauhaus design system
    └── app.js                # Recording, API calls, chat rendering
```

---

## Setup

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA support
- Conda (recommended)
- FFmpeg installed on system (`winget install Gyan.FFmpeg` on Windows)

### Installation

```bash
# 1. Create and activate conda environment
conda create -n trans_env python=3.10 -y
conda activate trans_env

# 2. Install Python dependencies
cd backend
pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# 3. Create .env file with your HuggingFace token
echo HF_AUTH_TOKEN=your_huggingface_token > .env
```

### HuggingFace Model Access

You must accept the terms for the following models on HuggingFace:

- [pyannote/embedding](https://huggingface.co/pyannote/embedding)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Running

**Terminal 1 -- Backend:**
```bash
conda activate trans_env
cd backend
uvicorn main:app --reload
```

**Terminal 2 -- Frontend:**
```bash
cd frontend
python -m http.server 8080
```

Open `http://localhost:8080` in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat/` | Upload audio -> STT + speaker ID + translation + TTS |
| POST | `/reset_speakers/` | Clear all speaker memory on the server |

### POST /chat/

**Request:** `multipart/form-data` with a `file` field containing a `.webm` audio blob.

**Response:**
```json
{
  "dialogues": [
    {
      "speaker": "SPEAKER_00",
      "original": "안녕하세요",
      "translated": "Hello",
      "language": "ko",
      "audio_b64": "base64-encoded-mp3..."
    }
  ]
}
```

---

## How Speaker Identification Works

1. Each recording's full audio is passed through the `pyannote/embedding` model to extract a 256-dimensional voice fingerprint vector.
2. This vector is compared against all previously seen speakers using **cosine similarity**.
3. If the best match score is **>= 0.75**, the recording is assigned to that existing speaker. The stored embedding is updated with a moving average (80% old + 20% new) for increasing accuracy over time.
4. If no match exceeds the threshold, a new speaker ID is registered.
5. The `RESET SPEAKERS` button clears all stored embeddings to start fresh.

---

## License

This project is for educational and personal use.
