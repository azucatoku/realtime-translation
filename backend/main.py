import os
import gc
import site
import base64
import shutil
import tempfile
import pandas as pd

import torch
import torch.nn.functional as F
import whisperx
from pyannote.audio import Inference, Pipeline
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import edge_tts


# --- Windows GPU DLL Path Fix ---
try:
    for sp in site.getsitepackages():
        for nvidia_pkg in ["cublas", "cudnn"]:
            bin_path = os.path.join(sp, "nvidia", nvidia_pkg, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
except Exception as e:
    print(f"Warning mapping NVIDIA DLLs: {e}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --- Configuration ---
load_dotenv()
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", "")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE  = "float16" if DEVICE == "cuda" else "int8"
MODEL_SIZE    = "large-v3"
BATCH_SIZE    = 8
SIMILARITY_THRESHOLD = 0.75  # 코사인 유사도 임계값


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Device: {DEVICE} | Compute: {COMPUTE_TYPE} | Model: {MODEL_SIZE} | Batch: {BATCH_SIZE}")
print("Loading models...")

# 1. WhisperX STT
whisper_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
print(f"[OK] WhisperX ({MODEL_SIZE}) loaded on {DEVICE}.")

# 2. Speaker Embedding Model (음성 지문 추출용)
try:
    embedding_model = Inference(
        "pyannote/embedding",
        use_auth_token=HF_AUTH_TOKEN,
        window="whole"
    )
    if DEVICE == "cuda":
        embedding_model.to(torch.device("cuda"))
    print(f"[OK] Speaker Embedding model loaded on {DEVICE}.")
except Exception as e:
    print(f"[WARN] Speaker Embedding model load failed: {e}")
    embedding_model = None

# 3. Diarization Pipeline (한 녹음 안에서 다중 화자 분리용, 보조)
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_AUTH_TOKEN
    )
    if DEVICE == "cuda":
        diarization_pipeline.to(torch.device("cuda"))
    print(f"[OK] Diarization Pipeline loaded on {DEVICE}.")
except Exception as e:
    print(f"[WARN] Diarization Pipeline load failed: {e}")
    print("  → HuggingFace에서 pyannote/speaker-diarization-3.1 약관 동의 여부를 확인하세요.")
    diarization_pipeline = None

print("All models loaded. Server is ready.\n")

# --- Cross-Recording Speaker Memory ---
# { "SPEAKER_00": tensor, "SPEAKER_01": tensor, ... }
known_speakers: dict[str, torch.Tensor] = {}
speaker_counter = 0


def identify_speaker(embedding: torch.Tensor) -> str:
    """기존 화자들과 코사인 유사도 비교 후, 동일인이면 기존 ID / 신규면 새 ID 반환"""
    global speaker_counter
    if not known_speakers:
        speaker_id = f"SPEAKER_{speaker_counter:02d}"
        known_speakers[speaker_id] = embedding
        speaker_counter += 1
        print(f"  [Speaker] New speaker registered: {speaker_id}")
        return speaker_id

    best_id = None
    best_score = -1.0
    for sid, known_emb in known_speakers.items():
        score = F.cosine_similarity(
            embedding.unsqueeze(0), known_emb.unsqueeze(0)
        ).item()
        if score > best_score:
            best_score = score
            best_id = sid

    print(f"  [Speaker] Best match: {best_id} (score={best_score:.3f}, threshold={SIMILARITY_THRESHOLD})")

    if best_score >= SIMILARITY_THRESHOLD:
        # 기존 화자의 임베딩을 이동평균으로 업데이트 (점점 정확해짐)
        known_speakers[best_id] = 0.8 * known_speakers[best_id] + 0.2 * embedding
        return best_id
    else:
        speaker_id = f"SPEAKER_{speaker_counter:02d}"
        known_speakers[speaker_id] = embedding
        speaker_counter += 1
        print(f"  [Speaker] New speaker registered: {speaker_id}")
        return speaker_id


# --- Helpers ---
async def generate_tts_base64(text: str, voice: str) -> str:
    communicate = edge_tts.Communicate(text, voice)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    return base64.b64encode(audio_data).decode("utf-8")


def cleanup_files(*file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"  cleanup error [{path}]: {e}")


# --- Main Endpoint ---
@app.post("/chat/")
async def handle_chat_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_input_path = tmp.name

    try:
        dialogues = []

        # WebM → WAV 변환
        wav_path = temp_input_path + ".wav"
        try:
            audio_seg = AudioSegment.from_file(temp_input_path)
            audio_duration_sec = len(audio_seg) / 1000.0
            audio_seg.export(wav_path, format="wav")
            background_tasks.add_task(cleanup_files, wav_path)
        except Exception as e:
            print(f"  pydub conversion error: {e}")
            wav_path = temp_input_path
            audio_duration_sec = 0.0

        # STEP 1: WhisperX STT
        audio  = whisperx.load_audio(wav_path)
        result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
        detected_lang = result["language"]
        print(f"  Detected language: {detected_lang}")

        if not result.get("segments"):
            background_tasks.add_task(cleanup_files, temp_input_path)
            return JSONResponse(status_code=400, content={"error": "No speech detected."})

        # STEP 2: 타임스탬프 정렬
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=DEVICE
            )
            result = whisperx.align(
                result["segments"], model_a, metadata,
                audio, DEVICE, return_char_alignments=False
            )
        except Exception as e:
            print(f"  Alignment skipped: {e}")
        finally:
            try:
                del model_a
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        # STEP 3: TTS / 번역 언어 설정
        if detected_lang == "en":
            target_lang       = "ko"
            tts_voices        = ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"]
            tts_voice_default = "ko-KR-SunHiNeural"
        else:
            target_lang       = "en"
            tts_voices        = ["en-US-AriaNeural", "en-US-GuyNeural"]
            tts_voice_default = "en-US-AriaNeural"

        translator = GoogleTranslator(source=detected_lang, target=target_lang)

        # STEP 4: 녹음 간 화자 식별 (Speaker Embedding 비교)
        # 이 녹음 전체의 음성 지문을 추출하여 기존 화자들과 비교
        recording_speaker = "SPEAKER_00"  # default fallback
        if embedding_model is not None:
            try:
                emb = embedding_model(wav_path)
                emb_tensor = torch.tensor(emb).squeeze()
                recording_speaker = identify_speaker(emb_tensor)
            except Exception as e:
                print(f"  Embedding extraction failed: {e}")

        # STEP 5: 번역 + TTS 생성
        # 녹음 전체를 식별된 화자 1명의 발화로 처리
        voice_idx = int(recording_speaker.split("_")[-1]) if "_" in recording_speaker else 0
        tts_voice = tts_voices[voice_idx % len(tts_voices)]

        original_sentence = " ".join(
            seg.get("text", "").strip() for seg in result["segments"]
        ).strip()

        if not original_sentence:
            background_tasks.add_task(cleanup_files, temp_input_path)
            return JSONResponse(status_code=400, content={"error": "No speech detected."})

        translated_text = translator.translate(original_sentence)
        audio_base64    = await generate_tts_base64(translated_text, tts_voice)

        dialogues.append({
            "speaker":    recording_speaker,
            "original":   original_sentence,
            "translated": translated_text,
            "language":   detected_lang,
            "audio_b64":  audio_base64,
        })

        print(f"  Result: [{recording_speaker}] {original_sentence} → {translated_text}")

        background_tasks.add_task(cleanup_files, temp_input_path)
        return JSONResponse(status_code=200, content={"dialogues": dialogues})

    except Exception as e:
        background_tasks.add_task(cleanup_files, temp_input_path)
        print(f"Backend Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Speaker Reset Endpoint ---
@app.post("/reset_speakers/")
async def reset_speakers():
    global known_speakers, speaker_counter
    known_speakers = {}
    speaker_counter = 0
    print("[Speaker] All speaker memory cleared.")
    return JSONResponse(status_code=200, content={"message": "Speaker memory cleared."})

