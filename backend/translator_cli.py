import os
import gc
import site
import tempfile
import asyncio

import torch
import whisperx
import speech_recognition as sr
from deep_translator import GoogleTranslator
import edge_tts
import pygame

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
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODEL_SIZE   = "large-v3"
BATCH_SIZE   = 8  # RTX 4060 Laptop 8GB 기준


def play_audio(file_path: str):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


async def generate_and_play_tts(text: str, voice: str):
    print(f"\n[TTS 생성 중...] {text}")
    tmp_dir  = tempfile.gettempdir()
    tmp_file = os.path.join(tmp_dir, "cli_output.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_file)
    print("[음성 재생 중...]")
    play_audio(tmp_file)
    try:
        os.remove(tmp_file)
    except OSError:
        pass


def main():
    print("=" * 55)
    print(f"⏳ WhisperX 모델 로딩 중... [{DEVICE} / {COMPUTE_TYPE}]")
    print("   (최초 실행 시 모델 다운로드로 수 분 소요)")

    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

    # 한국어·영어 align 모델 미리 로딩 (첫 발화 지연 방지)
    align_models = {}
    for lang in ["ko", "en"]:
        try:
            m, meta = whisperx.load_align_model(language_code=lang, device=DEVICE)
            align_models[lang] = (m, meta)
            print(f"  [OK] Align model for '{lang}' preloaded.")
        except Exception as e:
            print(f"  [WARN] Align model for '{lang}' failed: {e}")

    recognizer = sr.Recognizer()

    print("=" * 55)
    print("🎙️  실시간 양방향 번역기 시작 (종료: Ctrl+C)")
    print("   한국어 → 영어 / 영어 → 한국어 자동 감지")
    print("=" * 55)

    with sr.Microphone() as source:
        print("조용한 환경에서 주변 소음 적응 중... 잠시만 기다려주세요.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ 준비 완료! 자유롭게 말씀해주세요.\n")

        while True:
            try:
                print("🟢 듣는 중... (말씀해주세요)")
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                print("🔄 음성 분석 중...")

                # SpeechRecognition 버퍼 → 임시 WAV 파일
                wav_bytes = audio_data.get_wav_data()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav_bytes)
                    tmp_wav = f.name

                # WhisperX STT
                audio  = whisperx.load_audio(tmp_wav)
                result = model.transcribe(audio, batch_size=BATCH_SIZE)
                detected_lang = result["language"]

                # 타임스탬프 정렬 (align 모델 있을 때만)
                if detected_lang in align_models:
                    m, meta = align_models[detected_lang]
                    try:
                        result = whisperx.align(
                            result["segments"], m, meta,
                            audio, DEVICE, return_char_alignments=False
                        )
                    except Exception as e:
                        print(f"  Alignment skipped: {e}")

                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass

                transcribed_text = " ".join(
                    seg.get("text", "").strip() for seg in result["segments"]
                ).strip()

                if not transcribed_text:
                    print("  (음성 미감지, 다시 시도합니다)\n")
                    continue

                print(f"🌐 감지 언어: {detected_lang}")
                print(f"🗣️  원문: {transcribed_text}")

                # 번역 방향 설정
                if detected_lang == "en":
                    target_lang = "ko"
                    tts_voice   = "ko-KR-SunHiNeural"
                else:
                    target_lang = "en"
                    tts_voice   = "en-US-AriaNeural"

                translator      = GoogleTranslator(source=detected_lang, target=target_lang)
                translated_text = translator.translate(transcribed_text)
                print(f"📝 번역: {translated_text}")

                asyncio.run(generate_and_play_tts(translated_text, tts_voice))
                print("-" * 55)

            except sr.WaitTimeoutError:
                pass  # 조용한 구간, 정상
            except KeyboardInterrupt:
                print("\n🛑 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

    # 종료 시 VRAM 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
