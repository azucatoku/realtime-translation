from pyannote.audio import Pipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", "")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_AUTH_TOKEN
)
pipeline.to(torch.device("cuda"))

# 테스트용 wav 파일 경로 (10초 이상, 두 명 대화)
TEST_WAV = "test.wav"

result = pipeline(TEST_WAV)

print("\n--- 화자 분리 결과 ---")
for turn, _, speaker in result.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s ~ {turn.end:.1f}s] {speaker}")
