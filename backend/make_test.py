from pydub import AudioSegment
from pydub.generators import Sine
import random

# 사람 목소리 흉내 (주파수 다르게 — 화자 구분용)
def voice_segment(freq, duration_ms):
    tone = Sine(freq).to_audio_segment(duration=duration_ms)
    return tone - 20  # 볼륨 낮추기

# SPEAKER_00 (낮은 톤 = 남성 흉내): 150Hz
# SPEAKER_01 (높은 톤 = 여성 흉내): 280Hz
s0 = voice_segment(150, 3000)  # 3초
s1 = voice_segment(280, 3000)  # 3초
silence = AudioSegment.silent(duration=500)  # 0.5초 침묵

audio = s0 + silence + s1 + silence + s0 + silence + s1
audio.export("test.wav", format="wav")
print("test.wav 생성 완료 (총", len(audio)/1000, "초)")
