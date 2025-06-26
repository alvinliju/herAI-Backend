import sounddevice as sd
import numpy as np

# Record 5 seconds
print("Recording...")
audio = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype='float64')
sd.wait()
print("Done recording")
print(f"Audio shape: {audio.shape}")
