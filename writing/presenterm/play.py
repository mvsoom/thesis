from time import sleep as _sleep

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

sleep = _sleep  # re-export

# --- load audio ---
fs, x = wavfile.read("./bush.wav")
x = x.astype(np.float32)

# stereo -> mono
if x.ndim > 1:
    x = x.mean(axis=1)

# DC remove + normalize to [-1, 1]
x -= x.mean()
x /= np.max(np.abs(x)) + 1e-12

# --- playback ---
def play(arr):
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    arr = arr.astype(np.float32)
    arr -= arr.mean()
    arr /= np.max(np.abs(arr)) + 1e-12

    idx = 0
    blocksize = 1024

    def callback(outdata, frames, time, status):
        nonlocal idx
        if idx + frames > len(arr):
            raise sd.CallbackStop()
        outdata[:] = arr[idx : idx + frames, None]
        idx += frames

    with sd.OutputStream(
        samplerate=fs,
        channels=1,
        callback=callback,
        blocksize=blocksize,
    ):
        sd.sleep(int(len(arr) / fs * 1000))
