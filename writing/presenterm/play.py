from time import sleep as _sleep

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from scipy.io import wavfile

sleep = _sleep  # export

fs, x = wavfile.read("./bush.wav")
x = x.astype(np.float32)

# stereo -> mono
if x.ndim > 1:
    x = x.mean(axis=1)

# normalize
x /= np.max(np.abs(x)) + 1e-9

console = Console()


# ---------------------------------------------------
# Oscilloscope renderer
# ---------------------------------------------------
def _render_wave(chunk, height=20, width=80):
    if len(chunk) < width:
        chunk = np.pad(chunk, (0, width - len(chunk)))

    c = chunk[-width:]
    c /= np.max(np.abs(c)) + 1e-9

    top = []
    bot = []
    for v in c:
        pos = int(max(v, 0) * height)
        neg = int(max(-v, 0) * height)
        top.append("█" if pos > 0 else " ")
        bot.append("█" if neg > 0 else " ")

    return "\n".join(("".join(top), "".join(bot)))


# ---------------------------------------------------
# Play with oscilloscope
# ---------------------------------------------------
def play(arr):
    # ensure mono
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    # normalize per-call
    arr = arr.astype(np.float32)
    arr /= np.max(np.abs(arr)) + 1e-9

    disp = arr
    blocksize = 1024
    idx = 0

    with Live(console=console, refresh_per_second=30) as live:

        def callback(outdata, frames, time, status):
            nonlocal idx
            if idx + frames > len(arr):
                raise sd.CallbackStop()

            chunk = arr[idx : idx + frames]
            outdata[:] = chunk.reshape(-1, 1)

            live.update(_render_wave(chunk))
            idx += frames

        with sd.OutputStream(
            samplerate=fs,
            channels=1,
            callback=callback,
            blocksize=blocksize,
        ):
            sd.sleep(int(len(arr) / fs * 1000))
