# %%
# [parameters] [export]
seed = 0
wav_file = (
    "/home/marnix/thesis/data/OPENGLOT/RepositoryI/Vowel_O/O_normal_320Hz.wav"
)
initial_pitchedness = 0.5
noise_floor_db = -40.0

jax_enable_x64 = True
jax_platform_name = "gpu"

# %%
# First to set config flags wins!
from jax import config

from iklp.run import vi_frames_batched

config.update("jax_enable_x64", jax_enable_x64)
config.update("jax_platform_name", jax_platform_name)
# %%
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf

from iklp.hyperparams import Hyperparams
from iklp.periodic import periodic_kernel_phi
from utils.audio import frame_signal, resample

# %%
# Print out jax config
print(
    "x64:",
    jax.config.read("jax_enable_x64"),
    "default float:",
    jnp.array(0.0).dtype,
)
print("backend:", jax.default_backend(), "devices:", jax.devices())

# %%
# From OPENGLOT paper Table 1
true_resonance_frequencies = {
    "a": [730, 1090, 2440, 3500],
    "e": [530, 1840, 2480, 3500],
    "i": [390, 1990, 2550, 3500],
    "o": [570, 840, 2410, 3500],
    "u": [440, 1020, 2240, 3500],
    "ae": [660, 1720, 2410, 3500],
}

# %%
vowel, modality, true_pitch = (
    wav_file.split("/")[-1].split(".")[-2].split("_")[:3]
)
vowel = vowel.lower()
modality = modality.lower()
true_pitch = int(true_pitch.lower()[:-2])  # Remove 'Hz' from the pitch string

f1, f2, f3, f4 = true_resonance_frequencies[vowel]

# %%
# Use same parameters as in OPENGLOT and Yoshii
P = 9
I = 400
f0_min = 70  # Typical lower bound (Nielsen 2013)
f0_max = 400

# Adjust these together
target_sr = 8000
frame_len = 1024
hop = 80

# %%
audio, sr_in = sf.read(wav_file, always_2d=False, dtype="float64")

# Split channels
x = audio[:, 0]
dgf = audio[:, 1]

x = resample(x, sr_in, target_sr)
dgf = resample(dgf, sr_in, target_sr)

# Normalize data to unit power
scale = 1 / np.sqrt(np.mean(x**2))
x = x * scale
dgf = dgf * scale

frames = frame_signal(x, frame_len, hop)  # ((n_frames, frame_len)
frames = jnp.array(frames)

print(
    f"→ Loaded {wav_file} ({len(x)} samples, {sr_in} Hz), {frames.shape[0]} frames of {frame_len} samples each"
)

# %%
f0, Phi = periodic_kernel_phi(
    I=I,
    M=frame_len,
    fs=target_sr,
    f0_min=f0_min,
    f0_max=f0_max,
    noise_floor_db=noise_floor_db,
)

aw = initial_pitchedness / (1 - initial_pitchedness)


h = Hyperparams(Phi, P=P, aw=aw, num_vi_restarts=5, num_vi_iters=30)

master_key = jax.random.PRNGKey(seed)


# %%
t0 = time()
metrics = vi_frames_batched(master_key, frames, h, batch_size=4)
# Materialize
jax.block_until_ready(metrics)
walltime = time() - t0

# %%

print(f"Metrics shape: {metrics.elbo.shape}")
print(f"Walltime for VI: {walltime:.2f} seconds")

total_iters = np.prod(metrics.elbo.shape[:3])
print(f"Total iterations: {total_iters}")
print(f"Time per iteration: {walltime / total_iters:.2f} seconds")

# %%
# [export]
I, M, r = h.Phi.shape
