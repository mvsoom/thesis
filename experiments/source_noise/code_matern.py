# %%
import jax

# use CPU
jax.config.update("jax_platform_name", "cpu")

# %%

import jax.numpy as jnp
import numpy as np
import soundfile as sf
from IPython.display import display

from gp.spectral import Matern12
from iklp.hyperparams import pi_kappa_hyperparameters
from iklp.mercer import psd_svd
from iklp.periodic import f0_series
from iklp.psi import psi_matvec
from utils.audio import frame_signal, resample
from utils.jax import maybe32
from utils.plots import plt

# %%
# Parameters
initial_pitchedness = 0.99
noise_floor_db = -100.0  # No noise floor
seed = 0
wav_file = (
    "/home/marnix/thesis/data/OPENGLOT/RepositoryI/Vowel_O/O_normal_320Hz.wav"
)


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
vi_runs = 1

# Use same parameters as in OPENGLOT and Yoshii
P = 9
I = 100
f0_min = 70  # Typical lower bound (Nielsen 2013)
f0_max = 400

# Adjust these together
target_sr = 8000
frame_len = 512
hop = 40

max_iter = 50

verbose = True

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

print(f"→ Loaded {wav_file} ({len(x)} samples, {sr_in} Hz)")

# %%
f0 = f0_series(f0_min=f0_min, f0_max=f0_max, I=I)

ells = 1 / f0 * 1000  # in ms

kernels = [Matern12(scale=ell) for ell in ells]

t = jnp.arange(frame_len) / target_sr * 1000  # in sec
K = jnp.stack([k(t, t) for k in kernels], axis=0)

Phi = psd_svd(K)

alpha = 1.0
kappa = 1.0

h = pi_kappa_hyperparameters(
    Phi,
    pi=initial_pitchedness,
    kappa=kappa,
    alpha=maybe32(alpha),
    P=P,
)

master_key = jax.random.PRNGKey(seed)

frames = frame_signal(x, frame_len, hop)  # ((n_frames, frame_len)
dgf_frames = frame_signal(dgf, frame_len, hop)
frames = jnp.array(frames)

# %%
frame = frames[4]
dgf_frame = dgf_frames[4]

fig, ax = plt.subplots()
ax.plot(t, frame + 3, label="Data $x(t)$")
ax.plot(t, dgf_frame - 3, label="Glottal flow $u(t)$")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.set_yticklabels([])
ax.set_yticks([])
display(fig)

# %%

from iklp.run import print_progress, vi_run_criterion

h = h.replace(vi_criterion=1e-5, mercer_backend="cholesky")

state, metrics = vi_run_criterion(master_key, frame, h, callback=print_progress)

# %%
signal = metrics.signals[2]

e = psi_matvec(metrics.a, state.data.x)  # (M,)

noise = e - signal

fig, ax = plt.subplots()
ax.plot(t, signal, label="Inferred source signal")
ax.plot(t, noise, label="Inferred noise signal")
ax.legend()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.set_xlim(20, 30)
display(fig)

# %%
signals = metrics.signals


e = psi_matvec(metrics.a, state.data.x)  # (M,)

noises = e - signals

fig, ax = plt.subplots()
ax.plot(t, signals.T, label="Inferred source signal")
ax.legend()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.set_xlim(20, 30)
display(fig)

# %%
from ar import spectrum

f, power0 = spectrum.ar_power_spectrum(metrics.a, target_sr)
fig, ax = plt.subplots()
ax.plot(
    f,
    10 * np.log10(power0),
    color="C0",
    lw=2,
    label="Inferred AR power spectrum",
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Inferred AR power spectrum")
for formant in [f1, f2, f3, f4]:
    ax.axvline(formant, color="C1", ls="--", label=f"Formant {formant} Hz")
display(fig)

# %%
fig, ax = plt.subplots()
ax.plot(f0, metrics.E.theta)
fig
