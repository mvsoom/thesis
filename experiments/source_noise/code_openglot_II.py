# %%
import os

import jax

# use CPU
jax.config.update("jax_platform_name", "cpu")

# %%

import jax.numpy as jnp
import numpy as np
import soundfile as sf
from IPython.display import display

from iklp.hyperparams import pi_kappa_hyperparameters
from iklp.periodic import periodic_kernel_phi
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
    "/home/marnix/thesis/data/OPENGLOT/RepositoryII_male/Male_U_220Hz_small.wav"
)


# %%
base = os.path.splitext(os.path.basename(wav_file))[0]

true_resonance_frequencies = {
    "male": {
        "a": [752, 1095, 2616, 3169],
        "i": [340, 2237, 2439, 3668],
        "u": [367, 1180, 2395, 3945],
        "ae": [693, 1521, 2435, 3252],
    },
    "female": {
        "a": [848, 1210, 2923, 3637],
        "i": [379, 2634, 4256, 5395],
        "u": [420, 1264, 2714, 4532],
        "ae": [795, 1700, 2692, 3740],
    },
}

# %%
gender, vowel, p, adduction = base.split("_")
gender, vowel, true_pitch, adduction = (
    gender.lower(),
    vowel.lower(),
    int(p.rstrip("Hz").lower()),
    adduction.lower(),
)

f1, f2, f3, f4 = true_resonance_frequencies[gender][vowel]

# %%
vi_runs = 1

# Use same parameters as in OPENGLOT and Yoshii
P = 9
I = 400
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
gf = audio[:, 1]

x = resample(x, sr_in, target_sr)
gf = resample(gf, sr_in, target_sr)

# Normalize data to unit power
scale = 1 / np.sqrt(np.mean(x**2))
x = x * scale
gf = gf * scale

print(f"→ Loaded {wav_file} ({len(x)} samples, {sr_in} Hz)")

# %%
f0, Phi = periodic_kernel_phi(
    I=I,
    M=frame_len,
    fs=target_sr,
    f0_min=f0_min,
    f0_max=f0_max,
    noise_floor_db=noise_floor_db,
)

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
gf_frames = frame_signal(gf, frame_len, hop)
t_millisec = 1000 * np.arange(frame_len) / target_sr
frames = jnp.array(frames)

# %%
frame = frames[15]
dgf_frame = gf_frames[15]

fig, ax = plt.subplots()
ax.plot(t_millisec, frame + 3, label="Data $x(t)$")
ax.plot(t_millisec, dgf_frame - 3, label="Glottal flow $u(t)$")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.set_yticklabels([])
ax.set_yticks([])
display(fig)

# %%

from iklp.run import print_progress, vi_run_criterion

h = h.replace(vi_criterion=1e-4, mercer_backend="auto")

state, metrics = vi_run_criterion(master_key, frame, h, callback=print_progress)

# %%
pitchedness = metrics.E.nu_w / (metrics.E.nu_w + metrics.E.nu_e)
print("pitchedness =", pitchedness)

# %%
signal = metrics.signals[2]

e = psi_matvec(metrics.a, state.data.x)  # (M,)

noise = e - signal

fig, ax = plt.subplots()
ax.plot(t_millisec, signal, label="Inferred $u'(t)$ signal")
ax.plot(t_millisec, noise, label="Inferred noise signal")
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
ax.plot(t_millisec, noises.T, label="Inferred source signal")
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
