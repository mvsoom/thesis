# %%
import jax

# no compile info
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu")
# %%
import os

import jax.numpy as jnp
import numpy as np
import soundfile as sf
from IPython.display import display

from ar import spectrum
from ar.soft_divisibility import (
    constraints_from_features,
    me_soft_divisibility,
    pole_pair,
    spectral_tilt,
)
from iklp.hyperparams import ARPrior, pi_kappa_hyperparameters
from iklp.periodic import periodic_kernel_phi
from iklp.psi import psi_matvec
from utils.audio import frame_signal, resample
from utils.jax import maybe32
from utils.plots import plt

# %%
# Parameters
initial_pitchedness = 0.99
noise_floor_db = -100.0  # No noise floor
seed = np.random.randint(0, 2**31 - 1)
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
P = 16
I = 200
f0_min = 70  # Typical lower bound (Nielsen 2013)
f0_max = 400

# Adjust these together
target_sr = 16000
frame_len = 512
hop = 80

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


lam = 0.1
mu0 = np.zeros(P)
Sigma0 = lam * np.eye(P)


def formant_feature(freq, bandwidth=100):
    w = 2.0 * np.pi * freq / target_sr
    r = np.exp(-np.pi * bandwidth / target_sr)
    return pole_pair(r, w)


features = [
    formant_feature(500, 75),
    formant_feature(1000, 100),
    formant_feature(1500, 150),
    formant_feature(2000, 150),
    spectral_tilt(rho=0.95, k=2),
]

C, d = constraints_from_features(P, features)
mu_star, Sigma_star = me_soft_divisibility(mu0, Sigma0, P, features)


def get_power_spectrum_samples(mu, Sigma, n=5):
    a = np.random.default_rng().multivariate_normal(mu, Sigma, size=n)

    def power_spectrum(a):
        f, p = spectrum.ar_power_spectrum(a, target_sr)
        return 10 * np.log10(p)

    return np.array([power_spectrum(ai) for ai in a])


print("mu0     =", mu0)
print("mu*     =", mu_star)

f, power0 = spectrum.ar_power_spectrum(mu0, target_sr)
f, power_star = spectrum.ar_power_spectrum(mu_star, target_sr)


def plot_samples(ax, f, S, color, label="Samples"):
    lines = ax.plot(f, S.T, color=color, alpha=0.3)
    lines[0].set_label(label)
    for ln in lines[1:]:
        ln.set_label("_nolegend_")
    return lines


fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

# left: initial
ax0.plot(f, 10 * np.log10(power0), color="C0", lw=2, label="Mean")
s0 = get_power_spectrum_samples(mu0, Sigma0, n=5)
# plot_samples(ax0, f, s0, color="C0")
ax0.set_title("Initial: $a \sim \mathcal{N}(0, \lambda I_P)$")
ax0.set_xlabel("Frequency (Hz)")
ax0.set_ylabel("Power (dB)")
ax0.legend(loc="upper right")

# right: after ME update
ax1.plot(f, 10 * np.log10(power_star), color="C1", lw=2, label="Mean")
s_star = get_power_spectrum_samples(mu_star, Sigma_star, n=10)
# plot_samples(ax1, f, s_star, color="C1")
ax1.set_title("After ME update: $a \sim \mathcal{N}(\\mu^*, \\Sigma^*)$")
ax1.set_xlabel("Frequency (Hz)")
ax1.legend(loc="upper right")

fig.suptitle(f"Prior spectra ($P={P}$)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
display(fig)

arprior = ARPrior(mean=mu_star, precision=jnp.linalg.inv(Sigma_star))
# arprior = ARPrior(mean=mu0, precision=jnp.linalg.inv(Sigma0))

# %%
f0, Phi = periodic_kernel_phi(
    I=I,
    M=frame_len,
    r=0.25,
    fs=target_sr,
    f0_min=f0_min,
    f0_max=f0_max,
    noise_floor_db=noise_floor_db,
)

alpha = 1.00
kappa = 1.00

h = pi_kappa_hyperparameters(
    Phi,
    pi=initial_pitchedness,
    kappa=kappa,
    alpha=maybe32(alpha),
    arprior=arprior,
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

dt = t_millisec[1] - t_millisec[0]
de = jnp.cumsum(e).T * dt
de = (de - jnp.mean(de)) / jnp.std(de) * jnp.std(dgf_frame) + jnp.mean(
    dgf_frame
)

fig, ax = plt.subplots()
ax.plot(t_millisec, de, label="Inferred source signal")
ax.plot(t_millisec, dgf_frame, label="Ground truth glottal flow")
ax.legend()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
# ax.set_xlim(20, 30)
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
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Inferred AR power spectrum ($\lambda$ prior)")
for formant in [f1, f2, f3, f4]:
    ax.axvline(formant, color="C1", ls="--", label=f"Formant {formant} Hz")
display(fig)

# %%
fig, ax = plt.subplots()
ax.plot(f0, metrics.E.theta)
ax.set_xlabel("Fundamental frequency $F_0$ (Hz)")
ax.set_ylabel("Power")
display(fig)
