# %%
import pickle
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf  # pip install soundfile

from .eval import _frame_signal, _mono, _resample
from .gig import compute_gig_expectations
from .hyperparams import Hyperparams
from .state import Expectations, VariationalParams

##############################################################################
# Global constants / hyper-parameters
##############################################################################

jax.config.update("jax_enable_x64", True)  # use double precision

compute_gig_expectations = jax.jit(compute_gig_expectations)


WAVFILE = "eval/mini_aplawd/aw47e2.wav"  # path to the audio file
PICKLEFILE = "results/aw47e2.pkl"  # path to the pickle file


picklefile = open(PICKLEFILE, "rb")

data = pickle.load(picklefile)

dt = 1 / data["sample_rate"]

h = Hyperparams(data["Phi"])


def process_file(path: Path, fs):
    audio, sr_in = sf.read(path, always_2d=False, dtype="float64")
    audio = _mono(audio)
    audio = _resample(audio, sr_in, fs)

    print(f"→ Loaded {path} ({len(audio)} samples, {sr_in} Hz)")

    frames = _frame_signal(audio)
    return frames


def horrible_compute_expectations(
    xi: VariationalParams, h: Hyperparams
) -> Expectations:
    """Compute the GIG expectations from current state"""
    I = h.Phi.shape[0]

    theta, theta_inv = jax.vmap(compute_gig_expectations, in_axes=(None, 0, 0))(
        h.alpha / I, xi.rho_theta, xi.tau_theta
    )
    nu_w, nu_w_inv = compute_gig_expectations(h.aw, xi.rho_w, xi.tau_w)
    nu_e, nu_e_inv = compute_gig_expectations(h.ae, xi.rho_e, xi.tau_e)

    return Expectations(theta, theta_inv, nu_w, nu_w_inv, nu_e, nu_e_inv)


frames = process_file(WAVFILE, data["sample_rate"])
N, M = frames.shape

frame_centers = np.asarray(
    [i * data["hop"] + (data["frame_len"] // 2) for i in range(N)]
)
ts = frame_centers * dt


def process(frames, ts, data):
    for i, (frame, t, result) in enumerate(zip(frames, ts, data["results"])):
        # print(f"Frame {i}:")
        # print(f"Frame center: {ts} sec")

        xi, elbo_bound = result
        # print("ELBO bound:", elbo_bound)

        E = horrible_compute_expectations(xi, h)

        f0 = data["f0s"][E.theta.argmax()]
        # print("F0:", f0, "Hz")

        pitched = E.nu_w / (E.nu_w + E.nu_e)
        # print("Pitched:", pitched)

        yield f0, pitched


f0, pitched = zip(*process(frames, frame_centers, data))

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axs[0].plot(ts, f0, label="F0")
axs[0].set_ylabel("F0 (Hz)")
axs[0].legend()

axs[1].plot(ts, pitched, label="Pitched")
axs[1].set_ylabel("Pitched")
axs[1].set_xlabel("Time (s)")
axs[1].legend()

plt.tight_layout()
plt.show()
