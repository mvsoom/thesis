#!/usr/bin/env python3
"""
vi_batch.py – run a JAX-based Variational Inference (VI) routine on one or more
              audio files, resample to 16 kHz, and pickle per-frame results.

Usage
-----
    python vi_batch.py  speech1.wav  song.wav  -o  results/
"""

import argparse
import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf  # pip install soundfile
from scipy.signal import resample_poly  # pip install scipy

from .hyperparams import Hyperparams
from .mercer import psd_svd
from .periodic import periodic_kernel_batch  # your helper (MacKay kernel)
from .state import init_state
from .vi import compute_elbo_bound, update_delta_a, vi_step

##############################################################################
# Global constants / hyper-parameters
##############################################################################

jax.config.update("jax_enable_x64", True)  # use double precision

FRAME_LEN = 512  # analysis window M (samples)  –  do *not* change
HOP = 160  # shift between successive windows
TARGET_SR = 8000  # resample every file to 16 kHz
NOISE_FLOOR_DB = -60.0  # for psd_svd
MAX_ITER = 1000

update_delta_a = jax.jit(update_delta_a)
vi_step = jax.jit(vi_step)
compute_elbo_bound = jax.jit(compute_elbo_bound)

##############################################################################
#  Helper functions
##############################################################################


def _mono(x: np.ndarray) -> np.ndarray:
    """Return mono signal (pick channel 0 if multichannel)."""
    return x[:, 0] if x.ndim > 1 else x


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited polyphase resampling that preserves dtype."""
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_out, sr_in)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down, axis=0).astype(x.dtype)


def _frame_signal(x: np.ndarray) -> np.ndarray:
    """Return a view of `x` with shape (n_frames, FRAME_LEN)."""
    if len(x) < FRAME_LEN:
        return np.empty((0, FRAME_LEN), x.dtype)
    n_frames = 1 + (len(x) - FRAME_LEN) // HOP
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, FRAME_LEN),
        strides=(HOP * x.strides[0], x.strides[0]),
        writeable=False,
    )


##############################################################################
#  Core processing
##############################################################################


def process_frame(key, frame, h):
    # Test (first run very slow, then fast)
    state = init_state(key, frame, h)

    # Updating q(a) = delta(a* - a) as the very first update
    # is known to yield better convergence
    # as it is initalized to zeroes
    state = update_delta_a(state)
    # TODO shift order of updates by -1 such that update_delta_a is first

    score = -jnp.inf
    criterion = 0.00005

    for i in range(MAX_ITER):
        state = vi_step(state)

        lastscore = score
        score = compute_elbo_bound(state)

        if i == 0:
            improvement = 1.0
        else:
            improvement = (score - lastscore) / jnp.abs(lastscore)

        print(
            "iteration {}: bound = {:.2f} ({:+.5f} improvement)".format(
                i, score, improvement
            )
        )
        if improvement < 0.0:
            print("Diverged")
            break
        if improvement < criterion:
            print("Converged")
            break
        if jnp.isnan(improvement) and i > 0:
            print("NaN")
            break

    return (state.xi, score)


def process_file(path: Path) -> dict:
    """
    • read any libsndfile-supported format (incl. NIST/SPHERE)
    • mono → float32 → resample to TARGET_SR
    • pre-compute periodic kernels & their PSD-SVD basis (Phi)
    • run VI on every frame
    • return a pickle-friendly dict
    """
    # ------------------- LOAD & RESAMPLE -------------------------------------
    audio, sr_in = sf.read(path, always_2d=False, dtype="float64")
    audio = _mono(audio)
    audio = _resample(audio, sr_in, TARGET_SR)

    print(f"→ Loaded {path} ({len(audio)} samples, {sr_in} Hz)")

    # ------------------- BUILD KERNELS (once per file) -----------------------
    K, f0s = periodic_kernel_batch(M=FRAME_LEN, fs=TARGET_SR)
    Phi = psd_svd(K, noise_floor_db=NOISE_FLOOR_DB)
    h = Hyperparams(Phi)

    print(
        f"→ {len(f0s)} kernels, 1st F0 = {float(f0s[0]):.2f} Hz, "
        f"Phi.shape = {Phi.shape}, noise floor = {NOISE_FLOOR_DB} dB"
    )

    # ------------------- PER-FRAME VI ----------------------------------------
    frames = _frame_signal(audio)
    results = []

    # FAILURE: this is bad
    # First N-1 keys of split(N) are identical to split(N-1)!!
    # Need key at top level
    keys = jax.random.split(jax.random.PRNGKey(68456), len(frames))

    for key, frame in zip(keys, frames):
        print(f"→ → Processing frame {len(results) + 1}/{len(frames)}")
        results.append(process_frame(key, frame, h))

    # ------------------- PACK EVERYTHING UP ----------------------------------
    return {
        "sample_rate": TARGET_SR,
        "frame_len": FRAME_LEN,
        "hop": HOP,
        "f0s": f0s,  # 1-D array of F0s used for kernels
        "noise_floor_db": NOISE_FLOOR_DB,
        "Phi": Phi,  # SVD basis (could be big – your call)
        "results": results,
    }


##############################################################################
#  CLI glue
##############################################################################


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch-run JAX VI on audio files (resampled to 16 kHz)."
    )
    ap.add_argument("files", nargs="+", help="Audio files (.wav, .flac, …)")
    ap.add_argument(
        "-o",
        "--outdir",
        default="results",
        help="Directory for <name>.pkl (default: ./results)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"→ Output directory: {outdir}")
    print(f"→ Processing {len(args.files)} files …")

    for fname in args.files:
        p = Path(fname).expanduser()
        if not p.is_file():
            print(f"[WARN] {p} missing – skipped.")
            continue

        print(f"→ Processing {p} …")
        data = process_file(p)
        with open(outdir / (p.stem + ".pkl"), "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   Saved {p.stem}.pkl")

    print("✓ All done.")


if __name__ == "__main__":
    main()
