# %%
import math

import jax
import jax.numpy as jnp


def f0_series(
    f0_min: float = 100.0,
    f0_max: float = 400.0,
    cent_step: float = 6.0,
) -> jnp.ndarray:
    """
    Geometric F0 grid from `f0_min` to `f0_max` (inclusive) in `cent_step`-cent
    increments.  Returns a 1-D JAX array.

    cent_step = 6  → ratio 2^(6/1200) ≈ 1.003466 .
    """
    ratio = 2 ** (cent_step / 1200.0)
    n_steps = math.floor(math.log(f0_max / f0_min, ratio)) + 1
    f0s = f0_min * ratio ** jnp.arange(n_steps)
    # make sure the last value does not overshoot f0_max due to FP error
    return jnp.where(f0s > f0_max + 1e-9, f0_max, f0s)


def periodic_kernel_batch(
    M: int,
    fs: float | int = 16_000,
    r: float = 1.0,
    f0_min: float = 100.0,
    f0_max: float = 400.0,
    cent_step: float = 6.0,
):
    """
    Build a batch of MacKay‐style periodic kernel matrices.

    Parameters
    ----------
    M          : frame length (# samples)
    fs         : sampling rate in Hz  →  dt = 1/fs   [seconds]
    r          : MacKay length-scale (≈ 1.0;   r ≈ 0.77 is what the note suggests)
    f0_min/max : lowest & highest F0 in Hz
    cent_step  : spacing of the geometric F0 grid, in *cents*

    Returns
    -------
    K : jnp.ndarray with shape [B, M, M]   (B = # F0 values)
    f0s : 1-D array of F0s actually used   (handy for later)
    """
    # ---- build the F0 grid ---------------------------------------------------
    f0s = f0_series(f0_min=f0_min, f0_max=f0_max, cent_step=cent_step)  # [B]
    Ts = 1.0 / f0s  # periods in seconds                                # [B]

    # ---- time axis (seconds) -----------------------------------------------
    dt = 1.0 / fs
    t = jnp.arange(M) * dt  # [M]
    tau = t[None, :, None] - t[None, None, :]  # [1, M, M]

    # ---- MacKay periodic kernel --------------------------------------------
    arg = jnp.pi * tau / Ts[:, None, None]  # [B, M, M]
    K = jnp.exp(-0.5 * (jnp.sin(arg) ** 2) / (r**2))  # [B, M, M]
    return K, f0s


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    K, f0s = periodic_kernel_batch(M=512, fs=20_000, r=1.0)
    print(
        f"#kernels = {len(f0s)},  1st F0 = {float(f0s[0]):.2f} Hz,  "
        f"last F0 = {float(f0s[-1]):.2f} Hz"
    )
    # → #kernels = 50,  1st F0 = 100.00 Hz,  last F0 = 399.99 Hz
    print(K.shape)  # (50, 512, 512)
