# %%

import jax
import jax.numpy as jnp

from iklp.mercer import psd_svd
from utils import batch_generator
from utils.caching import memory


def f0_series(
    f0_min: float = 100.0,
    f0_max: float = 400.0,
    I: int = 400,
) -> jnp.ndarray:
    """Geometric F0 grid from `f0_min` to `f0_max` (inclusive) with `I` logarithmically spaced steps"""
    ratio = (f0_max / f0_min) ** (1.0 / (I - 1))
    f0s = f0_min * ratio ** jnp.arange(I)
    return f0s  # (I,)


def periodic_kernel_generator(
    I: int,
    M: int = 2048,
    fs: float | int = 16_000,
    r: float = 1.0,
    f0_min: float = 100.0,
    f0_max: float = 400.0,
):
    """
    Build a batch of MacKay-style periodic kernel matrices

    See periodickernel.jl for the original implementation and why setting r = 1.0 is a good idea.
    """
    dt = 1.0 / fs
    t = jnp.arange(M) * dt
    tau = t[:, None] - t[None, :]

    f0s = f0_series(f0_min=f0_min, f0_max=f0_max, I=I)
    for f0 in f0s:
        T = 1.0 / f0
        K_i = jnp.exp(-0.5 * (jnp.sin(jnp.pi * tau / T) / r) ** 2)  # (M, M)
        yield (f0, K_i)  # (M, M)


def periodic_kernel(**kwargs):
    """Get the full (I, M, M) K matrix in one go"""
    f0s, Ks = zip(*list(periodic_kernel_generator(**kwargs)))
    return jnp.array(f0s), jnp.stack(Ks)


@memory.cache(ignore=["batch_size"])
def periodic_kernel_phi(
    I: int = 400,
    M: int = 2048,
    fs: float | int = 16_000,
    r: float = 1.0,
    f0_min: float = 100.0,
    f0_max: float = 400.0,
    noise_floor_db: float = -60.0,
    batch_size: int = 25,
):
    """Get the periodic kernel matrices in batches and compute their Mercer expansions

    Note: for a given `noise_floor_db`, the rank `r` of the expansion is empirically exactly equal for all kernels in the batch, so we can process them in batches.
    For `noise_floor_db == -60` dB, the rank is 9 for M = 2048.
    """
    g = periodic_kernel_generator(
        I=I, M=M, fs=fs, r=r, f0_min=f0_min, f0_max=f0_max
    )

    f0s = []
    Phis = []
    for batch in batch_generator(g, batch_size):
        f0_batch, K_batch = zip(*batch)
        f0_batch = jnp.stack(f0_batch)  # (batch_size,)
        K_batch = jnp.stack(K_batch)  # (batch_size, M, M)

        # Compute the Mercer expansion for the batch
        Phi_batch = psd_svd(
            K_batch, noise_floor_db=noise_floor_db
        )  # (batch_size, M, r)

        f0s.append(f0_batch)
        Phis.append(Phi_batch)

    f0 = jnp.concatenate(f0s)  # (I,)
    Phi = jnp.vstack(Phis)  # (I, M, r)
    return f0, Phi


def periodic_mock_data(key, f0, Phi, noise_db=-60.0):
    """Pick a random frequency from the f0 series and sample a data vector x from the corresponding periodic kernel"""
    k1, k2, k3 = jax.random.split(key, 3)

    i = jax.random.randint(k1, shape=(), minval=0, maxval=f0.shape[0])
    e = jax.random.normal(k2, shape=(Phi[i].shape[1],))  # (r,)
    x = Phi[i] @ e  # (M,)
    x += jax.random.normal(k3, shape=x.shape) * 10 ** (noise_db / 20)
    return f0[i], x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    jax.config.update("jax_enable_x64", True)

    f0, K = periodic_kernel(I=10)
    f0_2, Phi = periodic_kernel_phi(I=10, batch_size=3, noise_floor_db=-90.0)

    # Test if the two methods yield the same f0s
    assert jnp.allclose(f0, f0_2), "F0 series do not match!"
    print("F0 series:", f0)

    # Test if Phi is approximately a square root of K
    K_approx = Phi @ jnp.swapaxes(Phi, -1, -2)
    err = jnp.max(jnp.abs(K - K_approx))
    print("Max absolute reconstruction error:", err)

    # %%

    # Plot the first kernel
    plt.imshow(K[0], aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"Periodic kernel for F0 = {f0[0]:.2f} Hz")
    plt.show()

    # Plot the last kernel
    plt.imshow(K[-1], aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"Periodic kernel for F0 = {f0[-1]:.2f} Hz")
    plt.show()

    # %%
    f0i, x = periodic_mock_data(jax.random.PRNGKey(1234), f0, Phi)

    plt.plot(x)
    plt.title(f"Sampled data vector for F0 = {f0i:.2f} Hz")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()