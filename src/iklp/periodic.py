# %%
import jax
import jax.numpy as jnp


def f0_series(
    f0_min: float = 100.0,
    f0_max: float = 400.0,
    I: int = 400,
) -> jnp.ndarray:
    """Geometric F0 grid from `f0_min` to `f0_max` (inclusive) with `I` logarithmically spaced steps"""
    ratio = (f0_max / f0_min) ** (1.0 / (I - 1))
    f0s = f0_min * ratio ** jnp.arange(I)
    return f0s # (I,)


def periodic_kernel_batch(
    I: int,
    M: int,
    fs: float | int = 16_000,
    r: float = 1.0,
    f0_min: float = 100.0,
    f0_max: float = 400.0,
):
    """
    Build a batch of MacKay-style periodic kernel matrices

    See periodickernel.jl for the original implementation and why setting r = 1.0 is a good idea.
    """
    f0s = f0_series(f0_min=f0_min, f0_max=f0_max, I=I)
    Ts = 1.0 / f0s

    dt = 1.0 / fs
    t = jnp.arange(M) * dt
    tau = t[None, :, None] - t[None, None, :]
    arg = jnp.pi * tau / Ts[:, None, None]
    K = jnp.exp(-0.5 * (jnp.sin(arg) / r) ** 2)  # (I, M, M)
    return K, f0s


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    K, f0s = periodic_kernel_batch(I=10, M=320, fs=16_000, r=1.0)
    print(
        f"#kernels = {len(f0s)},  1st F0 = {float(f0s[0]):.2f} Hz,  "
        f"last F0 = {float(f0s[-1]):.2f} Hz"
    )
    print(K.shape)

    # Plot the first kernel
    import matplotlib.pyplot as plt
    plt.imshow(K[0], aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"Periodic kernel for F0 = {f0s[0]:.2f} Hz")
    plt.xlabel("Time (samples)")
    plt.ylabel("Time (samples)")
    plt.show()
    
    # Plot the last kernel
    plt.imshow(K[-1], aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.title(f"Periodic kernel for F0 = {f0s[-1]:.2f} Hz")
    plt.xlabel("Time (samples)")
    plt.ylabel("Time (samples)")
    plt.show()
