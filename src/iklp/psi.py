# %%
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla


def build_Psi(M, a):
    a = jnp.asarray(a)
    col = jnp.concatenate(
        [jnp.array([1.0]), -a, jnp.zeros(M - a.size - 1, dtype=a.dtype)]
    )
    full = jla.toeplitz(col)
    return jnp.tril(full)


def solve_Psi(a: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """Solve x = Ψ(a)^{-1} eps (not exploiting Toeplitz)"""
    M = eps.shape[0]
    Psi = build_Psi(M, a)
    return jla.solve_triangular(Psi, eps, lower=True, unit_diagonal=True)


def psi_matvec_shift(a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    P = a.shape[0]
    y = x
    # y[p:] -= a[p-1] * x[:-p]
    for p in range(1, P + 1):
        y = y.at[p:].add(-a[p - 1] * x[:-p])
    return y


def psi_matvec_fft(a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    P, M = a.shape[0], x.shape[0]
    h = jnp.concatenate([jnp.array([1.0], x.dtype), -a])
    # next power‐of‐two ≥ M+P-1
    n = 1 << ((M + P - 1).bit_length())
    X = jnp.fft.rfft(jnp.pad(x, (0, n - M)))
    H = jnp.fft.rfft(jnp.pad(h, (0, n - h.shape[0])))
    y = jnp.fft.irfft(X * H)
    return y[:M]


def psi_matvec_naive(a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    M = x.shape[0]
    return build_Psi(M, a) @ x


def psi_matvec(
    a: jnp.ndarray, x: jnp.ndarray, mode: str = "shift"
) -> jnp.ndarray:
    if mode == "shift":
        return psi_matvec_shift(a, x)
    elif mode == "fft":
        return psi_matvec_fft(a, x)
    elif mode == "naive":
        return psi_matvec_naive(a, x)
    else:
        raise ValueError("mode must be 'shift', 'fft', or 'naive'")


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from jax import random

    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(4581)

    P = 30
    M = 2048
    lam = 0.1

    key, sub = random.split(key)
    x = random.normal(sub, (M,))

    # a ~ MvNormal(0, lam * I) to avoid unstable filter
    # If unstable, x_true and x_fast will blow up
    key, sub = random.split(key)
    a = jnp.sqrt(lam) * random.normal(sub, (P,))

    # Check matvec
    v_true = build_Psi(M, a) @ x

    psi_shift = jax.jit(lambda a, x: psi_matvec(a, x, mode="shift"))
    psi_fft = jax.jit(lambda a, x: psi_matvec(a, x, mode="fft"))
    psi_naive = jax.jit(lambda a, x: psi_matvec(a, x, mode="naive"))

    v_shift = psi_shift(a, x)
    v_fft = psi_fft(a, x)
    v_naive = psi_naive(a, x)

    print("≈ v_true?                             max-diff")
    print("------------------------------------------------")
    print(f"shift: {jnp.max(jnp.abs(v_shift - v_true)):.3e}")
    print(f"fft:   {jnp.max(jnp.abs(v_fft - v_true)):.3e}")
    print(f"naive: {jnp.max(jnp.abs(v_naive - v_true)):.3e}")

    # Which ones are faster?
    # %timeit psi_shift(a, x) # fastest
    # %timeit psi_fft(a, x,)  # fast
    # %timeit psi_naive(a, x) # very slow

    # Check solve
    eps = random.normal(key, (M,))

    x_true = jnp.linalg.solve(build_Psi(M, a), eps)
    x_fast = solve_Psi(a, eps)

    print("≈ x_true?                             max-diff")
    print("------------------------------------------------")
    print(f"solve: {jnp.max(jnp.abs(x_fast - x_true)):.3e}")
