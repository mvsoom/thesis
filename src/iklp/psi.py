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
