# %%
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

A = 0.5  # Bounds of integration [-A, A] in x space
M_MAX = 30  # Determines number of zonal harmonics m = (-M_MAX:M_MAX+1)
N_MAX = 3  # Maximum order of the arccos kernel
MPLUS_MAX = M_MAX + N_MAX  # Broaden recurrence triangle at the bottom (n = 0)
M_VEC = jnp.arange(-MPLUS_MAX, MPLUS_MAX + 1)

P_MAX = 256  # Number of harmonics omega = 2*pi*(1:P_MAX+1)
OMEGA_FULL = 2.0 * jnp.pi * jnp.arange(1, P_MAX + 1)  # Harmonics for T=1 period


N_QUAD = 256  # Number of quadrature nodes for Legendre-Gauss quadrature for (n = 0) case
nodes, wts = np.polynomial.legendre.leggauss(N_QUAD)
X_NODES = A * jnp.asarray(nodes)
W_NODES = A * jnp.asarray(wts)
THETAS = jnp.arctan(X_NODES)
BASE = W_NODES / (1.0 + X_NODES**2)


def _boundary_value(n: int, theta: float, m_vec: jnp.ndarray):
    o = OMEGA_FULL
    return (
        (-1j * o) ** -1
        * (1.0 / jnp.cos(theta)) ** n
        * jnp.exp(1j * m_vec[:, None] * theta - 1j * o * jnp.tan(theta))
    )  # (len(m_vec), P_MAX)


def _boundary_term(n: int, m_vec: jnp.ndarray):
    # Exactly real due to symmetric bounds [-A, A]
    THETA = jnp.arctan(A)
    return (
        _boundary_value(n, THETA, m_vec) - _boundary_value(n, -THETA, m_vec)
    ).real


@jax.jit  # cache this
def _compute_H():
    """Compute H[i, j, k] full matrix by fast and stable recurrence on n

    that is ∫(1 + x^2)^(n/2) * exp(i m arctan x) * exp(-i omega x) dx

    with n := i, m := j + M_MAX, omega := OMEGA_FULL[k], over [-A, A].
    """

    # Calculate n = 0 case with fast Legendre-Gauss quadrature of the z kernel
    phase = jnp.exp(
        1j
        * (M_VEC[:, None, None] * THETAS - OMEGA_FULL[None, :, None] * X_NODES)
    )
    z_kernel = jnp.sum(
        BASE * phase, -1
    ).real  # Real due to symmetric bounds [-A, A]

    H = jnp.full(
        (N_MAX + 1, 2 * MPLUS_MAX + 1, P_MAX), jnp.nan, dtype=z_kernel.dtype
    )
    diff0 = _boundary_term(0, M_VEC)
    H = H.at[0].set(diff0 + M_VEC[:, None] / OMEGA_FULL * z_kernel)

    # Recurrence relation for n > 0
    for n in range(1, N_MAX + 1):
        idx = jnp.arange(n, 2 * MPLUS_MAX + 1 - n)
        ms = M_VEC[idx]
        diff = _boundary_term(n, ms)
        H = H.at[n, idx].set(
            diff
            + 0.5
            / OMEGA_FULL
            * (
                (ms[:, None] - n) * H[n - 1, idx + 1]
                + (ms[:, None] + n) * H[n - 1, idx - 1]
            )
        )

    # Each increase of +1 in n leaves another pair of m's unknown due to the
    # triangular structure of the recurrence relation, so we leave them out.
    if N_MAX > 0:
        return H[:, N_MAX:-N_MAX, :]
    else:
        return H
    # (N_MAX+1, 2*M_MAX+1, P_MAX)


_H_FULL = _compute_H()


def _compute_H_elem_nintegrate(i, j, k, tol=1e-12):
    """Compute H matrix at [i, j, k] by numerical integration,

    that is ∫(1 + x^2)^(n/2) * exp(i m arctan x) * exp(-i omega x) dx with

    n := i, m := j + M_MAX, omega := OMEGA_FULL[k], over [-A, A].
    """
    n = i
    m = j - M_MAX
    omega = OMEGA_FULL[k]

    def integrand(x):
        return (1 + x**2) ** (n / 2) * np.exp(
            1j * m * np.arctan(x) - 1j * omega * x
        )

    real_part, real_err = quad(
        lambda t: integrand(t).real, -A, A, epsabs=tol, epsrel=tol
    )
    imag_part, imag_err = quad(
        lambda t: integrand(t).imag, -A, A, epsabs=tol, epsrel=tol
    )

    result = (
        real_part + 1j * imag_part
    )  # imag_part is ~= 0. due to symmetric bounds [-A, A]
    return result


def get_nyquist_H_matrix(fs_hz: float, T_sec: float):
    """Get H matrix with aliasing frequencies zeroed out"""
    K_raw = int(np.floor(fs_hz * T_sec / 2))
    K_use = min(max(K_raw, 1), P_MAX)
    if K_raw > P_MAX:
        print(f"p_raw {K_raw} exceeds cap {P_MAX}, truncating")
    return _H_FULL[..., :K_use]


# %%

if __name__ == "__main__":
    import time

    import numpy as np

    H_test = np.full_like(_H_FULL, np.nan, dtype=jnp.complex128)
    it = np.nditer(_H_FULL, flags=["multi_index"])

    timer = time.perf_counter()
    for _ in it:
        i, j, k = it.multi_index
        H_test[i, j, k] = _compute_H_elem_nintegrate(i, j, k)

    elapsed = time.perf_counter() - timer
    print(
        f"Time taken for numerical integration: {elapsed:.2f} seconds"
    )  # Takes 2 hrs

    # Test recurrence formula against numerical result
    diff = np.abs(_H_FULL - H_test)
    max_diff = np.max(diff)
    print(f"Maximum difference between _H_FULL and H_test: {max_diff}")

    # Check if imaginary part is indeed close to zero
    print(f"Maximum imaginary part of H_test: {np.max(np.abs(H_test.imag))}")

# %%
