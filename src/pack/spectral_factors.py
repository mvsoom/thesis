# %%
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

from pack.zonal_coeffs import complex_coeffs


def make_legendre_quadrature(t1, t2, n_quad=128):
    """Gauss-Legendre points/weights on [t1, t2]."""
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
    # scale from [-1,1] to [t1,t2]
    x = 0.5 * (t2 - t1) * nodes_np + 0.5 * (t2 + t1)
    w = 0.5 * (t2 - t1) * weights_np
    return jnp.asarray(x), jnp.asarray(w)


@jax.jit
def _I_table(m_vals, omega_vals, x_nodes, w_nodes):
    """
    Base integrals for n=0:

        I_m(ω) = ∫_{t1}^{t2} (1 + x^2)^(-1) exp(i m arctan x - i ω x) dx

    Returns array of shape (M, W), with M = len(m_vals), W = len(omega_vals).
    """
    theta = jnp.arctan(x_nodes)  # (Nq,)
    base = w_nodes / (1.0 + x_nodes**2)  # (Nq,)

    # m axis -> 0, ω axis -> 1, quadrature -> 2
    phase = jnp.exp(
        1j
        * (
            m_vals[:, None, None] * theta[None, None, :]
            - omega_vals[None, :, None] * x_nodes[None, None, :]
        )
    )  # (M, W, Nq)

    integrand = base[None, None, :] * phase  # (M, W, Nq)
    return jnp.sum(integrand, axis=-1)  # (M, W)


def _boundary_term_vec(n, theta, m_vals, omega_vals):
    """
    Vectorized boundary term:

      -1/(i ω) * sec(theta)^n * exp(i m theta - i ω tan(theta))
    """
    m_col = m_vals[:, None]  # (M,1)
    ω_row = omega_vals[None, :]  # (1,W)
    sec = 1.0 / jnp.cos(theta)
    return (
        (-1.0)
        / (1j * ω_row)
        * sec**n
        * jnp.exp(1j * m_col * theta - 1j * ω_row * jnp.tan(theta))
    )  # (M,W) when m_vals is full, or (K,W) when sliced


@partial(jax.jit, static_argnums=(2, 3))
def _build_H_from_I(I_mat, omega_vals, m_max, n_max, theta1, theta2):
    """
    Core recurrence builder for H_n(m, ω) given I_mat for n=0.

    I_mat: shape (2*m_max+1, W), for m in [-m_max..+m_max]
    omega_vals: shape (W,)
    theta1, theta2: scalars = arctan(t1), arctan(t2)

    Returns H with shape (n_max+1, 2*m_max+1, W).
    """
    M = 2 * m_max + 1
    W = omega_vals.shape[0]
    m_vals = jnp.arange(-m_max, m_max + 1)  # (M,)

    H = jnp.full((n_max + 1, M, W), jnp.nan + 0j, dtype=I_mat.dtype)

    # level n = 0
    diff0 = _boundary_term_vec(
        0, theta2, m_vals, omega_vals
    ) - _boundary_term_vec(0, theta1, m_vals, omega_vals)  # (M,W)
    H0 = diff0 + (m_vals[:, None] / omega_vals[None, :]) * I_mat
    H = H.at[0].set(H0)

    # higher levels
    for n in range(1, n_max + 1):
        idx = jnp.arange(n, M - n)  # valid m-shifts at this level
        m_sub = m_vals[idx]  # (K,)

        diff = _boundary_term_vec(
            n, theta2, m_sub, omega_vals
        ) - _boundary_term_vec(n, theta1, m_sub, omega_vals)  # (K,W)

        H_prev = H[n - 1]
        H_p = H_prev[idx + 1, :]  # m+1
        H_m = H_prev[idx - 1, :]  # m-1

        recur = (
            0.5
            / omega_vals[None, :]
            * ((m_sub[:, None] - n) * H_p + (m_sub[:, None] + n) * H_m)
        )  # (K,W)

        H = H.at[n, idx, :].set(diff + recur)

    return H


def build_H_table(omega_vals, t1, t2, m_max, n_max, n_quad=128):
    """
    Public entry point.

    Return:
      m_vals: jnp.arange(-m_max, m_max+1)             (M,)
      H:      jnp.array, shape (n_max+1, M, W)        (n,m_shift,omega)
    """
    x_nodes, w_nodes = make_legendre_quadrature(t1, t2, n_quad)
    m_vals = jnp.arange(-m_max, m_max + 1)
    I_mat = _I_table(m_vals, omega_vals, x_nodes, w_nodes)  # (M,W)

    theta1 = jnp.arctan(t1)
    theta2 = jnp.arctan(t2)

    H = _build_H_from_I(I_mat, omega_vals, m_max, n_max, theta1, theta2)
    return m_vals, H


def H_single_numeric(n, m, omega, t1, t2, tol=1e-12):
    """
    Numerically compute one element H_m^{(n)}(omega; t1, t2):

        H_m^{(n)}(omega) =
            integral_{t1}^{t2} (1 + x^2)^(n/2)
                                * exp(i m arctan x)
                                * exp(-i omega x)
                                dx

    Returns a complex scalar.
    """

    def integrand_real(x):
        return (1.0 + x * x) ** (n / 2.0) * np.cos(m * np.arctan(x) - omega * x)

    def integrand_imag(x):
        return (1.0 + x * x) ** (n / 2.0) * np.sin(m * np.arctan(x) - omega * x)

    re, err_r = quad(integrand_real, t1, t2, epsabs=tol, epsrel=tol)
    im, err_i = quad(integrand_imag, t1, t2, epsabs=tol, epsrel=tol)

    return re + 1j * im


def build_degree_block(d, omega_vals, t1, t2, m_max, n_quad=128):
    """
    Extract H_m^{(d)}(ω) for a single degree d, trimming off the NaN
    edges where the recurrence is not defined.

    Returns:
        m_vals_trimmed : shape (M_trim,)
        H_d_trimmed    : shape (M_trim, W)
    """
    m_vals, H = build_H_table(omega_vals, t1, t2, m_max, n_max=d, n_quad=n_quad)

    H_d = H[d]  # shape (2*m_max+1, W)

    # Find valid m indices: recurrence defines |m| <= m_max - d
    # m_vals runs from -m_max to +m_max
    valid_mask = jnp.abs(m_vals) <= (m_max - d)

    # Apply the mask
    m_vals_trim = m_vals[valid_mask]  # (M_trim,)
    H_d_trim = H_d[valid_mask, :]  # (M_trim, W)

    return m_vals_trim, H_d_trim


def build_pack_factors(d, omega_vals, t1, t2, m_max, n_quad=128):
    """
    Return:
        m_vals_trim  : (M_trim,)
        B            : (M_trim, W)

    where B[m,w] = sqrt(c_m^{(d)}) * H_m^{(d)}(omega_w)

    This avoids NaNs and ensures B is clean for forming K_ff.
    """
    m_vals_trim, H_d_trim = build_degree_block(
        d, omega_vals, t1, t2, m_max, n_quad
    )

    # complex Fourier coefficients c_m^{(d)} for m in |m| <= m_max
    c_full = complex_coeffs(d, m_max)  # shape (2*m_max+1,)

    # match trimmed m indices
    # m_vals_trim indexes into c_full by offsetting with m_max
    idx_full = (m_vals_trim + m_max).astype(int)
    c_trim = jnp.asarray(c_full[idx_full])  # (M_trim,)

    B = jnp.sqrt(c_trim)[:, None] * H_d_trim  # (M_trim, W)

    return m_vals_trim, B


def build_pack_ktilde(d, omega_vals, t1, t2, m_max, n_quad=128):
    """
    Spectral covariance K_ff = B^H B: complex hermitian.
    """
    _, B = build_pack_factors(d, omega_vals, t1, t2, m_max, n_quad)
    K_ff = B.conj().T @ B
    return K_ff


# %%

# time bounds in MILLISECONDS
t1 = -3.25
t2 = +3.25
t_c = t2 - t1

fs = 16000  # Hz

# choose period T > t_c in MILLISECONDS
T = 9.0

F0 = 1000.0 / T  # Hz

num_harmonics = int(np.floor((fs / F0) / 2))

print(
    f"Open phase: {t_c:.2f}msec embedded in period of {T:.2f} msec [F0 = {F0:.3f} Hz => {num_harmonics} harmonics]"
)

assert t_c <= T, "Interval length exceeds period T"

# harmonics
f_vals = jnp.arange(1, num_harmonics + 1) / T  # Hz
omega_vals = 2 * jnp.pi * f_vals  # rad/s

# quadrature
Nquad = 128
x_nodes, w_nodes = make_legendre_quadrature(t1, t2, Nquad)

# build H-table
m_max = 500
n_max = 3
m_vals, H = build_H_table(omega_vals, t1, t2, m_max, n_max)

# test element
n = 2
m = 1
k = -4  # means omega = omega_vals[50]

assert -m_max <= m <= m_max, "m out of range"
assert k < num_harmonics, "k out of range"

omega = omega_vals[k]
oracle = H_single_numeric(n, m, float(omega), float(t1), float(t2))
faster = H[n, m + m_max, k]
error = abs(oracle - faster)

print("error =", error / np.abs(oracle))
print("faster = ", faster)
print("oracle = ", oracle)

# %%
K = build_pack_ktilde(2, omega_vals, t1, t2, m_max)


plt.imshow(K.real)
plt.show()
plt.imshow(K.imag)

# %%
