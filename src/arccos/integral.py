# %%
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------
# 1)  problem parameters  (feel free to change)
# ---------------------------------------------------------------
a = 1  # integrate over x∈[-a, a]
m_max = 30
m_vals = jnp.arange(-m_max, m_max + 1)  # m = 0 … 30   (shape (M,))
f_vals = jnp.arange(1, 9) * jnp.pi  #   (shape (W,))
omega_vals = 2 * jnp.pi * f_vals
Nquad = 128  # 128-point Gauss–Legendre: don't go lower

# ---------------------------------------------------------------
# 2)  Gauss–Legendre nodes/weights on [-1,1]  ->  scale to [-a,a]
# ---------------------------------------------------------------
nodes_np, weights_np = np.polynomial.legendre.leggauss(Nquad)
nodes = jnp.asarray(nodes_np)  # in [-1,1]
weights = jnp.asarray(weights_np)
x_nodes = a * nodes  # in [-a,a]
w_nodes = a * weights  # include Jacobian


# ---------------------------------------------------------------
# 3)  fused integrator  I[m_idx, ω_idx]   (shape (M,W))
# ---------------------------------------------------------------
@jax.jit
def I_table(m_array, omega_array):
    """
    Return matrix I[m_index, omega_index] of shape (len(m_array), len(omega_array))
    """
    # broadcast dimensions:  m → axis 0,  ω → axis 1,  quadrature → axis 2
    theta = jnp.arctan(x_nodes)  # shape (N,)
    base = w_nodes / (1.0 + x_nodes**2)  # 1/(1+x^2) * weight  (N,)

    phase = jnp.exp(  # (M,W,N)
        1j
        * (
            m_array[:, None, None] * theta[None, None, :]  #  + i m arctan x
            - omega_array[None, :, None] * x_nodes[None, None, :]  #  - i ω x
        )
    )

    integrand = base[None, None, :] * phase  # (M,W,N)
    return jnp.sum(integrand, axis=-1)  # integrate over x


# ---------------------------------------------------------------
# 4)  evaluate and show first few rows to check Im≈0
# ---------------------------------------------------------------
I_mat = I_table(m_vals, omega_vals)  # complex matrix

# print first 5 ω for m = 0…5
for j, m in enumerate(m_vals):
    re_part = jnp.real(I_mat[j, :5])
    im_part = jnp.imag(I_mat[j, :5])
    print(f"m={m:2d}  Re:", [f"{v:.6f}" for v in re_part])
    print("      Im:", [f"{v:.6f}" for v in im_part])
    print()

# %%
import jax
import jax.numpy as jnp

# parameters
n = 0
m = -7

omega = omega_vals[0]
x1 = -a
x2 = a  # +1.0

# compute thetas
theta1 = jnp.arctan(x1)
theta2 = jnp.arctan(x2)

print("theta1 =", theta1)  # should be -π/4 ≈ -0.7854
print("theta2 =", theta2)  # should be +π/4 ≈  0.7854


# helper functions
def sec(theta):
    return 1.0 / jnp.cos(theta)


def z(m, theta, omega):
    return jnp.exp(1j * m * theta - 1j * omega * jnp.tan(theta))


def boundary_term(n, theta, m, omega):
    return -1.0 / (1j * omega) * sec(theta) ** n * z(m, theta, omega)


bt_theta1 = boundary_term(n, theta1, m, omega)
bt_theta2 = boundary_term(n, theta2, m, omega)

print("BoundaryTerm at θ1 =", bt_theta1)
print("BoundaryTerm at θ2 =", bt_theta2)

# %%


def H_m_0(m, omega):
    z_val = I_mat[m + m_max, 0]
    return bt_theta2 - bt_theta1 + m / omega * z_val


H_m_0(m, omega)

# %%
import numpy as np
from scipy.integrate import quad


def NH(m, n, omega, x1, x2, tol=1e-12):
    """
    Numerically compute
        ∫ₓ₁^ₓ₂ (1 + x^2)^(n/2) * exp(i m arctan x) * exp(-i 2π f x) dx
    and zero out the result if its magnitude is below `tol`.
    """

    def integrand(x):
        return (1 + x**2) ** (n / 2) * np.exp(
            1j * m * np.arctan(x) - 1j * omega * x
        )

    # integrate real and imag parts separately
    real_part, real_err = quad(
        lambda t: integrand(t).real, x1, x2, epsabs=tol, epsrel=tol
    )
    imag_part, imag_err = quad(
        lambda t: integrand(t).imag, x1, x2, epsabs=tol, epsrel=tol
    )

    result = real_part + 1j * imag_part
    # Chop: if the entire complex number is below tol, return exactly 0
    return result if abs(result) > tol else 0


val = NH(m, n, omega, x1, x2)
print("NH =", val)

# %%

from functools import partial

import jax
import jax.numpy as jnp


# ────────────────────────────────────────────────────────────────
# helpers (unchanged)
# ────────────────────────────────────────────────────────────────
def boundary_term_vec(n, theta, m_vec, omega_vec):
    m_col = m_vec[:, None]  # (M,1)
    ω_row = omega_vec[None, :]  # (1,W)
    return (
        (-1.0)
        / (1j * ω_row)
        * (1.0 / jnp.cos(theta)) ** n
        * jnp.exp(1j * m_col * theta - 1j * ω_row * jnp.tan(theta))
    )


# ────────────────────────────────────────────────────────────────
# main builder – m_max and n_max are *static_argnums*
# ────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnums=(2, 3))  # 2 → m_max, 3 → n_max
def build_H(
    I_mat, omega_vals, m_max: int, n_max: int, theta1: float, theta2: float
):
    """
    Return H[n, m_shift, w] with shape (n_max+1, 2*m_max+1, W).
    Outside the triangle (|m| > m_max-n) we leave NaN.
    """
    M = 2 * m_max + 1
    W = omega_vals.shape[0]
    m_vals = jnp.arange(-m_max, m_max + 1)  # (M,)

    # allocate with NaN (complex)
    H = jnp.full((n_max + 1, M, W), jnp.nan + 0j, dtype=I_mat.dtype)

    # ---------- level n = 0 ----------
    diff0 = boundary_term_vec(
        0, theta2, m_vals, omega_vals
    ) - boundary_term_vec(0, theta1, m_vals, omega_vals)  # (M,W)

    H0 = diff0 + (m_vals[:, None] / omega_vals[None, :]) * I_mat  # (M,W)
    H = H.at[0].set(H0)

    # ---------- higher levels ----------
    for n in range(1, n_max + 1):  # python loop: constant upper bound
        idx = jnp.arange(n, M - n)  # valid columns at this level
        m_sub = m_vals[idx]  # (K,)

        diff = boundary_term_vec(
            n, theta2, m_sub, omega_vals
        ) - boundary_term_vec(n, theta1, m_sub, omega_vals)  # (K,W)

        NH_p = H[n - 1, idx + 1, :]  # m+1
        NH_m = H[n - 1, idx - 1, :]  # m-1

        recur = (
            0.5
            / omega_vals
            * ((m_sub[:, None] - n) * NH_p + (m_sub[:, None] + n) * NH_m)
        )  # (K,W)

        H = H.at[n, idx, :].set(diff + recur)

    return H


# %%
n_max = 3

H = build_H(I_mat, omega_vals, m_max, n_max, theta1, theta2)

print("H shape:", H.shape)  # (4, 61, W)
print("row n=0, m=-3..+3:", H[0, m_max - 3 : m_max + 4, 0])
# %%
# ────────────────────────────────────────────────────────────────
#  Comparison:  build_H  vs.  direct quadrature NH(..)
# ────────────────────────────────────────────────────────────────
import numpy as np

# ---------- build the triangular table with JAX ----------
H_jax = build_H(
    I_mat,  # (2*m_max+1 , W)
    omega_vals,  # (W,)
    m_max,
    n_max,
    theta1,
    theta2,
)  # -> shape (n_max+1 , 2*m_max+1 , W)

# pick the first (only) ω for reference
ω = omega_vals[0]
H0 = np.asarray(H_jax[..., 0])  # numpy view, shape (n_max+1 , 2*m_max+1)

# ---------- direct quadrature for every (n,m) ----------
NH_tab = np.full_like(H0, np.nan + 0j)  # same shape, filled with NaN

for n in range(n_max + 1):
    for m in range(-m_max + n, m_max - n + 1):  # triangle region
        NH_tab[n, m + m_max] = NH(m, n, ω, x1, x2)

# ---------- error metrics ----------
mask = ~np.isnan(NH_tab)  # valid positions inside triangle
abs_err = np.abs(H0 - NH_tab)
rel_err = abs_err / np.maximum(1e-14, np.abs(NH_tab))

print(f"max |H_JAX − NH|     : {abs_err[mask].max():.3e}")
print(f"max relative error   : {rel_err[mask].max():.3e}")

# optional: pretty-print the first two n-levels
for n in range(n_max + 1):
    row_H = H0[n, m_max - 5 : m_max + 6]
    row_NH = NH_tab[n, m_max - 5 : m_max + 6]
    err = abs_err[n, m_max - 5 : m_max + 6]
    print(f"\n── n = {n}  (m = −5…+5) ──")
    for m, h, nh, e in zip(range(-5, 6), row_H, row_NH, err):
        print(f" m={m:+2d}  H={h: .6e}  NH={nh: .6e}  Δ={e:.1e}")

# %%
