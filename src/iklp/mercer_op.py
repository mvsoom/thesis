"""Mercer operator for the IKLP problem

TODO(optimize):
- [Likely huge impact] Prune $I$ dimensions when $\theta$ goes under a threshold as done in `~/pro/code/bp_nmf` for the gamma process NMF (GaP-NMF) of Matt Hoffman. Requires altering jitting process, but possible. How do MoE LLMs do this?
- In `mercer_op`: use (JAX) COLA annotations
- In `mercer_op`: can optimize memory and possibly avoid stacking completely
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import struct

from .hyperparams import Hyperparams


def build_X(x, P):
    """Eq. (3) in Yoshii & Goto (2013)"""
    col = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), x[:-1]])
    row = jnp.zeros(P, dtype=x.dtype)
    return jla.toeplitz(col, row)  # (M, P)


def safe_cholesky(A, lower=True, beta=1.0):
    """Cholesky with scaled jitter per common GPyTorch practice"""
    power = jnp.trace(A) / A.shape[0]
    tol = 1e-4 if A.dtype == jnp.float32 else 1e-6
    C = jla.cholesky(
        A + beta * tol * power * jnp.eye(A.shape[0], dtype=A.dtype), lower=lower
    )
    return C


@struct.dataclass
class Data:
    """Precompute functions of the data and hyperparameters

    Note that `L = I*r` where I is the number of theta components and r is the SVD rank of the Phi matrix (I, M, r).
    """

    h: Hyperparams  # back‑link to shared Hyperparams
    X: jnp.ndarray  # (M,P)
    x: jnp.ndarray  # (M,)
    Phi_cat: jnp.ndarray  # (M,L)
    Gram: jnp.ndarray  # (L,L)  = ΦᵀΦ
    B0: jnp.ndarray  # (L,P)  = ΦᵀX
    w0: jnp.ndarray  # (L,)   = Φᵀx


def build_data(x, h: Hyperparams) -> Data:
    x = jnp.asarray(x, dtype=h.Phi.dtype)
    I, M, r = h.Phi.shape
    L = I * r
    X = build_X(x, h.P)
    Phi_cat = jnp.transpose(h.Phi, (1, 0, 2)).reshape(M, L)  # Horizontal cat
    Gram = Phi_cat.T @ Phi_cat
    B0 = Phi_cat.T @ X
    w0 = Phi_cat.T @ x
    return Data(h, X, x, Phi_cat, Gram, B0, w0)


@struct.dataclass
class MercerOp:
    """Exact representation of "Mercer operator" via Woodbury tricks

    A "Mercer Operator" is a linear operator of the form

        S = Σ_i w_i * Phi_i @ Phi_iᵀ + nu * I

    where Phi_i (M, r) are assumed to be tall and skinny (M >> r) and the Woodbury
    implementation is efficient when the Woodbury ratio (I*r/M) is <= 2.0 (roughly).

    In other words this is a linear combination of SVD decompositions of the same rank r,
    plus a ridge component nu. All weights w_i must be >= 0.

    Note: this implementation is exact. The only approximation is (possibly) in the decompositions K_i ~= Phi_i @ Phi_iᵀ.
    If Phi_i is a dense Cholesky root, for example, then no approximation is made.
    """

    data: Data  # back‑link to shared Data
    nu: jnp.ndarray  # () - ridge ν (or σ²)
    sqrt_w: jnp.ndarray  # (I*r,)
    Phi_w: jnp.ndarray  # (M,I*r) - Φ_cat @ diag(√α)
    chol_core: jnp.ndarray  # (I*r,I*r) - Cholesky of A = I + (1/ν) Φ_wᵀΦ_w
    Phi_norms: jnp.ndarray  # (I,) - ‖Φ_i‖_F² / ν  (for tr S⁻¹K_i)


def build_operator(
    nu: jnp.ndarray, weights: jnp.ndarray, data: Data
) -> MercerOp:
    """Precompute Woodbury core for the Mercer operator which is reused by all ops"""
    I, M, r = data.h.Phi.shape
    sqrt_w = jnp.repeat(jnp.sqrt(weights), r)  # length I*r
    Phi_w = data.Phi_cat * sqrt_w[None, :]  # (M,I*r)
    Gram_w = sqrt_w[:, None] * data.Gram * sqrt_w[None, :]
    A = jnp.eye(Gram_w.shape[0], dtype=Gram_w.dtype) + Gram_w / nu
    chol = safe_cholesky(A, lower=True)
    Phi_norms = jnp.sum(data.h.Phi**2, axis=(1, 2)) / nu  # (I,)
    return MercerOp(data, nu, sqrt_w, Phi_w, chol, Phi_norms)


def _solve_core(op: MercerOp, y):
    y1 = jla.solve_triangular(op.chol_core, y, lower=True)
    return jla.solve_triangular(op.chol_core.T, y1, lower=False)


def solve(op: MercerOp, v):
    """Compute op⁻¹ @ v"""
    t = op.Phi_w.T @ v
    u = _solve_core(op, t)
    return v / op.nu - (op.Phi_w @ u) / op.nu**2  # (M,)


def solve_mat(op: MercerOp, V):
    """Compute op⁻¹ @ V"""
    return jax.vmap(lambda col: solve(op, col), in_axes=1, out_axes=1)(
        V
    )  # (M, V.shape[1])


def logdet(op: MercerOp):
    M = op.data.h.Phi.shape[1]
    return M * jnp.log(op.nu) + 2 * jnp.sum(
        jnp.log(jnp.diag(op.chol_core))
    )  # ()


def trinv(op: MercerOp):
    Gram_w = op.sqrt_w[:, None] * op.data.Gram * op.sqrt_w[None, :]
    trace = jnp.trace(_solve_core(op, Gram_w))
    M = op.data.h.Phi.shape[1]
    return M / op.nu - trace / op.nu**2  # ()


def trinv_Ki(op: MercerOp):
    G_i = jnp.einsum("ml,imk->ilk", op.Phi_w, op.data.h.Phi)  # (I,I*r,r)
    AinvG_i = jax.vmap(lambda g: _solve_core(op, g), in_axes=0)(
        G_i
    )  # (I,I*r,r)
    corr = jnp.einsum("ilr,ilr->i", G_i, AinvG_i) / op.nu**2
    return op.Phi_norms - corr  # (I,)


def solve_normal_eq(op: MercerOp, lam):
    """Solve for a that minimises ‖x - X a‖_op² + λ‖a‖²"""
    X, x = op.data.X, op.data.x  # (M,P), (M,)

    SinvX = solve_mat(op, X)  # (M,P)
    Sinvx = solve(op, x)  # (M,)

    G = X.T @ SinvX  # (P,P)
    r = X.T @ Sinvx  # (P,)

    H = G + lam * jnp.eye(X.shape[1], dtype=X.dtype)
    L = safe_cholesky(H, lower=True)

    y = jla.solve_triangular(L, r, lower=True)
    a = jla.solve_triangular(L.T, y, lower=False)
    return a  # (P,)


def sample_parts(op: MercerOp, key, shape=()):
    """Sample signal and noise parts from MvNormal(0, op)"""
    k1, k2 = jax.random.split(key)
    M, L = op.Phi_w.shape

    z0 = jax.random.normal(k1, shape + (L,))
    z1 = jax.random.normal(k2, shape + (M,))

    signal = jnp.matmul(z0, op.Phi_w.T)  # (..., M)
    noise = jnp.sqrt(op.nu) * z1  # (..., M)

    return signal, noise  # both are (..., M)


def sample(op: MercerOp, key, shape=()):
    """Sample from MvNormal(0, op)"""
    signal, noise = sample_parts(op, key, shape)
    return signal + noise  # (..., M)


def matmul_parts(op: MercerOp, v):
    """Compute op @ v, returning the parts (signal, noise) separately"""
    Phi = op.data.Phi_cat
    w = op.sqrt_w * op.sqrt_w
    s = Phi.T @ v
    signal = Phi @ (w * s)
    noise = op.nu * v  # (M,)
    return signal, noise


def sample_parts_given_observation(op: MercerOp, x, key) -> jnp.ndarray:
    """Given observed data x ~ MvNormal(0, op), sample (signal, noise) | (signal + noise = x)"""
    signal0, noise0 = sample_parts(op, key)

    residual = x - (signal0 + noise0)
    c = solve(op, residual)

    # We can do a quick shortcut here because there are only two components so get one in terms of the other...
    c_noise = op.nu * c  # (M,)

    noise = noise0 + c_noise
    signal = x - noise
    return signal, noise  # both are (M,)

    # ... and here is the full version (tested: works!)
    c_signal, c_noise = matmul_parts(op, c)

    signal = signal0 + c_signal
    noise = noise0 + c_noise
    return signal, noise  # both are (M,) and sum to x
