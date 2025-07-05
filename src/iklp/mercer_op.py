"""Mercer operator for the IKLP problem"""
# %%

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import struct

from .hyperparams import Hyperparams, random_periodic_kernel_hyperparams


def build_X(x, P):
    """Eq. (3) in Yoshii & Goto (2013)"""
    col = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), x[:-1]])
    row = jnp.zeros(P, dtype=x.dtype)
    return jla.toeplitz(col, row)  # (M, P)


@struct.dataclass
class Data:
    """Precompute functions of the data and hyperparameters

    Note: the `woodbury_ratio` is a quick indicator of whether `L = I*r`
    is small enough for the Woodbury trick to be efficient. It is computed
    as `L / M` where `M` is the number of samples. It is not used in the code.

    Rough guideline for woodbury_ratio (empirical):
        ratio ≤ 1.5         Woodbury clearly faster than CG
        1.5 < ratio < 2     about equal, still safe
        ratio > 2           consider CG / SLQ / Hutchinson
    """

    h: Hyperparams  # back‑link to shared Hyperparams
    X: jnp.ndarray  # (M,P)
    x: jnp.ndarray  # (M,)
    Phi_cat: jnp.ndarray  # (M,L)
    Gram: jnp.ndarray  # (L,L)  = ΦᵀΦ
    B0: jnp.ndarray  # (L,P)  = ΦᵀX
    w0: jnp.ndarray  # (L,)   = Φᵀx
    woodbury_ratio: float  # L / M


def build_data(x, h: Hyperparams) -> Data:
    I, M, r = h.Phi.shape
    L = I * r
    X = build_X(x, h.P)
    Phi_cat = jnp.transpose(h.Phi, (1, 0, 2)).reshape(M, L)  # Horizontal cat
    Gram = Phi_cat.T @ Phi_cat
    B0 = Phi_cat.T @ X
    w0 = Phi_cat.T @ x
    woodbury_ratio = L / M
    return Data(h, X, x, Phi_cat, Gram, B0, w0, woodbury_ratio)


@struct.dataclass
class MercerOp:
    """Exact representation of "Mercer operator" via Woodbury tricks

    A "Mercer Operator" is a linear operator of the form

        S = nu * I + Σ_i w_i * Phi_i @ Phi_iᵀ

    where Phi_i (M, r) are assumed to be tall and skinny (M >> r) and the Woodbury
    implementation is efficient when `data.woodbury_ratio` is <= 2.0 (roughly).

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
    chol = jla.cholesky(A, lower=True)
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
    L = jla.cholesky(H, lower=True)

    y = jla.solve_triangular(L, r, lower=True)
    a = jla.solve_triangular(L.T, y, lower=False)
    return a  # (P,)


def sample(op: MercerOp, key, shape=()):
    """Sample from MvNormal(0, op)"""
    k1, k2 = jax.random.split(key)
    M, L = op.Phi_w.shape

    z0 = jax.random.normal(k1, shape + (M,))
    z1 = jax.random.normal(k2, shape + (L,))

    eps = jnp.sqrt(op.nu) * z0 + jnp.matmul(z1, op.Phi_w.T)
    return eps  # (..., M)


if __name__ == "__main__":
    from . import mercer

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    key = jax.random.PRNGKey(0)

    def sk():
        global key
        key, k = jax.random.split(key)
        return k

    # Mock data and hyperparameters
    I = 40
    M = 1024
    P = 30
    nu = 1.56
    lam = 0.1

    kernel_kwargs = {
        "noise_floor_db": -60.0,
    }

    hyper_kwargs = {
        "P": P,
        "lam": lam,
    }

    h, K = random_periodic_kernel_hyperparams(
        sk(), I, M, kernel_kwargs, hyper_kwargs, return_K=True
    )

    print("K shape:", K.shape)

    x = jax.random.normal(sk(), (M,))
    coeff = jax.random.normal(sk(), (I,)) ** 2
    v = jax.random.normal(sk(), (M,))

    # Compute ground truth
    X = build_X(x, P)
    S = nu * jnp.eye(M)
    for i in range(I):
        S += coeff[i] * K[i]

    Sinv_explicit = jla.inv(S)
    logdet_exp = 2 * jnp.sum(jnp.log(jnp.diag(jla.cholesky(S, lower=True))))
    trinv_exp = jnp.trace(Sinv_explicit)
    trinv_Ki_exp = jnp.trace(Sinv_explicit @ K, axis1=1, axis2=2)

    def compute_naieve_a(S_inv, X, P, lam):
        """Uses Sigma^(-1) == S^(-1) -- see _compute_sigma_inv_with_upsilon()"""
        A = X.T.dot(S_inv).dot(X) + lam * jnp.eye(P)
        b = X.T.dot(S_inv).dot(x)
        a = jnp.linalg.solve(A, b)
        return a

    a_exp = compute_naieve_a(Sinv_explicit, X, P, lam)

    # As noise_floor_db increases, calculations become more exact while Woodbury becomes less efficient
    errs = []
    db = 0.0

    for _ in range(5):
        db -= 20.0

        print()
        print(f"Noise floor: {db:.0f} dB")

        Phi = mercer.psd_svd(K, noise_floor_db=db)
        h = h.replace(Phi=Phi)
        print("Phi shape:", Phi.shape)

        data = build_data(x, h)
        print("Woodbury ratio L/M:", data.woodbury_ratio)

        op = build_operator(nu, coeff, data)

        err = [
            jnp.max(jnp.abs(solve(op, v) - Sinv_explicit @ v)),
            jnp.abs(logdet(op) - logdet_exp),
            jnp.abs(trinv(op) - trinv_exp),
            jnp.max(jnp.abs(trinv_Ki(op) - trinv_Ki_exp)),
            jnp.max(jnp.abs(solve_normal_eq(op, lam) - a_exp)),
        ]
        errs.append(err)

        print("\tmax |S⁻¹v - exact|:", err[0])
        print("\tlogdet diff:", err[1])
        print("\ttrinv   diff:", err[2])
        print("\ttrinv_Ki max diff:", err[3])
        print("\ta max diff:", err[4])
