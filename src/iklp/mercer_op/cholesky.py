"""Exact naive Cholesky Mercer operator (small M regime)"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import struct

from . import Data, MercerOp, safe_cholesky


@struct.dataclass
class MercerCholeskyOp(MercerOp):
    """Naive dense representation of Mercer operator via Cholesky factor."""

    chol: jnp.ndarray  # (M, M) lower Cholesky factor of S
    S: jnp.ndarray  # (M, M) full covariance (optional, can drop for memory)


def build_operator(
    nu: jnp.ndarray, weights: jnp.ndarray, data: Data
) -> MercerCholeskyOp:
    """Form S explicitly and factor it by Cholesky."""
    I, M, r = data.h.Phi.shape

    # Form K_i = Phi_i Phi_i^T
    def one_K(i):
        Phi_i = data.h.Phi[i]  # (M, r)
        return Phi_i @ Phi_i.T

    K = jax.vmap(one_K)(jnp.arange(I))  # (I, M, M)
    S = nu * jnp.eye(M, dtype=data.h.Phi.dtype) + jnp.tensordot(
        weights, K, axes=1
    )
    chol = safe_cholesky(S, lower=True, beta=data.h.beta)
    return MercerCholeskyOp(data=data, nu=nu, w=weights, chol=chol, S=S)


def solve(op: MercerCholeskyOp, v: jnp.ndarray) -> jnp.ndarray:
    """Compute S^{-1} v."""
    y = jla.solve_triangular(op.chol, v, lower=True)
    return jla.solve_triangular(op.chol.T, y, lower=False)


def solve_mat(op: MercerCholeskyOp, V: jnp.ndarray) -> jnp.ndarray:
    """Compute S^{-1} V for multiple RHS."""
    y = jla.solve_triangular(op.chol, V, lower=True)
    return jla.solve_triangular(op.chol.T, y, lower=False)


def logdet(op: MercerCholeskyOp) -> jnp.ndarray:
    """Compute log|S|."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(op.chol)))


def trinv(op: MercerCholeskyOp) -> jnp.ndarray:
    """Compute tr(S^{-1})."""
    Sinv = jla.cho_solve(
        (op.chol, True), jnp.eye(op.S.shape[0], dtype=op.S.dtype)
    )
    return jnp.trace(Sinv)


def trinv_Ki(op: MercerCholeskyOp) -> jnp.ndarray:
    """Compute tr(S^{-1} K_i) for each i."""
    I = op.data.h.Phi.shape[0]

    def one(i):
        Phi_i = op.data.h.Phi[i]  # (M, r)
        Ki = Phi_i @ Phi_i.T
        return jnp.trace(solve_mat(op, Ki))

    return jax.vmap(one)(jnp.arange(I))