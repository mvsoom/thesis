"""Exact Woodbury Mercer operator (L < M regime)"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import struct

from iklp.mercer_op import Data, MercerOp, _phi_t_v, _phi_v, safe_cholesky


@struct.dataclass
class MercerWoodburyOp(MercerOp):
    """Exact representation of "Mercer operator" via Woodbury tricks

    Note: this implementation is exact. The only approximation is (possibly) in the decompositions K_i ~= Phi_i @ Phi_iᵀ.
    If Phi_i is a dense Cholesky root, for example, then no approximation is made.
    """

    chol_core: jnp.ndarray  # (L,L), L=I*r, chol of A = I + (1/nu) Phi_w^T Phi_w
    Phi_norms: jnp.ndarray  # (I,), ||Phi_i||_F^2 / nu
    r: int  # rank per kernel
    L: int  # I*r


def _flatten_blocks(x_ir):
    I, r = x_ir.shape[0], x_ir.shape[1]
    return jnp.reshape(x_ir, (I * r,))


def _unflatten_vec(v, I, r):
    return jnp.reshape(v, (I, r))


def _assemble_core_chol(Phi, w, nu):
    # Phi: (I, M, r), w: (I,)
    # Build G_ij = Phi_i^T Phi_j in one shot: G in (I, I, r, r)
    # Then scale each (i,j) block by sqrt(w_i w_j)/nu and reshape to (L, L).
    I, M, r = Phi.shape
    L = I * r

    # Block Gram: G[i,j] = Phi[i].T @ Phi[j]  -> (I, I, r, r)
    # einsum: sum over m: (i m r) * (j m s) -> (i j r s)
    G_blocks = jnp.einsum("imr,jms->ijrs", Phi, Phi)

    # Scalar weights per block
    sw = jnp.sqrt(w)
    S = (sw[:, None] * sw[None, :]) / nu  # (I, I)

    # Apply weights and reshape to (L, L)
    A_blocks = G_blocks * S[:, :, None, None]  # (I, I, r, r)
    # Reorder blocks to contiguous (i,r, j,s) then flatten
    A_mat = jnp.transpose(A_blocks, (0, 2, 1, 3)).reshape(L, L)

    # Add identity
    A = jnp.eye(L, dtype=Phi.dtype) + A_mat

    chol = safe_cholesky(A, lower=True)
    return chol, L, r


def build_operator(
    nu: jnp.ndarray, weights: jnp.ndarray, data: Data
) -> MercerWoodburyOp:
    """Precompute Woodbury core for the Mercer operator which is reused by all ops"""
    Phi = data.h.Phi  # (I,M,r)
    chol_core, L, r = _assemble_core_chol(Phi, weights, nu)
    Phi_norms = jnp.sum(Phi * Phi, axis=(1, 2)) / nu  # (I,)
    return MercerWoodburyOp(
        data=data,
        nu=nu,
        w=weights,
        chol_core=chol_core,
        Phi_norms=Phi_norms,
        r=r,
        L=L,
    )


def _solve_core(op: MercerWoodburyOp, Y):
    # Solve A Y = B with A = chol_core @ chol_core.T
    Y1 = jla.solve_triangular(op.chol_core, Y, lower=True)
    return jla.solve_triangular(op.chol_core.T, Y1, lower=False)


def solve(op: MercerWoodburyOp, v):
    """Compute op⁻¹ @ v"""
    Phi = op.data.h.Phi
    I, r = Phi.shape[0], op.r
    t_ir = _phi_t_v(Phi, v)  # (I,r)
    t_ir = t_ir * jnp.sqrt(op.w)[:, None]  # weighted
    t = _flatten_blocks(t_ir)  # (L,)
    u = _solve_core(op, t)  # (L,)
    u_ir = _unflatten_vec(u, I, r)  # (I,r)
    z_ir = jnp.sqrt(op.w)[:, None] * u_ir  # (I,r)
    corr = _phi_v(Phi, z_ir)  # (M,)
    return v / op.nu - corr / (op.nu * op.nu)  # (M,)


def solve_mat(op: MercerWoodburyOp, V):
    """Compute op⁻¹ @ V"""
    return jax.vmap(lambda col: solve(op, col), in_axes=1, out_axes=1)(
        V
    )  # (M, V.shape[1])


def logdet(op: MercerWoodburyOp):
    M = op.data.h.Phi.shape[1]
    return M * jnp.log(op.nu) + 2.0 * jnp.sum(
        jnp.log(jnp.diag(op.chol_core))
    )  # ()


def _trace_inv_from_chol(L, block_cols=128):
    # tr(A^{-1}) = ||L^{-1}||_F^2 with A = L L^T
    n = L.shape[0]
    nblocks = (n + block_cols - 1) // block_cols

    def body(b, acc):
        i0 = b * block_cols
        idx = i0 + jnp.arange(block_cols)  # (block_cols,)
        col_mask = (idx < n).astype(L.dtype)  # (block_cols,)

        # RHS E has standard basis columns at rows idx (masked beyond n).
        # one_hot(num_classes=n) requires static n; OK since n = L.shape[0].
        E = jax.nn.one_hot(
            jnp.minimum(idx, n - 1), n, dtype=L.dtype
        ).T  # (n, block_cols)
        E = E * col_mask[None, :]  # zero out overflow cols

        Y = jla.solve_triangular(L, E, lower=True)  # (n, block_cols)
        contrib = jnp.sum(jnp.sum(Y * Y, axis=0) * col_mask)
        return acc + contrib

    return jax.lax.fori_loop(0, nblocks, body, jnp.array(0.0, dtype=L.dtype))


def trinv(op: MercerWoodburyOp):
    # tr(S^{-1}) = (M - L)/nu + (1/nu) tr(A^{-1}), with A = I + (1/nu) Phi_w^T Phi_w
    M = op.data.h.Phi.shape[1]
    Ldim = op.L
    tr_Ainv = _trace_inv_from_chol(op.chol_core)
    return (M - Ldim) / op.nu + tr_Ainv / op.nu  # ()


def trinv_Ki(op: MercerWoodburyOp):
    # tr(S^{-1} K_i) = (1/nu) ||Phi_i||_F^2 - (1/nu^2) * ||L^{-1} B_i||_F^2
    # with B_i = Phi_w^T Phi_i = vertstack_j [ sqrt(w_j) * (Phi_j^T Phi_i) ] in R^{L x r}
    Phi = op.data.h.Phi
    I, M, r = Phi.shape
    L = M * r

    def one_i(i):
        def blk(j):
            return jnp.matmul(Phi[j].T, Phi[i]) * jnp.sqrt(op.w[j])  # (r, r)

        Bij_blocks = jax.vmap(blk)(jnp.arange(I))  # (I, r, r)
        B_i = jnp.reshape(
            Bij_blocks, (I * r, r)
        )  # (L, r), row order matches A rows

        # Frobenius form: tr(B^T A^{-1} B) = ||L^{-1} B||_F^2
        Y = jla.solve_triangular(op.chol_core, B_i, lower=True)  # (L, r)
        corr = jnp.sum(Y * Y)  # scalar

        return op.Phi_norms[i] - corr / (op.nu * op.nu)

    return jax.vmap(one_i)(jnp.arange(I))  # (I,)


def solve_normal_eq(op: MercerWoodburyOp, lam):
    """Solve for a that minimises ‖x - X a‖_op² + λ‖a‖²"""
    X, x = op.data.X, op.data.x  # (M,P), (M,)
    SinvX = solve_mat(op, X)
    Sinvx = solve(op, x)
    G = jnp.matmul(X.T, SinvX)  # (P,P)
    rvec = jnp.matmul(X.T, Sinvx)
    H = G + lam * jnp.eye(X.shape[1], dtype=X.dtype)
    L = safe_cholesky(H, lower=True)
    y = jla.solve_triangular(L, rvec, lower=True)
    a = jla.solve_triangular(L.T, y, lower=False)
    return a  # (P,)
