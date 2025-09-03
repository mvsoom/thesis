"""Inexact Krylov Mercer operator (L >> M regime)"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import struct
from jax.scipy.sparse.linalg import cg as jax_cg

from iklp.mercer_op import Data, MercerOp, _phi_t_v, _phi_v


@struct.dataclass
class Precond:
    diag: jnp.ndarray  # (M,)


@struct.dataclass
class Sketch:
    alphas: jnp.ndarray  # (p, m)
    betas: jnp.ndarray  # (p, m-1)
    evals: jnp.ndarray  # (p, m)
    w0: jnp.ndarray  # (p, m)
    Z: jnp.ndarray  # (M, p)
    V: jnp.ndarray | None = None  # (M, m, p) optional Lanczos bases (per probe)


@struct.dataclass
class MercerKrylovOp(MercerOp):
    r: int = struct.field(pytree_node=False)
    I: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    pc: Precond
    sketch: Sketch
    m: int = struct.field(pytree_node=False)  # Lanczos num iterations


def _row_norms_sq(Phi):
    return jnp.sum(Phi * Phi, axis=2)  # (I,M)


def _normalize_cols(X):
    nrm = jnp.maximum(jnp.linalg.norm(X, axis=0, keepdims=True), 1e-32)
    return X / nrm


def _eigs_T(al, be):
    T = jnp.diag(al) + jnp.diag(be, 1) + jnp.diag(be, -1)
    evals, U = jnp.linalg.eigh(T)
    w0 = U[0, :] * U[0, :]
    return evals, w0


def _default_cg_tol(dtype):
    """logdet() and trinv() etc are typically within O(1%) accuracy, so we don't need super accurate CG -- this typically cuts runtimes in half"""
    if dtype == jnp.float32:
        return 3e-3
    if dtype == jnp.bfloat16 or dtype == jnp.float16:
        return 1e-2
    return 1e-4  # float64


def matvec(op: MercerKrylovOp, v):
    Phi = op.data.h.Phi
    t = _phi_t_v(Phi, v)  # (I,r)
    signal = _phi_v(Phi, op.w[:, None] * t)  # (M,)
    return op.nu * v + signal


def matmat(op: MercerKrylovOp, V):
    return jax.vmap(lambda col: matvec(op, col), in_axes=1, out_axes=1)(V)


def _prec_apply(op: MercerKrylovOp, v):
    return v / op.pc.diag


def solve(op: MercerKrylov, b, tol=None, maxiter=512, x0=None):
    if tol is None:
        tol = _default_cg_tol(b.dtype)
    Aop = lambda x: matvec(op, x)
    Mop = lambda x: _prec_apply(op, x)
    x, info = jax_cg(Aop, b, tol=tol, maxiter=maxiter, M=Mop, x0=x0)
    return x


def solve_mat(op: MercerKrylov, B, tol=None, maxiter=512, X0=None):
    if tol is None:
        tol = _default_cg_tol(B.dtype)
    if X0 is None:
        # zeros like B, but column-major vmap
        X0 = jnp.zeros_like(B)
    return jax.vmap(
        lambda col, x0: solve(op, col, tol=tol, maxiter=maxiter, x0=x0),
        in_axes=(1, 1),
        out_axes=1,
    )(B, X0)


def _lanczos_coeffs_and_basis(op: MercerKrylovOp, q0, m: int):
    def body(carry, _):
        q_prev, q, beta_prev = carry
        z = matvec(op, q)
        alpha = jnp.vdot(q, z).real
        z = z - alpha * q - beta_prev * q_prev
        beta = jnp.linalg.norm(z)
        q_next = jnp.where(beta > 0, z / beta, z)
        return (q, q_next, beta), (
            alpha,
            beta,
            q,
        )  # also emit current basis vector

    init = (jnp.zeros_like(q0), q0, jnp.array(0.0, dtype=q0.dtype))
    (_, _, _), (alphas, betas, Qs) = jax.lax.scan(body, init, None, length=m)
    # Qs has m vectors stacked along scan axis -> shape (m, M)
    return alphas, betas[:-1], Qs  # (m,), (m-1,), (m, M)


def build_operator(
    nu: jnp.ndarray,
    w: jnp.ndarray,
    data: Data,
) -> MercerKrylovOp:
    """Precompute Krylov subspace sketch with orthogonal Gaussian probes."""
    h = data.h
    Phi = h.Phi
    I, M, r = Phi.shape

    nprobe = h.krylov.nprobe
    m = h.krylov.lanczos_iter
    key = h.krylov.key

    # Jacobi preconditioner
    rn = _row_norms_sq(Phi)  # (I,M)
    diag = nu + jnp.sum(w[:, None] * rn, axis=0)
    pc = Precond(diag=diag)

    # Orthogonal Gaussian probes
    G = jax.random.normal(key, (M, nprobe), dtype=Phi.dtype)  # (M,p)
    Q_ortho, _ = jnp.linalg.qr(G, mode="reduced")  # (M,p), Q^T Q = I_p
    Z = (
        jnp.sqrt(jnp.asarray(M, dtype=Phi.dtype)) * Q_ortho
    )  # (M,p) for Hutchinson
    Q = Q_ortho  # unit columns for Lanczos

    # Temporary op for matvecs during Lanczos
    tmp_op = MercerKrylovOp(
        data=data, nu=nu, w=w, r=r, I=I, M=M, pc=pc, sketch=None, m=m
    )

    # Lanczos per probe; Qs_j has shape (m, M)
    al, be, Qs = jax.vmap(
        lambda q: _lanczos_coeffs_and_basis(tmp_op, q, m),
        in_axes=1,
        out_axes=(0, 0, 0),
    )(Q)  # al:(p,m), be:(p,m-1), Qs:(p,m,M)
    V = jnp.transpose(Qs, (2, 1, 0))  # (M, m, p)

    # Eigen-decomp of small tridiagonals
    ev, w0 = jax.vmap(_eigs_T, in_axes=(0, 0), out_axes=(0, 0))(al, be)

    sketch = Sketch(alphas=al, betas=be, evals=ev, w0=w0, Z=Z, V=V)
    return MercerKrylovOp(
        data=data, nu=nu, w=w, r=r, I=I, M=M, pc=pc, sketch=sketch, m=m
    )


def _trace_f_from_sketch(op: MercerKrylovOp, f):
    # Each probe column was normalized to ||q||=1, corresponding Rademacher z has ||z||^2 = M
    vals = jnp.sum(op.sketch.w0 * f(op.sketch.evals), axis=1)  # (p,)
    M_as = jnp.asarray(op.M, dtype=vals.dtype)
    return M_as * jnp.mean(vals)


def logdet(op: MercerKrylovOp):
    return _trace_f_from_sketch(op, jnp.log)


def trinv(op: MercerKrylovOp):
    return _trace_f_from_sketch(op, lambda x: 1.0 / x)


def trinv_Ki(op: MercerKrylovOp, maxiter=128):
    """Control-variate (CV) estimator for tr(S^{-1} K_i).

    CV uses D^{-1} with D = diag(S). Unbiased:
      E[ z^T S^{-1} K_i z - z^T D^{-1} K_i z ] + tr(D^{-1} K_i)

    Reuses cached probes Z and stored Lanczos bases V to warm-start solves.
    """
    Phi = op.data.h.Phi  # (I,M,r)
    I, M, r = Phi.shape
    Z = op.sketch.Z  # (M,p), orthogonal probes scaled by sqrt(M)
    p, m = op.sketch.alphas.shape

    # diag(S) and its inverse
    Dinv = 1.0 / op.pc.diag  # (M,)

    # build warm start Y0 ≈ S^{-1} Z from the stored Lanczos bases (no CG yet)
    V = op.sketch.V  # (M,m,p)
    e1 = jnp.zeros((m,), dtype=Phi.dtype).at[0].set(1.0)

    def _solve_small(j):
        a = op.sketch.alphas[j]
        b = op.sketch.betas[j]
        T = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
        return jnp.linalg.solve(T, e1)  # (m,)

    t_inv_e1 = jax.vmap(_solve_small)(jnp.arange(p))  # (p,m)
    z_norms = jnp.linalg.norm(Z, axis=0)  # (p,)
    Y0 = jnp.einsum("akp,pk,p->ap", V, t_inv_e1, z_norms)  # (M,p)

    # refine with a few CG iterations (keeps estimator near-unbiased, but fast)
    tol = op.data.h.krylov.cg_tol
    if tol is None:
        # looser on purpose; warm start does most of the work
        tol = _default_cg_tol(Phi.dtype)
    maxiter = op.data.h.krylov.cg_maxiter
    Y = solve_mat(op, Z, tol=tol, maxiter=maxiter, X0=Y0)  # (M,p)

    # control variate pieces, fully vectorized over i and probes
    # P_i = Phi_i^T Z, Q_i = Phi_i^T Y, R_i = Phi_i^T (D^{-1} Z)
    DZ = Dinv[:, None] * Z  # (M,p)
    P = jnp.einsum("imr,mp->irp", Phi, Z)  # (I,r,p)
    Q = jnp.einsum("imr,mp->irp", Phi, Y)  # (I,r,p)
    R = jnp.einsum("imr,mp->irp", Phi, DZ)  # (I,r,p)

    # per-i probe averages of z^T S^{-1} K_i z - z^T D^{-1} K_i z
    resid = jnp.mean(jnp.sum(P * (Q - R), axis=1), axis=1)  # (I,)

    # add closed-form tr(D^{-1} K_i) = tr(Phi_i^T D^{-1} Phi_i)
    rn = jnp.sum(Phi * Phi, axis=2)  # (I,M) = row_norms_sq
    base = jnp.sum(rn * Dinv[None, :], axis=1)  # (I,)

    return base + resid


def solve_normal_eq(op: MercerKrylovOp, lam):
    tol = op.data.h.krylov.cg_tol
    maxiter = op.data.h.krylov.cg_maxiter

    X, x = op.data.X, op.data.x
    SinvX = solve_mat(op, X, tol=tol, maxiter=maxiter)
    Sinvx = solve(op, x, tol=tol, maxiter=maxiter)
    G = jnp.matmul(X.T, SinvX)
    rvec = jnp.matmul(X.T, Sinvx)
    H = G + lam * jnp.eye(X.shape[1], dtype=X.dtype)
    L = jla.cholesky(H, lower=True)
    y = jla.solve_triangular(L, rvec, lower=True)
    a = jla.solve_triangular(L.T, y, lower=False)
    return a
