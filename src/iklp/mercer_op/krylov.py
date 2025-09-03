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


def solve(op: MercerKrylovOp, b, tol=None, maxiter=512):
    if tol is None:
        tol = _default_cg_tol(b.dtype)
    Aop = lambda x: matvec(op, x)
    Mop = lambda x: _prec_apply(op, x)
    x, info = jax_cg(Aop, b, tol=tol, maxiter=maxiter, M=Mop)
    return x


def solve_mat(op: MercerKrylovOp, B, tol=None, maxiter=512):
    if tol is None:
        tol = _default_cg_tol(B.dtype)
    return jax.vmap(
        lambda col: solve(op, col, tol=tol, maxiter=maxiter),
        in_axes=1,
        out_axes=1,
    )(B)


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
    """Precompute Krylov subspace sketch"""
    h = data.h

    Phi = h.Phi
    I, M, r = Phi.shape

    nprobe = h.krylov.nprobe
    m = h.krylov.lanczos_iter
    key = h.krylov.key

    # Jacobi preconditioner diag(S) = nu + sum_i w_i * row_norms_sq(Phi_i)
    rn = _row_norms_sq(Phi)  # (I,M)
    diag = nu + jnp.sum(w[:, None] * rn, axis=0)

    pc = Precond(diag=diag)

    # Probes and Lanczos sketch (always cached)
    idx = jnp.arange(nprobe)
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(idx)  # (p,2)
    Zt = jax.vmap(lambda k: jax.random.rademacher(k, (M,), dtype=Phi.dtype))(
        keys
    )  # (p,M)
    Z = Zt.T  # (M,p)
    Q = _normalize_cols(Z)

    # temp op only for matvec in Lanczos
    tmp_op = MercerKrylovOp(
        data=data, nu=nu, w=w, r=r, I=I, M=M, pc=pc, sketch=None, m=m
    )

    # vmap over probes, get coeffs and bases; Qs_j has shape (m, M)
    al, be, Qs = jax.vmap(
        lambda q: _lanczos_coeffs_and_basis(tmp_op, q, m),
        in_axes=1,
        out_axes=(0, 0, 0),
    )(Q)  # al:(p,m) be:(p,m-1) Qs:(p,m,M)
    V = jnp.transpose(Qs, (2, 1, 0))  # (M, m, p)

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


def trinv_Ki(op: MercerKrylovOp):
    Phi, I = op.data.h.Phi, op.I
    Z = op.sketch.Z  # (M,p)

    # per-probe tiny solves: T_j^{-1} e1
    p, m = op.sketch.alphas.shape
    e1 = jnp.zeros((m,), dtype=Phi.dtype).at[0].set(1.0)

    def solve_small(j):
        a = op.sketch.alphas[j]
        b = op.sketch.betas[j]
        T = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
        return jnp.linalg.solve(T, e1)  # (m,)

    t_inv_e1 = jax.vmap(solve_small)(jnp.arange(p))  # (p,m)
    z_norms = jnp.linalg.norm(Z, axis=0)  # (p,)

    # V: (M, m, p), t_inv_e1: (p, m), z_norms: (p,)
    # Y = sum_k V[:,k,:] * t_inv_e1[:,k] * ||z||
    Y = jnp.einsum("akp,pk,p->ap", op.sketch.V, t_inv_e1, z_norms)  # (M,p)

    def one_i(i):
        P = jnp.matmul(Phi[i].T, Z)  # (r,p)
        Q = jnp.matmul(Phi[i].T, Y)  # (r,p)
        return jnp.mean(jnp.sum(P * Q, axis=0))

    return jax.vmap(one_i)(jnp.arange(I))


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
