"""Mercer operator for the IKLP problem

# Scaling for Mercer ops

## Approx cutoffs

|Var|Small  |Large  |
|--:|:-----:|:-----:|
|`M`|`≲2000`|`≳5000`|
|`r`|`≲50`  |`≳200` |
|`I`|`≲5`   |`≳50`  |

Other factors: precision, GPU vs CPU.

## Reduced-rank GP (I = 1)
|Regime              |Method    |Compute         |Memory              |
|--------------------|----------|---------------:|--------------------|
|`M` small, `r` small|Cholesky  |O(M^3)          |O(M^2)              |
|`M` small, `r` large|Cholesky  |O(M^3)          |O(M^2)              |
|`M` large, `r` small|Woodbury  |O(M r^2 + r^3)  |O(M r + r^2)        |
|`M` large, `r` large|Krylov (≈)|O(M r (p m + k))|O(M r + M p + M m p)|

## VI Mercer mixture (L = I·r)
|Regime              |Method    |Compute                    |Memory                    |
|--------------------|----------|--------------------------:|--------------------------:|
|`M` small           |Cholesky  |O(M^3 + I M^2 r)           |O(M^2 + I M r)            |
|`M` large, `L/M ≲ 2`|Woodbury  |O(M L^2 + L^3)             |O(I M r + L^2)            |
|`M` large, `L/M > 2`|Krylov (≈)|O(I M r (p m + p + k))     |O(I M r + M p + M m p)    |

## Variables

- `M`: number of data points.
- `r`: number of basis functions in reduced-rank GP (per kernel).
- `I`: number of kernels in a Mercer mixture.
- `L = I r`: total reduced rank across all kernels.
- `p`: number of probe vectors for stochastic trace/logdet estimation.
- `m`: Krylov/Lanczos subspace depth.
- `k`: number of CG iterations.


# TODO(optimize):
- [Likely huge impact] Prune $I$ dimensions when $\theta$ goes under a threshold as done in `~/pro/code/bp_nmf` for the gamma process NMF (GaP-NMF) of Matt Hoffman. Requires altering jitting process, but possible. How do MoE LLMs do this?
- In `mercer_op`: use (JAX) COLA annotations
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import struct

from ..hyperparams import ARPrior, Hyperparams


@struct.dataclass
class Data:
    """Precomputed functions of the data and hyperparameters

    Note that `L = I*r` where I is the number of theta components and r is the SVD rank of the Phi matrix (I, M, r).
    """

    h: Hyperparams
    X: jnp.ndarray  # (M,P)
    x: jnp.ndarray  # (M,)


def build_X(x, P):
    """Eq. (3) in Yoshii & Goto (2013)"""
    col = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), x[:-1]])
    row = jnp.zeros(P, dtype=x.dtype)
    return jla.toeplitz(col, row)  # (M, P)


def build_data(x, h: Hyperparams) -> Data:
    x = jnp.asarray(x, dtype=h.Phi.dtype)  # Convert data to chosen precision
    x = x - jnp.mean(x)  # Zero-mean
    P = h.arprior.mean.shape[0]
    X = build_X(x, P)
    return Data(h, X, x)


def safe_cholesky(A, lower=True, beta=1.0):
    """Cholesky with scaled jitter per common GPyTorch practice

    WARNING: disabled this for now because beta=1 (recommended case) causes quite big descrepancies with for trinv_Ki() and trinv() functions
    """
    # return jla.cholesky(A, lower=lower)

    power = jnp.trace(A) / A.shape[0]
    tol = 1e-4 if A.dtype == jnp.float32 else 1e-6
    C = jla.cholesky(
        A + beta * tol * power * jnp.eye(A.shape[0], dtype=A.dtype), lower=lower
    )
    return C


@struct.dataclass
class MercerOp:
    """Exact representation of "Mercer operator" via Woodbury tricks

    A "Mercer Operator" is a linear operator of the form

        S = Σ_i w_i * Phi_i @ Phi_iᵀ + nu * I

    where Phi_i (M, r) are assumed to be tall and skinny (M >> r) and the Woodbury
    implementation is efficient when the Woodbury ratio (I*r/M) is <= 2.0 (roughly).

    In other words this is a linear combination of SVD decompositions of the same rank r,
    plus a ridge component nu. All weights w_i must be >= 0.

    Specific backends are implemented in separate modules in this subpackage.
    """

    data: Data  # backlink to shared Data
    nu: jnp.ndarray  # noise variance
    w: jnp.ndarray  # (I,)


def backend(h: Hyperparams):
    """Dynamic dispath to appropriate Mercer operator backend"""
    from iklp.mercer_op import cholesky, krylov, woodbury

    if h.mercer_backend == "cholesky":
        return cholesky
    elif h.mercer_backend == "woodbury":
        return woodbury
    elif h.mercer_backend == "krylov":
        return krylov
    elif h.mercer_backend == "auto":
        I, M, r = h.Phi.shape
        L = I * r
        if M < L and M < 5000:
            return cholesky
        elif L / M <= 2.0:
            return woodbury
        else:
            return krylov
    else:
        raise ValueError(f"Unknown mercer_backend: {h.mercer_backend}")


def build_operator(nu, w, data) -> MercerOp:
    return backend(data.h).build_operator(nu, w, data)


def solve(op: MercerOp, v):
    return backend(op.data.h).solve(op, v)


def solve_mat(op: MercerOp, V):
    return backend(op.data.h).solve_mat(op, V)


def logdet(op: MercerOp):
    return backend(op.data.h).logdet(op)


def trinv(op: MercerOp):
    return backend(op.data.h).trinv(op)


def trinv_Ki(op: MercerOp):
    return backend(op.data.h).trinv_Ki(op)


def solve_normal_eq(op: MercerOp, arprior: ARPrior):
    """
    Solve the generalized ridge / Gaussian-prior normal equations.

    Minimizes
        (x - X a)^T S^{-1} (x - X a) + (a - mu)^T Q (a - mu),

    which is the MAP estimator under a Gaussian prior a ~ N(mu, Q^{-1}).
    The normal equations are:
        (X^T S^{-1} X + Q) a = X^T S^{-1} x + Q mu.

    Note on lambda conventions: Yoshii & Goto (2013) use lambda for prior covariance; their regularized normal equation is wrong, replace lambda -> 1/lambda

    Returns the minimizer a.
    """
    X, x = op.data.X, op.data.x
    SinvX = solve_mat(op, X)
    Sinvx = solve(op, x)
    G = X.T @ SinvX  # X^T S^{-1} X
    r = X.T @ Sinvx  # X^T S^{-1} x

    Q = arprior.precision
    mu = arprior.mean

    H = G + Q
    b = r + Q @ mu

    L = safe_cholesky(H, lower=True, beta=op.data.h.beta)
    y = jla.solve_triangular(L, b, lower=True)
    a = jla.solve_triangular(L.T, y, lower=False)
    return a


def _phi_t_v(Phi, v):
    def k_dot(i):
        return jnp.matmul(Phi[i].T, v)  # (r,)

    return jax.vmap(k_dot)(jnp.arange(Phi.shape[0]))  # (I,r)


def _phi_v(Phi, u_ir):
    # Phi: (I,M,r), u_ir: (I,r) -> sum_i Phi_i @ u_i
    def k_apply(i):
        return jnp.matmul(Phi[i], u_ir[i])  # (M,)

    return jnp.sum(jax.vmap(k_apply)(jnp.arange(Phi.shape[0])), axis=0)


def matmul_parts(op: MercerOp, v):
    """Compute op @ v, returning the parts (signal, noise) separately"""
    Phi = op.data.h.Phi
    t_ir = _phi_t_v(Phi, v)  # (I,r)
    signal = _phi_v(Phi, op.w[:, None] * t_ir)
    noise = op.nu * v
    return signal, noise


def sample_parts(op: MercerOp, key, shape=()):
    """Sample the signal and noise parts separately"""
    k1, k2 = jax.random.split(key)
    Phi = op.data.h.Phi
    I, M, r = Phi.shape
    z0 = jax.random.normal(k1, shape + (I, r), dtype=Phi.dtype)
    z1 = jax.random.normal(k2, shape + (M,), dtype=Phi.dtype)
    signal = _phi_v(Phi, jnp.sqrt(op.w)[:, None] * z0)
    noise = jnp.sqrt(op.nu) * z1
    return signal, noise


def sample(op: MercerOp, key, shape=()):
    """Sample from MvNormal(0, op)"""
    s, n = sample_parts(op, key, shape)
    return s + n


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
