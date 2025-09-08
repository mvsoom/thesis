from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from flax import struct
from mpmath import digamma, findroot

from utils.jax import maybe32, static_constant

from .mercer import psd_svd
from .util import _periodic_kernel_batch


@struct.dataclass
class ARPrior:
    mean: jnp.ndarray = static_constant(jnp.zeros(30))  # (P,)
    precision: jnp.ndarray = static_constant((1 / 0.1) * jnp.eye(30))  # (P,P)

    @staticmethod
    def yoshii_lambda(P, lam=0.1):
        lam = 0.1
        mu = np.zeros(P)
        Sigma = lam * np.eye(P)
        return ARPrior(mean=mu, precision=jnp.linalg.inv(Sigma))

    def sample(self, key, shape=(), jitter=1e-6):
        Q = self.precision + jitter * jnp.eye(
            self.P, dtype=self.precision.dtype
        )
        L = jnp.linalg.cholesky(Q)
        z = jax.random.normal(
            key, shape + (self.P,), dtype=self.precision.dtype
        )
        y = jnp.linalg.solve_triangular(L, z.T, lower=True).T
        return self.mean + y  # (*shape, P)


@struct.dataclass
class KrylovParams:
    """Configuration for Krylov Mercer operator"""

    nprobe: int = static_constant(16)
    lanczos_iter: int = static_constant(32)

    cg_tol: int | None = static_constant(None)
    cg_maxiter: int = static_constant(512)

    key: jnp.ndarray = jax.random.PRNGKey(0)  # non-static


@struct.dataclass
class Hyperparams:
    """VI hyperparameters

    Both model and simulation hyperparameters are set here.
      * `static_constant`s are specialized on when jitting
      * `maybe32` is used to allow constants to become x32 when jax_enable_x64 is False
        NOTE: Use maybe32() *again* when initializing, e.g. `h = Hyperparams(Phi, aw=maybe32(aw))`
    """

    Phi: jnp.ndarray  # (I,M,r)

    alpha: jnp.ndarray = maybe32(1.0)
    aw: jnp.ndarray = maybe32(1.0)
    bw: jnp.ndarray = maybe32(1.0)
    ae: jnp.ndarray = maybe32(1.0)
    be: jnp.ndarray = maybe32(1.0)

    arprior: ARPrior = struct.field(default_factory=ARPrior)

    smoothness: float = static_constant(100.0)

    num_vi_restarts: int = static_constant(1)
    num_vi_iters: int = static_constant(30)
    vi_criterion: float = static_constant(1e-4)
    num_metrics_samples: int = static_constant(5)

    mercer_backend: str = static_constant(
        "auto"
    )  # "cholesky" (exact method), "woodbury" (exact method), "krylov" (approximate method), "auto" (auto-select based on shape of Phi)

    krylov: KrylovParams = struct.field(default_factory=KrylovParams)


def random_periodic_kernel_hyperparams(
    key, I=32, M=512, kernel_kwargs={}, hyper_kwargs={}, return_K=False
) -> Hyperparams:
    """Cannot vmap this over key because the shape of Phi depends on it"""
    noise_floor_db = kernel_kwargs.pop("noise_floor_db", -60.0)

    T = jnp.sort(jax.random.exponential(key, (I,)) * 10)
    K = _periodic_kernel_batch(T, M, **kernel_kwargs)  # (I, M, M)

    Phi = psd_svd(K, noise_floor_db)

    h = Hyperparams(Phi, **hyper_kwargs)

    return (h, K) if return_K else h


def pi_kappa_hyperparameters(
    Phi, pi: float = 0.5, s: float = 1.0, kappa: float = 1.0, **kwargs
) -> Hyperparams:
    """Create hyperparameters with the pi (pitchedness) and kappa (concentration) parametrization

    By specifiying expected pitchedness,

    Instead of the aw, bw, ae, be parameters, this uses pi and kappa to define the expected pitchedness and total power of the prior distribution.

    Here `pi` constrains the expected pitchedness of the prior , `s` the expected total power, and `kappa` a concentration parameter that determines how peaked the pi and s distributions are around their expectation values.

    Args:
        Phi: Compute Mercer expansion of the K = [..., M, M] matrix using SVD. Shaped (I, M, r)
        pi: Expected pitchedness `nu_w/(nu_w + nu_e)`
        s: Expected total power `nu_w + nu_e`
        kappa: Concentration parameter, higher means more peaked around pi

    Returns:
        Hyperparams: Hyperparams(Phi, ...) with the induced `aw`, `bw`, `ae`, `be` parameters
    """
    aw = kappa * pi
    ae = kappa * (1 - pi)
    bw = kappa / s
    be = kappa / s

    kwargs.update(
        {
            "aw": maybe32(aw),
            "bw": maybe32(bw),
            "ae": maybe32(ae),
            "be": maybe32(be),
        }
    )

    return Hyperparams(Phi, **kwargs)


def expected_entropy_of_normalized_thetas(alpha, I):
    """Exact E[H(p)] for theta/(sum theta) ~ Dirichlet(alpha/I)"""
    return float(digamma(alpha + 1.0) - digamma(1.0 + alpha / I))


def expected_active_components(alpha, I):
    """I_eff_expected = exp(E[H(p)]) for p ~ Dirichlet(alpha/I)"""
    return np.exp(expected_entropy_of_normalized_thetas(alpha, I))


def active_components(theta):
    """I_eff = exp(H(p)) for p = theta/(sum theta)"""
    H = scipy.stats.entropy(theta)
    return np.exp(H)


def solve_for_alpha(I):
    """Solve for alpha such that the expected number of active components under p(theta|I) is "a handful" [exp(1)]"""
    lo, hi = 1e-12, 1.0
    # bracket
    while expected_entropy_of_normalized_thetas(hi, I) < 1.0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Failed to bracket root for EH=1.")
    # root
    try:
        return float(
            findroot(
                lambda x: expected_entropy_of_normalized_thetas(x, I) - 1.0,
                (lo, hi),
            )
        )
    except Exception:
        # fallback bisection
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if expected_entropy_of_normalized_thetas(mid, I) < 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
