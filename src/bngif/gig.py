"""Generalized Inverse Gaussian (GIG) distribution"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import scipy
import tensorflow_probability.substrates.jax as tfp

tfm = tfp.math  # for bessel_kve, log_bessel_kve


def _log_Kv(v: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """log K_v(z) via the scaled TFP kernel:  log K = log kve − |z|."""
    return tfm.log_bessel_kve(v, z) - jnp.abs(z)


def _Kv_ratio(v: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """K_{v+1}(z) / K_v(z) in log-space for stability."""
    return jnp.exp(tfm.log_bessel_kve(v + 1, z) - tfm.log_bessel_kve(v, z))


def _dlogK_dv(v: jnp.ndarray, z: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """∂/∂v log K_v(z) (central finite difference, AD-compatible)."""
    return (_log_Kv(v + eps, z) - _log_Kv(v - eps, z)) / (2 * eps)


def moments(
    p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""E[X], E[1/X], E[log X]  for  GIG(p, a, b)."""
    z = jnp.sqrt(a * b)
    r = _Kv_ratio(p, z)

    mean_x = jnp.sqrt(b / a) * r
    mean_invx = jnp.sqrt(a / b) * r - 2.0 * p / b
    mean_logx = 0.5 * (jnp.log(b) - jnp.log(a)) + _dlogK_dv(p, z)
    return mean_x, mean_invx, mean_logx


def entropy(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    r"""Differential entropy of  X ~ GIG(p,a,b)."""
    z = jnp.sqrt(a * b)
    # log normalizer   log Z = log 2 + log K_p(z) − (p/2)[log a − log b]
    log_Z = jnp.log(2.0) + _log_Kv(p, z) - 0.5 * p * (jnp.log(a) - jnp.log(b))

    mean_x, mean_invx, mean_logx = moments(p, a, b)
    H = log_Z - (p - 1.0) * mean_logx + 0.5 * (a * mean_x + b * mean_invx)
    return H


def E_log_p_under_q(a, b, q: GIG):
    """E_q[ log p ] where q ~ GIG and p ~ Gamma(a, b) uses (shape, rate) parametrization."""
    mean_z, _, mean_logz = q.moments()

    const = a * jnp.log(b) - jss.gammaln(a)
    return const + (a - 1) * mean_logz - b * mean_z


@jax.tree_util.register_pytree_node_class
@dataclass
class GIG:
    p: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray

    def tree_flatten(self):
        return ((self.p, self.a, self.b), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    def moments(self):
        return moments(self.p, self.a, self.b)

    def entropy(self):
        return entropy(self.p, self.a, self.b)

    def KL_from_gamma(self, a, b):
        """KL(GIG|Gamma) where p = Gamma(a, b) uses (shape, rate) parametrization"""
        KL = -E_log_p_under_q(a, b, self) - self.entropy()
        return KL

    def to_scipy(self):
        p_val = self.p
        b_scipy = jnp.sqrt(self.a * self.b)
        scale_val = jnp.sqrt(self.b / self.a)

        return scipy.stats.geninvgauss(p_val, b_scipy, scale=scale_val)


@jax.tree_util.register_pytree_node_class
class Gamma(GIG):
    """Gamma(shape, rate) as a *degenerate* GIG.

    Mapping: Gamma(k, λ)  →  GIG(p=k,  a=2λ,  b→0⁺).
    A tiny positive ε keeps b>0 so √(ab) and Bessel calls stay valid.
    """

    def __init__(self, shape: jnp.ndarray, rate: jnp.ndarray, eps: float = 1e-32):
        a = 2.0 * rate
        b = jnp.full_like(a, eps)
        super().__init__(p=shape, a=a, b=b)

    def _entropy(self):
        k = self.p
        rate = self.a / 2.0
        return k - jnp.log(rate) + jss.gammaln(k) + (1.0 - k) * jss.digamma(k)

    def _to_scipy(self):
        k = float(self.p)
        scale = float(1.0 / (self.a / 2.0))
        return scipy.stats.gamma(a=k, scale=scale)
