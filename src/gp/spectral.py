from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln
from tinygp import transforms
from tinygp.helpers import JAXArray
from tinygp.kernels import Constant, Kernel, stationary
from tinygp.kernels.distance import L2Distance

from gp.util import require_1d


class Spectral:
    r"""Mixin to slap a spectral density on Stationary kernels

    Convention: we use the same convention as Rasmussen & Williams (2006) equation (4.5, p. 82) ...

    .. math::

        k(\tau) = \int S(s) e^{2 \pi i s\cdot\tau} ds,

    ... BUT we set

    .. math::

        omega^2 := 2 \pi s

    so the spectral densities here can be obtained from those in Rasmussen & Williams (2006) by substituting s = omega^2 / (2 pi).

    **Note**: unlike the Stationary kernels, L2 distance is enforced, as the spectral density formulas depend on this.
    """

    def __post_init__(self):
        object.__setattr__(self, "distance", L2Distance())

    def log_radial_spectral_density(
        self, D: int, omega2: float, ell: float
    ) -> JAXArray:
        raise NotImplementedError

    def log_spectral_density(self, s: JAXArray) -> JAXArray:
        """Has same shape semantics as Kernel.evaluate(): let vmap() do the broadcasting and handle only a single frequency vector here"""
        s = require_1d(s)  # (D,)
        D = s.size
        omega2 = jnp.sum(s * s)
        ell = self.scale
        return self.log_radial_spectral_density(D, omega2, ell)

    def spectral_density(self, s: JAXArray) -> JAXArray:
        return jnp.exp(self.log_spectral_density(s))

    def __mul__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, Constant):
            return Scale(factor=other.c, kernel=self)
        elif isinstance(other, Kernel):
            return super().__mul__(other)
        else:  # assume scalar
            return Scale(factor=other, kernel=self)

    def __rmul__(self, other) -> Kernel:
        if isinstance(other, Kernel):
            return super().__rmul__(other)
        return self.__mul__(other)


class Exp(Spectral, stationary.Exp):
    def log_radial_spectral_density(self, D, omega2, ell):
        c = (
            0.5 * D * jnp.log(4.0 * jnp.pi)
            + gammaln(0.5 * (D + 1.0))
            - 0.5 * jnp.log(jnp.pi)
            - jnp.log(ell)
        )
        return c - 0.5 * (D + 1.0) * jnp.log(1.0 / (ell * ell) + omega2)


class ExpSquared(Spectral, stationary.ExpSquared):
    def log_radial_spectral_density(self, D, omega2, ell):
        return (
            0.5 * D * jnp.log(2.0 * jnp.pi)  # was D * log(2*pi)
            + D * jnp.log(ell)
            - 0.5 * (ell**2) * omega2
        )


Matern12 = Exp


class Matern32(Spectral, stationary.Matern32):
    def log_radial_spectral_density(self, D, omega2, ell):
        c = (
            0.5 * D * jnp.log(4.0 * jnp.pi)
            + 1.5 * jnp.log(3.0)
            + gammaln(0.5 * (D + 3.0))
            - gammaln(1.5)  # Gamma(3/2) = sqrt(pi)/2
            - 3.0 * jnp.log(ell)
        )
        return c - 0.5 * (D + 3.0) * jnp.log(3.0 / (ell * ell) + omega2)


class Matern52(Spectral, stationary.Matern52):
    def log_radial_spectral_density(self, D, omega2, ell):
        c = (
            0.5 * D * jnp.log(4.0 * jnp.pi)
            + 2.5 * jnp.log(5.0)
            + gammaln(2.5 + 0.5 * D)
            - gammaln(2.5)
            - 5.0 * jnp.log(ell)
        )
        return c - (2.5 + 0.5 * D) * jnp.log(5.0 / (ell * ell) + omega2)


class Scale(Spectral, Kernel):
    factor: JAXArray
    kernel: Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.factor * self.kernel.evaluate(X1, X2)

    def log_spectral_density(self, s: JAXArray) -> JAXArray:
        return jnp.log(self.factor) + self.kernel.log_spectral_density(s)


class Cholesky(Spectral, transforms.Cholesky):
    """x' = L^{-1} x  =>  S'(s) = |det L| * S(L^T s)"""

    def log_spectral_density(self, s: JAXArray) -> JAXArray:
        # TODO(mvsoom): assume s is (D,)
        # TODO(mvsoom): rewrite everything with bijectors
        s = jnp.atleast_1d(s)  # [..., D]
        D = s.shape[-1]
        L = jnp.asarray(self.factor)

        # coerce L to (D, D)
        if L.ndim == 0:
            L2 = jnp.eye(D) * L
        elif L.ndim == 1:
            if L.shape[0] != D:
                raise ValueError("factor diag length != input dim")
            L2 = jnp.diag(L)
        elif L.ndim == 2:
            if L.shape != (D, D):
                raise ValueError("factor shape must be (D, D)")
            L2 = L
        else:
            raise ValueError("factor must be 0-, 1-, or 2-D")

        mapped = s @ L2  # L^T * omega
        logdet = jnp.sum(jnp.log(jnp.diag(L2)))  # Cholesky: diag > 0
        return logdet + self.kernel.log_spectral_density(mapped)
