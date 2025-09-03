"""
Hilbert kernels are reduced-rank approximations to stationary kernels.
Implementation is based on https://arxiv.org/pdf/1401.5508 and Riutort-Mayol et al. (2020).

Note: we don't subclass :class:`Stationary` because
- :class:`Stationary` assume L1 distance by default and the spectral density formulas are based on L2 distance -- we impose that here
- Not all stationary kernels have a known spectral density (e.g. periodic) and we want to define sum and product of Hilbert kernels
- :class:`QuasiSep` does the same
"""

from __future__ import annotations

import jax

__all__ = [
    "Hilbert",
    "Exp",
    "ExpSquared",
    "Matern12",
    "Matern32",
    "Matern52",
]


import equinox as eqx
import jax.numpy as jnp
import numpy as np
from tinygp.helpers import JAXArray
from tinygp.kernels.base import (
    Constant,
    Kernel,
)
from tinygp.kernels.base import (
    Product as BaseProduct,
)
from tinygp.kernels.base import (
    Sum as BaseSum,
)
from tinygp.kernels.distance import L2Distance


class Hilbert(Kernel):
    scale: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def correlation_time_1d(self):
        raise NotImplementedError

    def _unscaled_spectral_density_1d(self, s: JAXArray) -> JAXArray:
        raise NotImplementedError

    def spectral_density_1d(self, s: JAXArray) -> JAXArray:
        return self.scale * self._unscaled_spectral_density_1d(self.scale * s)

    def __add__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, Stationary):
            return StationarySum(self, other)
        return super().__add__(other)

    def __radd__(self, other: Kernel | JAXArray) -> Kernel:
        if other == 0:
            return self
        if isinstance(other, Stationary):
            return StationarySum(other, self)
        return super().__radd__(other)

    def __mul__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, Stationary):
            return StationaryProduct(self, other)
        if isinstance(other, Constant):
            return StationaryScale(base=self, factor=other.value)
        if jnp.ndim(other) == 0:
            return StationaryScale(base=self, factor=other)
        return super().__mul__(other)

    def __rmul__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, Stationary):
            return StationaryProduct(other, self)
        if isinstance(other, Constant):
            return StationaryScale(base=self, factor=other.value)
        if jnp.ndim(other) == 0:
            return StationaryScale(base=self, factor=other)
        return super().__rmul__(other)


class StationaryScale(Stationary):
    base: Stationary = eqx.field(default=None)
    factor: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.factor * self.base.evaluate(X1, X2)

    def spectral_density_1d(self, s: JAXArray) -> JAXArray:
        return self.factor * self.base.spectral_density_1d(s)

    def correlation_time_1d(self):
        return self.base.correlation_time_1d()  # independent of scale


class StationarySum(BaseSum):
    kernel1: Stationary
    kernel2: Stationary

    def spectral_density_1d(self, s: JAXArray) -> JAXArray:
        return self.kernel1.spectral_density_1d(
            s
        ) + self.kernel2.spectral_density_1d(s)

    def correlation_time_1d(self):
        v1 = self.kernel1.evaluate_diag(0)
        v2 = self.kernel2.evaluate_diag(0)
        t1 = self.kernel1.correlation_time_1d()
        t2 = self.kernel2.correlation_time_1d()
        return (v1 * t1 + v2 * t2) / (v1 + v2)


class StationaryProduct(BaseProduct):
    # TODO(mvsoom): test this with known analytical results

    kernel1: Stationary
    kernel2: Stationary

    def _freq_grid(self, s: JAXArray, gridsize=4096, pad=1.5) -> JAXArray:
        s = jnp.atleast_1d(s)
        smax = jnp.max(s) if s.size > 0 else jnp.array(0.0)
        inv1 = 1.0 / jnp.asarray(self.kernel1.scale)
        inv2 = 1.0 / jnp.asarray(self.kernel2.scale)
        W = pad * jnp.maximum(smax, 10.0 * (inv1 + inv2))
        return jnp.linspace(-W, W, gridsize)

    def spectral_density_1d(self, s: JAXArray) -> JAXArray:
        s = jnp.atleast_1d(s)
        w = self._freq_grid(s)
        S1 = self.kernel1.spectral_density_1d(jnp.abs(w))

        def conv_at(si):
            integrand = S1 * self.kernel2.spectral_density_1d(jnp.abs(si - w))
            return jax.scipy.integrate.trapezoid(integrand, w) / (2.0 * jnp.pi)

        return jax.vmap(conv_at)(s)

    def correlation_time_1d(self, tmult=12.0, gridsize=4096):
        # integrate in time domain: τ = ∫_0^∞ (k1(τ)k2(τ))/(k1(0)k2(0)) dτ
        z = jnp.array(0.0)
        v1 = self.kernel1.evaluate_diag(0.0)
        v2 = self.kernel2.evaluate_diag(0.0)

        # pick T from scales; stable for common kernels
        ell_eff = 0.5 * (self.kernel1.scale + self.kernel2.scale)
        T = tmult * ell_eff
        t = jnp.linspace(0.0, T, gridsize)

        def k_at_tau(k, tau):
            return k.evaluate(jnp.array(0.0), tau)

        k1t = jax.vmap(lambda tau: k_at_tau(self.kernel1, tau))(t)
        k2t = jax.vmap(lambda tau: k_at_tau(self.kernel2, tau))(t)
        integrand = (k1t * k2t) / (v1 * v2)
        return jax.scipy.integrate.trapezoid(integrand, t)


class Exp(Stationary):
    r"""The exponential kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        if jnp.ndim(self.scale):
            raise ValueError(
                "Only scalar scales are permitted for stationary kernels; use"
                "transforms.Linear or transforms.Cholesky for more flexiblity"
            )
        return jnp.exp(-self.distance.distance(X1, X2) / self.scale)

    def _unscaled_spectral_density_1d(self, s):
        return 2 / (1 + s**2)

    def correlation_time_1d(self):
        return self.scale * 1.0


class ExpSquared(Stationary):
    r"""The exponential squared or radial basis function kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r^2 / 2)

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
    """

    distance: Distance = eqx.field(default_factory=L2Distance)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = self.distance.squared_distance(X1, X2) / jnp.square(self.scale)
        return jnp.exp(-0.5 * r2)

    def _unscaled_spectral_density_1d(self, s):
        return jnp.sqrt(2 * jnp.pi) * jnp.exp(-(s**2) / 2)

    def correlation_time_1d(self):
        return self.scale * jnp.sqrt(np.pi / 2)


Matern12 = Exp


class Matern32(Stationary):
    r"""The Matern-3/2 kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + \sqrt{3}\,r)\,\exp(-\sqrt{3}\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        arg = np.sqrt(3) * r
        return (1 + arg) * jnp.exp(-arg)

    def _unscaled_spectral_density_1d(self, s):
        return 12 * jnp.sqrt(3) / (3 + s**2) ** 2

    def correlation_time_1d(self):
        return self.scale * 2.0 / jnp.sqrt(3)


class Matern52(Stationary):
    r"""The Matern-5/2 kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + \sqrt{5}\,r +
            5\,r^2/3)\,\exp(-\sqrt{5}\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        arg = np.sqrt(5) * r
        return (1 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)

    def _unscaled_spectral_density_1d(self, s):
        return 400 * jnp.sqrt(5) / (3 * (5 + s**2) ** 3)

    def correlation_time_1d(self):
        return self.scale * 8.0 / (3 * jnp.sqrt(5))


class Cosine(Stationary):
    r"""The cosine kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \cos(2\,\pi\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    Args:
        scale: The parameter :math:`P`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        return jnp.cos(2 * jnp.pi * r)


class ExpSineSquared(Stationary):
    r"""The exponential sine squared or quasiperiodic kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-\Gamma\,\sin^2 \pi r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    Args:
        scale: The parameter :math:`P`.
        gamma: The parameter :math:`\Gamma`.
    """

    gamma: JAXArray | float | None = None

    def __check_init__(self):
        if self.gamma is None:
            raise ValueError("Missing required argument 'gamma'")

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        assert self.gamma is not None
        r = self.distance.distance(X1, X2) / self.scale
        return jnp.exp(-self.gamma * jnp.square(jnp.sin(jnp.pi * r)))


class RationalQuadratic(Stationary):
    r"""The rational quadratic

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + r^2 / 2\,\alpha)^{-\alpha}

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
        alpha: The parameter :math:`\alpha`.
    """

    alpha: JAXArray | float | None = None

    def __check_init__(self):
        if self.alpha is None:
            raise ValueError("Missing required argument 'alpha'")

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        assert self.alpha is not None
        r2 = self.distance.squared_distance(X1, X2) / jnp.square(self.scale)
        return (1.0 + 0.5 * r2 / self.alpha) ** -self.alpha
