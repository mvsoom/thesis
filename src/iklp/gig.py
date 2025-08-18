"""Generalized Inverse Gaussian (GIG) distribution"""

# pip install -Uq tfp-nightly[jax]

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import scipy
import tensorflow_probability.substrates.jax.math as tfm

from utils.jax import maybe32

TAU_CUTOFF = 1e-200


def compute_gig_expectations(gamma, rho, tau):
    """Compute E[x] and E[1/x] for GIG(gamma, rho, tau)

    Here GIG(gamma, rho, tau) is given as:

        p(x) \propto x^(gamma - 1) * exp(-0.5 * (rho * x + tau / x))

    for x > 0, gamma > 0, rho > 0, tau > 0.

    This is the parametrization used by the [GIG wiki page](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution)
    and Yoshii & Goto (2013).

    Note 1: this code is based on Hoffman+ (2010)'s implementation of GaP VI, with following changes:
    - Switched computations for GIG to log regime for more stability
    - Switched from Hoffman's to this parametrization:
        * gamma(Hoffman) = gamma(this)
        * rho(Hoffman) = 2 * rho(this)
        * tau(Hoffman) = 2 * tau(this)

    Note 2: For very small values of tau (<TAU_CUTOFF) and positive values of gamma, the GIG distribution becomes a gamma distribution, and its expectations are both cheaper and more stable to compute that way.
    """

    def gig_branch(args):
        gamma, rho, tau = args

        log_rho = jnp.log(rho)
        log_tau = jnp.log(tau)

        log_sqrt_product = 0.5 * (log_rho + log_tau)
        log_sqrt_ratio = 0.5 * (log_tau - log_rho)

        sqrt_product = jnp.exp(log_sqrt_product)

        log_bm = tfm.log_bessel_kve(gamma - 1.0, sqrt_product)
        log_bb = tfm.log_bessel_kve(gamma + 0.0, sqrt_product)
        log_bp = tfm.log_bessel_kve(gamma + 1.0, sqrt_product)

        Ex = jnp.exp(log_bp - log_bb + log_sqrt_ratio)
        Exinv = jnp.exp(log_bm - log_bb - log_sqrt_ratio)
        return Ex, Exinv

    def gamma_branch(args):
        gamma, rho, _tau = args  # tau is effectively zero
        Ex = 2.0 * gamma / rho
        Exinv = 0.5 * rho / (gamma - 1.0)
        Exinv = jnp.where(Exinv < 0, jnp.inf, Exinv)  # handle gamma ~= 1
        # Exinv = +inf is fine because the ELBO and updates can all be written in terms of Ex and (1/Exinv), the latter then becomes 0
        return Ex, Exinv

    return jax.lax.cond(
        tau > TAU_CUTOFF, gig_branch, gamma_branch, operand=(gamma, rho, tau)
    )


def gig_dkl_from_gamma(Ex, Exinv, rho, tau, a, b):
    """Evaluate D_KL(GIG(a, rho, tau) | Gamma(a, b))

    That is the KL divergence to GIG(a, rho, tau) from Gamma(a, b), where Gamma uses shape, rate parameterization.

    Here Ex = E[x] and Exinv = E[1/x] for GIG(a, rho, tau)
    Note order of GIG (gamma) is set to a, so the E[log x] term cancels out.

    Uses same parametrization as compute_gig_expectations(), and the same notes apply.
    """
    a = maybe32(a)
    b = maybe32(b)

    def gig_branch(args):
        Ex, Exinv, rho, tau, a, b = args

        sqrt_product = jnp.sqrt(rho * tau)
        logK = tfm.log_bessel_kve(a, sqrt_product) - sqrt_product

        return (
            # Linear terms
            (b - 0.5 * rho) * Ex
            - 0.5 * tau * Exinv
            # Nonlinear terms
            - logK
            + jsp.gammaln(a)
            - a * jnp.log(b)
            + 0.5 * a * (jnp.log(rho) - jnp.log(tau))
            # Const
            - jnp.log(2.0)
        )

    def gamma_branch(args):
        Ex, _Exinv, rho, _tau, a, b = (
            args  # Exinv unused, _tau effectively zero
        )
        return (
            # Linear term
            (b - 0.5 * rho) * Ex
            # Nonlinear term
            + a * (jnp.log(rho) - jnp.log(2.0))
            - a * jnp.log(b)
        )

    return jax.lax.cond(
        tau > TAU_CUTOFF,
        gig_branch,
        gamma_branch,
        operand=(Ex, Exinv, rho, tau, a, b),
    )  # >= 0.


def _numpy_rng_from_jax_key(key):
    """https://github.com/jax-ml/jax/discussions/8446#discussioncomment-1584247"""
    return np.random.default_rng(
        np.asarray(key)
    )  # array([k0, k1], dtype=uint32)


def sample_gig(key, gamma, rho, tau, size=None):
    """Sample from GIG(gamma, rho, tau) via SciPy"""
    p_val = gamma
    tau += 1e-10
    b_scipy = jnp.sqrt(rho * tau)
    scale_val = jnp.sqrt(tau / rho)
    gig = scipy.stats.geninvgauss(p_val, b_scipy, scale=scale_val)
    random_state = _numpy_rng_from_jax_key(key)
    return jnp.asarray(gig.rvs(size=size, random_state=random_state))
