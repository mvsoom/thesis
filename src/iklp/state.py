from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from jax import random

from .gig import compute_gig_expectations, sample_gig
from .hyperparams import Hyperparams
from .mercer_op import (
    Data,
    MercerOp,
    build_data,
    build_operator,
    sample,
    trinv_Ki,
)
from .psi import solve_Psi
from .util import gamma_shape_rate, gamma_shape_scale, stabilize_ar


@struct.dataclass
class LatentVars:
    """Latent variables z = (theta, nu_w, nu_e, a)"""

    theta: jnp.ndarray  # (I,)
    nu_w: jnp.ndarray  # ()
    nu_e: jnp.ndarray  # ()
    a: jnp.ndarray  # (P,)


def sample_z_from_prior(key, h: Hyperparams) -> LatentVars:
    """Sample z ~ p(z) from its prior distribution"""
    k1, k2, k3, k4 = random.split(key, 4)
    I = h.Phi.shape[0]

    theta = gamma_shape_rate(k1, h.alpha / I, h.alpha, (I,))
    nu_w = gamma_shape_rate(k2, h.aw, h.bw)
    nu_e = gamma_shape_rate(k3, h.ae, h.be)
    a = h.arprior.sample(k4)  # (P,)

    return LatentVars(theta, nu_w, nu_e, a)


def sample_x_from_prior(key, h: Hyperparams) -> jnp.ndarray:
    """Data x under the priors (15-17)"""
    k1, k2 = random.split(key, 2)
    z = sample_z_from_prior(k1, h)
    return sample_x_from_z(k2, z, h)


def sample_x_from_z(key, z: LatentVars, h: Hyperparams) -> jnp.ndarray:
    """Sample x|z from Yoshii+ (2013) eq. (14)"""
    # Build Mercer operator for sandwiched covariance matrix from the Phi and the z sample
    M = h.Phi.shape[1]
    data = build_data(
        jnp.zeros((M,)), h
    )  # Not used, but needed for build_operator
    covar = build_operator(z.nu_e, z.nu_w * z.theta, data)

    # Sample from ε ~ MvNormal(0, covar)
    eps = sample(covar, key)

    # Solve x = Ψ^{-1} ε ~ MvNormal(0, Ψ^{-1} covar Ψ^{-T})
    x = solve_Psi(z.a, eps)
    return x


@struct.dataclass
class VariationalParams:
    """Variational params xi describing q(z|xi) = q(theta) q(nu_w) q(nu_e) q(a)"""

    rho_theta: jnp.ndarray  # (I,)
    tau_theta: jnp.ndarray  # (I,)

    rho_w: jnp.ndarray  # ()
    tau_w: jnp.ndarray  # ()

    rho_e: jnp.ndarray  # ()
    tau_e: jnp.ndarray  # ()

    delta_a: jnp.ndarray  # (P,)


def sample_z_from_q(key, xi: VariationalParams, h: Hyperparams) -> LatentVars:
    """Sample z from the variational distribution q(z)"""
    I = h.Phi.shape[0]

    k1, k2, k3 = random.split(key, 3)
    k1s = random.split(k1, I)

    theta = jnp.asarray(
        [
            sample_gig(k1s[i], h.alpha / I, xi.rho_theta[i], xi.tau_theta[i])
            for i in range(I)
        ]
    )

    nu_w = sample_gig(k2, h.aw, xi.rho_w, xi.tau_w)
    nu_e = sample_gig(k3, h.ae, xi.rho_e, xi.tau_e)
    a = xi.delta_a

    return LatentVars(theta, nu_w, nu_e, a)


def sample_x_from_q(key, xi: VariationalParams, h: Hyperparams) -> jnp.ndarray:
    """Data x under the variational posterior q(z|xi)"""
    k1, k2 = random.split(key, 2)
    z = sample_z_from_q(k1, xi, h)
    return sample_x_from_z(k2, z, h)


@struct.dataclass
class VIState:
    data: Data
    xi: VariationalParams


def init_variational_params(key, h: Hyperparams) -> VariationalParams:
    """
    Initialize the variational parameters as in the code implementation of Hoffman+ (2010)
    (which differs from paragraph 4 from that paper). This initialization sets the GIG
    values to be expecting a power unit-normalized signal:

    - E(theta_i) ~= 1/I => sum over i ~= 1.
    - E(nu_w) ~= 1
    - E(nu_e) ~= 1

    The strange 10000 prefactors just make sure that
    - the GIG calculations stay in a numerically safe regime (log_kve() extremely stable); without them we would start
    in the Gamma (tau ~= 0) regime.
    - These expectation values do not depend strongly on gamma_theta, gamma_w, gamma_e parameters.
    """
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    s = h.smoothness
    I, _M, _r = h.Phi.shape

    # E(nu_w) ~= 1/I, E(1/nu_w) ~= I
    rho_theta = 1e4 * I * gamma_shape_scale(k1, s, 1 / s, (I,))
    tau_theta = 1e4 / I * gamma_shape_scale(k2, s, 1 / s, (I,))

    # E(nu_w) ~= 1, E(1/nu_w) ~= 1
    rho_w = 1e4 * gamma_shape_scale(k3, s, 1 / s)
    tau_w = 1e4 * gamma_shape_scale(k4, s, 1 / s)

    # E(nu_e) ~= 1, E(1/nu_e) ~= 1
    rho_e = 1e4 * gamma_shape_scale(k5, s, 1 / s)
    tau_e = 1e4 * gamma_shape_scale(k6, s, 1 / s)

    # E(a)_prior = 0
    P = h.arprior.mean.shape[0]
    delta_a = jnp.zeros((P,))

    return VariationalParams(
        rho_theta, tau_theta, rho_w, tau_w, rho_e, tau_e, delta_a
    )


def init_state(key, x, h: Hyperparams) -> VIState:
    data = build_data(x, h)
    xi = init_variational_params(key, h)
    return VIState(data, xi)


def init_test_stable_state(key, h: Hyperparams, remove_dc=False) -> VIState:
    """Sample xi, then E(x) from q(x|xi) with AR poles stabilized"""
    k1, k2, k3 = random.split(key, 3)

    xi = init_variational_params(k1, h)

    z = sample_z_from_q(k2, xi, h)
    z_stable = z.replace(a=stabilize_ar(z.a))  # Cheat
    x = sample_x_from_z(k3, z_stable, h)

    if remove_dc:
        x = x - jnp.mean(x)

    data = build_data(x, h)

    return VIState(data, xi)


@struct.dataclass
class Expectations:
    """Expectations of GIG for E[x] and E[1/x] for the z"""

    theta: jnp.ndarray  # (I,)
    theta_inv: jnp.ndarray  # (I,)

    nu_w: jnp.ndarray  # ()
    nu_w_inv: jnp.ndarray

    nu_e: jnp.ndarray  # ()
    nu_e_inv: jnp.ndarray  # ()


def compute_expectations(state: VIState) -> Expectations:
    """Compute the GIG expectations from current state"""
    xi = state.xi
    h = state.data.h
    I = h.Phi.shape[0]

    theta, theta_inv = jax.vmap(compute_gig_expectations, in_axes=(None, 0, 0))(
        h.alpha / I, xi.rho_theta, xi.tau_theta
    )
    nu_w, nu_w_inv = compute_gig_expectations(h.aw, xi.rho_w, xi.tau_w)
    nu_e, nu_e_inv = compute_gig_expectations(h.ae, xi.rho_e, xi.tau_e)

    return Expectations(theta, theta_inv, nu_w, nu_w_inv, nu_e, nu_e_inv)


@struct.dataclass
class Auxiliaries:
    """Pure functions of state"""

    E: Expectations
    Omega: MercerOp
    S: MercerOp
    tKi_Omega: jnp.ndarray


def compute_auxiliaries(state: VIState) -> Auxiliaries:
    """Compute the auxiliary variables from state needed for all 4 variational updates and the ELBO"""
    E = compute_expectations(state)
    Omega = build_operator(E.nu_e, E.nu_w * E.theta, state.data)
    S = build_operator(
        1 / E.nu_e_inv, 1 / (E.nu_w_inv * E.theta_inv), state.data
    )
    tKi_Omega = trinv_Ki(Omega)
    return Auxiliaries(E, Omega, S, tKi_Omega)
