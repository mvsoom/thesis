# %%
import jax.numpy as jnp
import numpy as np
from jax import random


def gamma_shape_rate(key, k, beta, size=()):
    """Gamma(shape=k, rate=β)"""
    return random.gamma(key, k, shape=size) / beta


def gamma_shape_scale(key, k, theta, size=()):
    """Gamma(shape=k, scale=θ)"""
    return random.gamma(key, k, shape=size) * theta


def _periodic_kernel_batch(Ts, M, dt=1.0, ell=1.0):
    # Ts: [B]
    t = jnp.arange(M) * dt  # [M]
    tau = t[None, :, None] - t[None, None, :]  # [1, M, M]
    arg = jnp.pi * tau / Ts[:, None, None]  # [B, M, M]
    K = jnp.exp(-2 * (jnp.sin(arg) ** 2) / ell**2)  # [B, M, M]
    return K


def stabilize_ar(a: np.ndarray) -> np.ndarray:
    """
    Given AR-coeffs a = [a1, …, aP] for the polynomial
        Q(z) = z^P - a1 z^(P-1) - … - aP,
    this returns new coeffs a_stable so that all roots of Q(z) lie
    inside the unit circle (|z|<=1), by reflecting any |root|>1 to 1/conj(root).
    """
    # form the monic polynomial Q(z) = z^P - a1 z^(P-1) - … - aP
    # its coef vector (highest power first) is [1, -a1, -a2, …, -aP]
    coefs = np.concatenate(([1.0], -a))

    # find its roots (these are the poles of the corresponding AR filter)
    roots = np.roots(coefs)

    # reflect any root outside the unit circle back inside
    roots[np.abs(roots) > 1] = 1.0 / np.conj(roots[np.abs(roots) > 1])

    # rebuild the stabilized polynomial (monic) from these new roots
    new_coefs = np.poly(roots)  # length P+1, highest→constant

    # extract the new AR coeffs: new_coefs = [1, c1, c2, …, cP]
    # but we want Q(z)=z^P - \tilde a1 z^(P-1) - …, so
    #  c1 = -\tilde a1, …, cP = -\tilde aP
    a_stable = -new_coefs[1:]

    return a_stable