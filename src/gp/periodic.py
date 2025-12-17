# %%
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tinygp.helpers import JAXArray

from gp.mercer import Mercer

tfmath = tfp.math


def periodic_se_series_coeffs(ell: JAXArray, J: int):
    """
    Compute q_j for j = 0..J for the periodic SE kernel

        k(t, t') = sigma^2 * exp(-2 sin^2(pi (t - t') / T) / ell^2).

    We use the cosine expansion

        k(t, t') = sigma^2 * sum_{j>=0} qtilde_j^2 cos(j * omega0 * (t - t'))

    with omega0 = 2 pi / T, and truncate at j = 0..J.

    Using bessel_ive:

        bessel_ive(j, z) = I_j(z) * exp(-z), for z > 0

    the coefficients with sigma^2 = 1 are

        qtilde_0^2 = bessel_ive(0, z)
        qtilde_j^2 = 2 * bessel_ive(j, z), j >= 1

    where z = 1 / ell^2.

    Returns
    -------
    q : (J+1,)
        Square roots of qtilde_j^2 (no sigma yet).
    q2 : (J+1,)
        qtilde_j^2, same as above but not square-rooted.
    """
    z = 1.0 / (ell * ell)
    js = jnp.arange(J + 1, dtype=z.dtype)

    ive = tfmath.bessel_ive(js, z)  # I_j(z) * exp(-z), shape (J+1,)

    q2 = 2.0 * ive
    q2 = q2.at[0].set(ive[0])  # j = 0

    q = jnp.sqrt(q2)
    return q, q2


class PeriodicSE(Mercer):
    """
    Low-rank Mercer representation of the 1D periodic SE kernel

        k(t, t') = exp(-2 sin^2(pi (t - t') / T) / ell^2),

    with period T and lengthscale ell. We use the truncated Fourier series

        k(t, t') ≈ phi(t) @ W @ phi(t')^T

    where phi(t) are cosine/sine basis functions up to harmonic J, and
    W = L L^T is diagonal with entries built from the Bessel coefficients.

    Parameters
    ----------
    ell : float
        Lengthscale of the periodic SE kernel.
    period : float
        Period T of the kernel (in the same units as t).
    J : int
        Highest harmonic index in the truncation. Rank is 2J+1.
    """

    ell: JAXArray
    period: JAXArray
    J: int = eqx.field(static=True)

    def compute_phi(self, t: JAXArray) -> JAXArray:
        omega0 = 2.0 * jnp.pi / self.period
        js = jnp.arange(self.J + 1, dtype=t.dtype)

        theta = omega0 * t * js
        cos_terms = jnp.cos(theta)  # (J+1,)
        sin_terms = jnp.sin(theta)[1:]  # (J,)
        return jnp.concatenate([cos_terms, sin_terms], axis=0)  # (2J+1,)

    def compute_weights_root(self) -> JAXArray:
        """
        L such that W = L @ L.T gives the weights of the Mercer expansion.

        We build a diagonal L with entries

            diag_L = [sqrt(q2_0), ..., sqrt(q2_J),
                      sqrt(q2_1), ..., sqrt(q2_J)]

        so that cos_j and sin_j share the same weight.
        """
        q, q2 = periodic_se_series_coeffs(self.ell, self.J)  # (J+1,)

        # cos(0..J): q_0..q_J
        # sin(1..J): q_1..q_J
        diag_L = jnp.concatenate([q, q[1:]], axis=0)  # (2J+1,)

        return jnp.diag(diag_L)

if __name__ == "__main__":
    # PeriodicSE
    import numpy as np

    dt = 0.5
    M = 512
    t = np.arange(M) * dt
    tau = t[:, None] - t[None, :]
    T = 10.5
    ell = 1.3
    K = np.exp(-2 * (np.sin(np.pi * tau / T)) ** 2 / (ell**2))

    J = 30
    k = PeriodicSE(ell=jnp.array(ell), period=T, J=J)
    K_tinygp = k(t, t)

    print("Max abs diff PeriodicSE:", np.max(np.abs(K - K_tinygp)))