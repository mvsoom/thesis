# %%
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tinygp.helpers import JAXArray

from gp.mercer import Mercer
from pack.bitransform import build_k_tilde_matrix, compute_phi_tilde

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

# %%
class SPACK(Mercer):
    d: int = eqx.field(static=True)

    period: JAXArray
    J: int = eqx.field(static=True)

    t1: JAXArray
    t2: JAXArray

    M = 128  # Sweet spot; don't change

    def compute_omegas(self) -> JAXArray:
        omega0 = 2.0 * jnp.pi / self.period
        js = jnp.arange(1, self.J + 1)  # No DC component
        omegas = omega0 * js  # (J,)
        return omegas  # (J,)

    def compute_phi(self, t: JAXArray) -> JAXArray:
        omegas = self.compute_omegas()  # (J,)

        phase = omegas * t  # (J,)
        scale = 2.0 / self.period  # from the inverse Fourier series transform

        cos_terms = scale * jnp.cos(phase)
        sin_terms = -scale * jnp.sin(phase)

        return jnp.concatenate([cos_terms, sin_terms], axis=0)  # (2J,)

    def compute_phi_integrated(self, t: JAXArray, t0: JAXArray) -> JAXArray:
        """Compute ∫_{t0}^t phi'(τ) dτ"""
        omegas = self.compute_omegas()  # (J,)

        phase = omegas * t
        phase0 = omegas * t0
        scale = 2.0 / (self.period * omegas)

        cos_terms = jnp.sin(phase) - jnp.sin(phase0)
        sin_terms = jnp.cos(phase) - jnp.cos(phase0)

        return jnp.concatenate(
            [scale * cos_terms, scale * sin_terms],
            axis=0,
        )  # (2J,)

    def compute_weights_root(self) -> JAXArray:
        omegas = self.compute_omegas()  # (J,)

        Phi_complex = jax.vmap(
            lambda w: compute_phi_tilde(w, self.d, self.M, self.t1, self.t2)
        )(omegas)

        Phi_R = jnp.real(Phi_complex)
        Phi_I = jnp.imag(Phi_complex)

        # A = [Phi_R  -Phi_I; Phi_I  Phi_R] / sqrt(2)
        top = jnp.concatenate([Phi_R, -Phi_I], axis=1)
        bot = jnp.concatenate([Phi_I, Phi_R], axis=1)
        A = jnp.concatenate([top, bot], axis=0)

        return A / jnp.sqrt(2.0)  # shape (2J, 2R)


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    # kernel params
    period = 5.0
    t1, t2 = -3.0, 4.0
    J = 20
    d = 0

    sp = SPACK(
        d=d, period=jnp.array(period), J=J, t1=jnp.array(t1), t2=jnp.array(t2)
    )

    # ---------- old slow method ----------
    omegas = sp.compute_omegas()
    K_complex = build_k_tilde_matrix(omegas, d, sp.M, t1, t2)  # (J,J)

    ReK = jnp.real(K_complex)
    ImK = jnp.imag(K_complex)

    top = jnp.concatenate([ReK, -ImK], axis=1)
    bot = jnp.concatenate([ImK, ReK], axis=1)
    Sigma_old = 0.5 * jnp.concatenate([top, bot], axis=0)  # (2J,2J)

    L_old = jnp.linalg.cholesky(Sigma_old + 1e-8 * jnp.eye(2 * J))  # (2J,2J)

    # ---------- new rectangular method ----------
    L_new = sp.compute_weights_root()  # (2J, 2R)

    Sigma_new = L_new @ L_new.T

    # ---------- errors ----------
    err = jnp.max(jnp.abs(Sigma_old - Sigma_new))
    fro = jnp.linalg.norm(Sigma_old - Sigma_new) / jnp.linalg.norm(Sigma_old)

    print("max abs error:", float(err))
    print("relative Frobenius:", float(fro))

    # Quick sanity: check PSD-ness
    eigs_old = jnp.linalg.eigvalsh(Sigma_old)
    eigs_new = jnp.linalg.eigvalsh(Sigma_new)

    print("min eig (old):", float(jnp.min(eigs_old)))
    print("min eig (new):", float(jnp.min(eigs_new)))
    print(
        "rank(old)=",
        int(jnp.sum(eigs_old > 1e-8)),
        "rank(new)=",
        int(jnp.sum(eigs_new > 1e-8)),
    )


# %%
class AffineTransformedSPACK(Mercer):
    alpha: JAXArray
    beta: JAXArray
    kernel: SPACK  # defines the external harmonic grid (period, J, etc.)

    def __init__(self, *, alpha, beta, d, period, J, t1, t2):
        self.alpha = alpha
        self.beta = beta
        self.kernel = SPACK(d=d, period=period, J=J, t1=t1, t2=t2)

    def compute_omegas(self):
        return self.kernel.compute_omegas()

    def compute_phi(self, t):
        # base features on the original grid
        phi = self.kernel.compute_phi(t)  # (2J,)

        # apply beta phase as rotation per harmonic
        omegas = self.compute_omegas()  # (J,)
        theta = omegas * (
            self.beta / self.alpha
        )  # matches exp(-i beta * omega/alpha)

        c = jnp.cos(theta)
        s = jnp.sin(theta)

        J = self.kernel.J
        cos = phi[:J]
        sin = phi[J:]

        cos2 = c * cos - s * sin
        sin2 = s * cos + c * sin

        # apply global alpha^{-1} (since Sigma gets alpha^{-2})
        pref = 1.0 / self.alpha
        return pref * jnp.concatenate([cos2, sin2], axis=0)

    def compute_phi_integrated(self, t, t0):
        # same story for integrated features: rotate, then scale
        phi = self.kernel.compute_phi_integrated(t, t0)

        omegas = self.compute_omegas()
        theta = omegas * (self.beta / self.alpha)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        J = self.kernel.J
        cos = phi[:J]
        sin = phi[J:]

        cos2 = c * cos - s * sin
        sin2 = s * cos + c * sin

        # integral introduces another 1/alpha if your definition is in u-coordinates;
        # in this "external basis" gauge, the only required scale is the same alpha^{-1}
        # as for compute_phi, unless your integrated feature is defined as ∫ phi'(tau) dt.
        pref = 1.0 / self.alpha
        return pref * jnp.concatenate([cos2, sin2], axis=0)

    def compute_weights_root(self):
        spack = SPACK(
            d=self.kernel.d,
            period=self.alpha * self.kernel.period,
            J=self.kernel.J,
            t1=self.alpha * self.kernel.t1 - self.beta,
            t2=self.alpha * self.kernel.t2 - self.beta,
        )
        return spack.compute_weights_root()


def _compute_weights_root2(self) -> JAXArray:
    k = self.kernel  # canonical SPACK

    # original harmonic grid (period = T)
    omegas = k.compute_omegas()  # (J,)

    # evaluate canonical bitransform at scaled frequencies + shifted window
    Phi_complex = jax.vmap(
        lambda w: compute_phi_tilde(
            w / self.alpha,
            k.d,
            k.M,
            self.alpha * k.t1 - self.beta,
            self.alpha * k.t2 - self.beta,
        )
    )(omegas)  # (J, R) complex

    # apply affine phase: exp(-i beta * omega / alpha)
    phase = jnp.exp(-1j * self.beta * omegas / self.alpha)  # (J,)
    Phi_complex = phase[:, None] * Phi_complex

    # real/imag split
    Phi_R = jnp.real(Phi_complex)
    Phi_I = jnp.imag(Phi_complex)

    # A = [Phi_R  -Phi_I; Phi_I  Phi_R]
    top = jnp.concatenate([Phi_R, -Phi_I], axis=1)
    bot = jnp.concatenate([Phi_I, Phi_R], axis=1)
    A = jnp.concatenate([top, bot], axis=0)

    # alpha^{-1} at the root level (since Sigma gets alpha^{-2})
    return A / (jnp.sqrt(2.0) * self.alpha)


if __name__ == "__main__":
    params = dict(
        alpha=jnp.array(3.0),
        beta=jnp.array(-1.0),
        d=3,
        period=jnp.array(1.0),
        J=20,
        t1=jnp.array(-5.0),
        t2=jnp.array(4.0),
    )

    # build kernel
    sp = AffineTransformedSPACK(**params)

    # grid for evaluation
    t = jnp.linspace(-5.0, 5.0, 200)

    # ---- representation A (basis-soaked) ----
    Phi1 = jax.vmap(sp.compute_phi)(t)  # (N, 2J)
    L1 = sp.compute_weights_root()  # (2J, R)
    K1 = Phi1 @ (L1 @ L1.T) @ Phi1.T  # (N, N)

    # ---- representation B (weight-soaked) ----
    Phi2 = jax.vmap(sp.kernel.compute_phi)(t)  # canonical basis
    L2 = _compute_weights_root2(sp)
    K2 = Phi2 @ (L2 @ L2.T) @ Phi2.T

    # ---- compare kernels ----
    err = jnp.max(jnp.abs(K1 - K2))
    print(
        "max kernel diff:", float(err)
    )  # OK, still suffers from ill conditioned recursive relation for d>1, but good enough

# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tinygp import GaussianProcess

    from utils.jax import vk

    params = dict(
        alpha=jnp.array(1 / 10.0),
        beta=jnp.array(10.0),
        d=1,
        period=jnp.array(5.0),
        J=20,
        t1=jnp.array(0.0),
        t2=jnp.array(2.0),
    )

    sp = AffineTransformedSPACK(**params)
    t = jnp.linspace(-5.0, 5.0, 200)
    gp = GaussianProcess(sp, t)

    f = gp.sample(vk())

    plt.plot(t, f)


# %%
class PACK(AffineTransformedSPACK):
    sigma_b: float = eqx.field(default_factory=lambda: 1.0)
    sigma_c: float = eqx.field(default_factory=lambda: 1.0)
    center: float = eqx.field(default_factory=lambda: 0.0)

    def __init__(
        self,
        *,
        sigma_b: JAXArray = 1.0,
        sigma_c: JAXArray = 1.0,
        center: JAXArray = 0.0,
        d: int,
        period: JAXArray,
        J: int,
        t1: JAXArray,
        t2: JAXArray,
    ):
        if jnp.any(sigma_b == 0):
            raise ValueError("sigma_b must be nonzero")

        self.sigma_b = sigma_b
        self.sigma_c = sigma_c
        self.center = center

        alpha = sigma_c / sigma_b
        beta = alpha * center

        print("PACK init:", alpha, beta)

        super().__init__(
            alpha=alpha,
            beta=beta,
            d=d,
            period=period,
            J=J,
            t1=t1,
            t2=t2,
        )

    def compute_weights_root(self):
        return super().compute_weights_root() * (self.sigma_b**self.kernel.d)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tinygp import GaussianProcess

    from gp.blr import blr_from_mercer
    from utils.jax import vk

    jax.config.update("jax_debug_nans", True)

    params = dict(
        sigma_b=0.010,
        sigma_c=1,
        center=+10,  # FIXME
        d=0,
        period=jnp.array(100.0),
        J=50,
        t1=jnp.array(-50.0),
        t2=jnp.array(50.0),
    )

    sp = PACK(**params)
    t = jnp.linspace(-20, 20, 500)
    gp = blr_from_mercer(sp, t)

    f = gp.sample(vk())

    plt.plot(t, f)
    # plt.plot(t, np.cumsum(f) * (t[1] - t[0]))
