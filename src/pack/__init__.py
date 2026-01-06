# %%
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from tensorflow_probability.substrates.jax.math import (
    cholesky_update as jax_cholesky_update,
)
from tinygp.helpers import JAXArray

from gfm.ack import DiagonalTACK, compute_Jd
from gp.blr import blr_from_mercer
from gp.mercer import Mercer
from pack import zonal
from pack.filon import filon_tab_iexp

FILON_N = 129  # odd > 1
FILON_PANELS = 64  # total panels: increase THIS if need more accuracy

M_MAX = 512

ORACLE_PARTS = 64
ORACLE_EPS = 1e-11
ORACLE_LIMIT = 300


def quad_filon_H_factor(g, omega, u1, u2):
    a = jnp.minimum(u1, u2)
    b = jnp.maximum(u1, u2)
    sgn = jnp.where(u2 >= u1, 1.0, -1.0)

    edges = jnp.linspace(a, b, FILON_PANELS + 1)

    def body(i, acc):
        p = edges[i]
        q = edges[i + 1]

        j = jnp.arange(FILON_N)
        h = (q - p) / (FILON_N - 1)
        u = p + h * j

        gtab = jax.vmap(g, in_axes=(0,))(u)
        return acc + filon_tab_iexp(gtab, p, q, omega)

    acc0 = jnp.array(0.0 + 0.0j)
    integral = jax.lax.fori_loop(0, FILON_PANELS, body, acc0)
    return sgn * integral


def quad_oracle_H_factor(integrand, m, f, t1, t2):
    """Numerically integrate with weighted scipy quad as oracle reference (slow but accurate, not JAX compatible)"""
    w = 2.0 * np.pi * f
    edges = np.linspace(t1, t2, ORACLE_PARTS + 1)

    def cos_m(t):
        return np.real(integrand(t, m))

    def sin_m(t):
        return np.imag(integrand(t, m))

    tot = 0.0 + 0.0j

    params = dict(epsabs=ORACLE_EPS, epsrel=ORACLE_EPS, limit=ORACLE_LIMIT)

    if w == 0.0:
        for i in range(ORACLE_PARTS):
            a, b = edges[i], edges[i + 1]
            re = quad(cos_m, a, b, **params)[0]
            im = quad(sin_m, a, b, **params)[0]
            tot += re + 1j * im
        return tot

    for i in range(ORACLE_PARTS):
        a, b = edges[i], edges[i + 1]
        Ac = quad(cos_m, a, b, weight="cos", wvar=w, **params)[0]
        Bs = quad(sin_m, a, b, weight="sin", wvar=w, **params)[0]
        Bc = quad(sin_m, a, b, weight="cos", wvar=w, **params)[0]
        As = quad(cos_m, a, b, weight="sin", wvar=w, **params)[0]
        tot += (Ac + Bs) + 1j * (Bc - As)

    return tot


def quad_scipy_H_factor(k_d, f, fp, t1, t2):
    """
    Oracle computation of

        ∫_{t1}^{t2} ∫_{t1}^{t2}
            k_d(t, tp) exp(-i 2π f t) exp(+i 2π fp tp)
        dt dtp

    using weighted quad in the oscillatory (t) direction.
    """

    w = 2.0 * np.pi * f
    edges = np.linspace(t1, t2, ORACLE_PARTS + 1)

    params = dict(epsabs=ORACLE_EPS, epsrel=ORACLE_EPS, limit=ORACLE_LIMIT)

    def inner_integral(tp):
        """Compute ∫ k(t,tp) e^{-i2π f t} dt"""

        def a(t):
            return np.real(k_d(t, tp))

        def b(t):
            return np.imag(k_d(t, tp))

        val = 0.0 + 0.0j

        if w == 0.0:
            for i in range(ORACLE_PARTS):
                a0, b0 = edges[i], edges[i + 1]
                re = quad(a, a0, b0, **params)[0]
                im = quad(b, a0, b0, **params)[0]
                val += re + 1j * im
            return val

        for i in range(ORACLE_PARTS):
            a0, b0 = edges[i], edges[i + 1]

            Ac = quad(a, a0, b0, weight="cos", wvar=w, **params)[0]
            Bs = quad(b, a0, b0, weight="sin", wvar=w, **params)[0]
            Bc = quad(b, a0, b0, weight="cos", wvar=w, **params)[0]
            As = quad(a, a0, b0, weight="sin", wvar=w, **params)[0]

            val += (Ac + Bs) + 1j * (Bc - As)

        return val

    def outer_real(tp):
        return np.real(inner_integral(tp) * np.exp(1j * 2.0 * np.pi * fp * tp))

    def outer_imag(tp):
        return np.imag(inner_integral(tp) * np.exp(1j * 2.0 * np.pi * fp * tp))

    re = quad(outer_real, t1, t2, **params)[0]
    im = quad(outer_imag, t1, t2, **params)[0]

    return re + 1j * im


class PACK(Mercer):
    k: DiagonalTACK

    period: JAXArray
    t1: JAXArray
    t2: JAXArray

    J: int = eqx.field(static=True)  # number of harmonics is 2J + 1
    M: int = M_MAX

    def compute_H_factor(self, m: int, f: JAXArray) -> JAXArray:
        """Compute H_m(f) with integrand provided by the (un)normalized DiagonalTACK kernel with Filon quadrature

        Note: for sigma_c := 1 this has typically errors smaller than 1e-4 (tests are in src/pack/fourier.py).
        Only when sigma_b becomes (very) small or (very) large can this sometimes have bad accuracy, but in this regime the kernels are likely degenerate anyway.
        """

        beta = self.k.sigma_b / self.k.sigma_c
        w = 2.0 * jnp.pi * f

        # Change of variables: u = arctan((t - center) / beta)
        u1 = jnp.arctan((self.t1 - self.k.center) / beta)
        u2 = jnp.arctan((self.t2 - self.k.center) / beta)

        if self.k.normalized:
            Jd0 = compute_Jd(self.k.d, 1.0, 0.0)
            scale = 1.0 / jnp.sqrt(Jd0)

            def g(u):
                cu = jnp.cos(u)
                sec2 = 1.0 / (cu * cu)
                t = self.k.center + beta * jnp.tan(u)
                jac = beta * sec2
                return scale * jac * jnp.exp(-1j * w * t)

        else:
            scale = 1.0 / jnp.sqrt(2.0 * jnp.pi)

            def g(u):
                cu = jnp.cos(u)
                sec = 1.0 / cu
                sec2 = sec * sec
                t = self.k.center + beta * jnp.tan(u)
                jac = beta * sec2
                d = self.k.d
                poly = (self.k.sigma_c**d) * (beta**d) * (sec**d)
                return scale * poly * jac * jnp.exp(-1j * w * t)

        omega = jnp.asarray(m, dtype=jnp.float64)

        l = quad_filon_H_factor(g, omega, u1, 0.0)
        r = quad_filon_H_factor(g, omega, 0.0, u2)

        return l + r

    def compute_H_factor_oracle(self, m: int, f: JAXArray) -> JAXArray:
        """Compute H_m(f) with integrand provided by the (un)normalized DiagonalTACK kernel using scipy quad as oracle"""
        beta = self.k.sigma_b / self.k.sigma_c
        d = self.k.d

        if self.k.normalized:
            Jd0 = compute_Jd(d, 1.0, 0.0)
            scale_norm = 1.0 / np.sqrt(Jd0)
        else:
            scale_unnorm = 1.0 / np.sqrt(2 * np.pi)

        def integrand(t, m):
            tau = t - self.k.center
            z = (1.0 + 1j * (tau / beta)) / np.sqrt(1.0 + (tau / beta) ** 2)
            psi_t = z**m

            if self.k.normalized:
                return scale_norm * psi_t
            else:
                poly = self.k.sigma_c**d * (beta * beta + tau * tau) ** (d / 2)
                return scale_unnorm * poly * psi_t

        return quad_oracle_H_factor(integrand, m, f, self.t1, self.t2)

    def compute_phi_tilde(self, f, eps=1e-8):
        """Compute the factors of the Mercer series of k_tilde(f, f') = sum_m c_m phi_tilde_m(f) phi_tilde_m(f')

        Here `eps` determines the cutoff for very small c_m coefficients.

        Note: the weights are baked in the phi factors.

            k_tilde(f, f') = <phi_tilde(f), phi_tilde(f')>

        where this is the conjugate dot product.
        """
        ms, c = zonal.nonzero_complex_coeffs(self.k.d, self.M, eps=eps)

        compute_Hm = jax.vmap(lambda m: self.compute_H_factor(m, f))

        phi_tilde = jnp.sqrt(c) * compute_Hm(ms)
        return phi_tilde  # (R,)

    def compute_k_tilde(self, f, f_prime):
        """Compute the Hermitian kernel k_tilde(f, f')"""
        phi_tilde = self.compute_phi_tilde
        k_tilde = jnp.vdot(
            phi_tilde(f), phi_tilde(f_prime)
        )  # conjugate dot product
        return k_tilde  # ()

    def compute_k_tilde_oracle(self, f, f_prime):
        kd = jax.jit(self.k.evaluate)
        return quad_scipy_H_factor(kd, f, f_prime, self.t1, self.t2)  # ()

    def compute_harmonics(self) -> JAXArray:
        f0 = 1 / self.period
        js = jnp.arange(1, self.J + 1)  # No DC component
        return js * f0  # (J,)

    def compute_phi(self, t: JAXArray) -> JAXArray:
        omegas = 2.0 * jnp.pi * self.compute_harmonics()
        phase = omegas * t  # (J,)
        scale = 2.0 / self.period  # from the inverse Fourier series transform

        cos_terms = scale * jnp.cos(phase)
        sin_terms = -scale * jnp.sin(phase)

        return jnp.concatenate([cos_terms, sin_terms], axis=0)  # (2J,)

    def compute_phi_integrated(self, t: JAXArray, t0: JAXArray) -> JAXArray:
        """Compute ∫_{t0}^t phi'(τ) dτ"""
        omegas = 2.0 * jnp.pi * self.compute_harmonics()

        phase = omegas * t
        phase0 = omegas * t0
        scale = 2.0 / (self.period * omegas)

        cos_terms = jnp.sin(phase) - jnp.sin(phase0)
        sin_terms = jnp.cos(phase) - jnp.cos(phase0)

        return jnp.concatenate(
            [scale * cos_terms, scale * sin_terms],
            axis=0,
        )  # (2J,)

    def compute_phi_derivative(self, t: JAXArray) -> JAXArray:
        omegas = 2.0 * jnp.pi * self.compute_harmonics()  # (J,)
        scale = 2.0 / self.period

        phase = omegas * t

        cos_part = -scale * omegas * jnp.sin(phase)
        sin_part = -scale * omegas * jnp.cos(phase)

        return jnp.concatenate([cos_part, sin_part], axis=0)  # (2J,)

    def compute_weights_root(self):
        harmonics = self.compute_harmonics()  # (J,)
        Phi_c = jax.vmap(self.compute_phi_tilde)(harmonics)  # (J, R)

        Phi_R = jnp.real(Phi_c)
        Phi_I = jnp.imag(Phi_c)

        top = jnp.concatenate([Phi_R, -Phi_I], axis=1)
        bot = jnp.concatenate([Phi_I, Phi_R], axis=1)
        L = jnp.concatenate([top, bot], axis=0) / jnp.sqrt(2.0)  # (2J, 2R)
        return L


def _project_root_hard(L, Q):
    """
    Enforce Q z = 0 by projecting coefficient space.
    Q: (Kc, D)
    L: (D, R)
    """
    if Q.shape[0] == 0:
        return L

    U, _ = jnp.linalg.qr(Q.T, mode="reduced")  # (D, r)
    P = jnp.eye(U.shape[0], dtype=L.dtype) - U @ U.T
    return P @ L


def _suppress_interval_energy(L, Phi, lam):
    """
    Apply (W^{-1} + lam Phi^T Phi)^{-1} update in root form.
    Phi: (K, D)
    L:   (D, R)
    """
    Phi_tilde = jnp.sqrt(lam) * Phi  # (K, D)
    A = Phi_tilde @ L  # (K, R)

    C = jnp.eye(A.shape[0], dtype=L.dtype) + A @ A.T
    R = jnp.linalg.cholesky(C)

    tmp = jax.scipy.linalg.solve_triangular(R, A, lower=True)
    tmp = jax.scipy.linalg.solve_triangular(R.T, tmp, lower=False)

    return L - (L @ A.T) @ tmp


class ConstrainedPACK(Mercer):
    k: PACK

    # hard constraints
    closure_condition: bool = True  # ∫_{t1}^{t2} f = 0
    pin_phase: bool = True  # f(0) = 0
    pin_derivative_phase: bool = True  # f'(0) = 0

    # soft constraint
    suppress_interval: bool = True
    lam_interval: float = 1e4
    K_interval: int = 128

    def compute_phi(self, t: JAXArray) -> JAXArray:
        return self.k.compute_phi(t)

    def compute_weights_root(self) -> JAXArray:
        L = self.k.compute_weights_root()  # (D, R), D = 2J

        Qs = []

        if self.closure_condition:
            q = self.k.compute_phi_integrated(self.k.t2, self.k.t1)
            Qs.append(q)

        if self.pin_phase:
            Qs.append(self.k.compute_phi(0.0))
        if self.pin_derivative_phase:
            Qs.append(self.k.compute_phi_derivative(0.0))

        if Qs:
            Q = jnp.stack(Qs, axis=0)  # (Kc, D)
            L = _project_root_hard(L, Q)

        if self.suppress_interval:
            K = self.K_interval
            ts = jnp.linspace(
                self.k.t2,
                self.k.period,
                K,
                endpoint=False,
            )
            Phi = jax.vmap(self.k.compute_phi)(ts)  # (K, D)

            L = _suppress_interval_energy(
                L,
                Phi,
                self.lam_interval,
            )

        return L


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")

    import matplotlib.pyplot as plt
    import numpy as np

    from utils.jax import vk

    c = zonal.complex_coeffs(2, 10)

    k = DiagonalTACK(
        d=1, normalized=False, sigma_b=0.1, sigma_c=1.0, center=0.8
    )

    pack = PACK(k, period=1.0, t1=0.0, t2=0.9, J=1024)

    f = 4.3
    f_prime = 5.0

    x = pack.compute_H_factor(1, 0.3)
    y = pack.compute_H_factor_oracle(1, 0.3)

    print("H_factor diff:", jnp.abs(x - y))

    # %%
    x = pack.compute_k_tilde(f, f_prime)
    y = pack.compute_k_tilde_oracle(f, f_prime)

    print("k_tilde diff:", jnp.abs(x - y))
    # %%

    W = pack.compute_weights_root()

    plt.matshow(np.array(W))

    # %%
    k = DiagonalTACK(d=1, normalized=True, sigma_b=0.1, sigma_c=1.0, center=0.7)

    pack = PACK(k, period=1.0, t1=0.0, t2=0.8, J=512)

    t = jnp.linspace(0, 2, 1024)

    gp = blr_from_mercer(pack, t)

    # %%
    f = gp.sample(vk())
    u = np.cumsum(f) * (t[1] - t[0])

    plt.plot(t, f)
    plt.plot(t, u)

    # %%
    # You can constrain this into behaving really like a LF model
    cpack = ConstrainedPACK(pack)

    cgp = blr_from_mercer(cpack, t)

    # %%
    f = cgp.sample(vk())
    u = np.cumsum(f) * (t[1] - t[0])

    plt.plot(t, f)
    plt.plot(t, u)
