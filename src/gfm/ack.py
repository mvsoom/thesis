# %%

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel

from gfm.filon import filon_tab_iexp


def compute_Jd(d, c, s):
    theta = jnp.arctan2(s, c)  # more stable than arccos()
    ctheta = jnp.pi - theta

    match d:
        case 0:
            Jd = ctheta
        case 1:
            Jd = s + ctheta * c
        case 2:
            Jd = 3.0 * s * c + ctheta * (1.0 + 2.0 * c * c)
        case 3:
            Jd = (
                15.0 * s
                - 11.0 * s * s * s
                + ctheta * (9.0 * c + 6.0 * c * c * c)
            )
        case _:
            raise NotImplementedError(f"Degree {d} not implemented")

    return Jd


def compute_Jd_theta(d, theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return compute_Jd(d, c, s)


def compute_Jd_theta_even(d, theta):
    return compute_Jd_theta(d, jnp.abs(theta))


class ACK(Kernel):
    """Arc cosine kernel of degree `d` (as in Cho & Saul, 2009)

    The `normalized` version is k(x,x')/sqrt(k(x,x) k(x',x')).
    """

    d: int = eqx.field(static=True)

    normalized: bool = eqx.field(default_factory=lambda: False, static=True)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        X1_norm = jnp.linalg.norm(X1)
        X2_norm = jnp.linalg.norm(X2)

        R1 = X1 / X1_norm
        R2 = X2 / X2_norm

        c = jnp.clip(jnp.dot(R1, R2), -1.0, 1.0)
        s = jnp.sqrt(jnp.maximum(0.0, 1.0 - c * c))

        Jd = compute_Jd(self.d, c, s)

        if self.normalized:
            Jd0 = compute_Jd(self.d, 1.0, 0.0)
            K12 = Jd / Jd0
        else:
            K12 = (1.0 / jnp.pi) * (X1_norm**self.d) * (X2_norm**self.d) * Jd
        return K12


def exp_im_psi(t, m):
    """Compute e^{i m psi(t)} where psi(t) = arctan(t)"""
    z = (1.0 + 1j * t) / jnp.sqrt(1.0 + t * t)
    return z**m


class TACK(Kernel):
    """Temporal arc cosine kernel of degree `d`

    Note that this has the prefactor (1/2) to compensate for ACK convention (we use use the literal GP expectation instead).

    The `normalized` version is k(x,x')/sqrt(k(x,x) k(x',x')).
    Optionally, the time indices are centered around `center` and scaled by `LSigma` which is Sigma^{1/2} (any root will do).
    """

    d: int = eqx.field(static=True)

    normalized: bool = eqx.field(default_factory=lambda: False, static=True)

    center: float = eqx.field(default_factory=lambda: 0.0)
    LSigma: JAXArray = eqx.field(default_factory=lambda: jnp.eye(2))

    def evaluate(self, t1: JAXArray, t2: JAXArray) -> JAXArray:
        if jnp.ndim(t1) or jnp.ndim(t2):
            raise ValueError("Expected scalar inputs")

        X1 = jnp.array([1.0, t1 - self.center])
        X2 = jnp.array([1.0, t2 - self.center])

        ack = ACK(d=self.d, normalized=self.normalized)
        pre = 1.0 if self.normalized else 0.5
        K12 = pre * ack.evaluate(self.LSigma @ X1, self.LSigma @ X2)
        return K12


FILON_N = 129  # odd > 1
FILON_PANELS = 32  # total panels: increase THIS if need more accuracy


def quad_filon_omega(g, omega, u1, u2):
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


class DiagonalTACK(Kernel):
    """Temporal arc cosine kernel of degree `d` with LSigma = diag(sigma_b, sigma_c)

    Note: when fitting this kernel with an amplitude:

        The user should probably constrain sigma_c = 1.0 if we assume that the kernel used as

            k = sigma_a * DiagonalTACK(..., sigma_c=sigma_c);

        in this case sigma_a and sigma_c are non-distinguishable.
    """

    d: int = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)

    center: float = 0.0
    sigma_b: float = 1.0
    sigma_c: float = 1.0

    def evaluate(self, t1: JAXArray, t2: JAXArray) -> JAXArray:
        if jnp.ndim(t1) or jnp.ndim(t2):
            raise ValueError("Expected scalar inputs")

        LSigma = jnp.diag([self.sigma_b, self.sigma_c])

        tack = TACK(
            d=self.d,
            normalized=self.normalized,
            LSigma=LSigma,
            center=self.center,
        )
        return tack.evaluate(t1, t2)

    def compute_H_factor(
        self, m: int, f: JAXArray, t1: JAXArray, t2: JAXArray
    ) -> JAXArray:
        """Compute H_m(f)"""

        beta = self.sigma_b / self.sigma_c
        w = 2.0 * jnp.pi * f

        # Change of variables: u = arctan((t - center) / beta)
        u1 = jnp.arctan((t1 - self.center) / beta)
        u2 = jnp.arctan((t2 - self.center) / beta)

        if self.normalized:
            Jd0 = compute_Jd(self.d, 1.0, 0.0)
            scale = 1.0 / jnp.sqrt(Jd0)

            def g(u):
                cu = jnp.cos(u)
                sec2 = 1.0 / (cu * cu)
                t = self.center + beta * jnp.tan(u)
                jac = beta * sec2
                return scale * jac * jnp.exp(-1j * w * t)

        else:
            scale = 1.0 / jnp.sqrt(2.0 * jnp.pi)

            def g(u):
                cu = jnp.cos(u)
                sec = 1.0 / cu
                sec2 = sec * sec
                t = self.center + beta * jnp.tan(u)
                jac = beta * sec2
                poly = (self.sigma_c**self.d) * (beta**self.d) * (sec**self.d)
                return scale * poly * jac * jnp.exp(-1j * w * t)

        omega = jnp.asarray(m, dtype=jnp.float64)
        return quad_filon_omega(g, omega=omega, u1=u1, u2=u2)


class STACK(DiagonalTACK):
    """Standard temporal arc cosine kernel of degree `d`"""

    pass


if __name__ == "__main__":
    LSigma = jnp.array([[3.1, 0.3], [0.3, 0.27789]])
    t = jnp.linspace(0.0, 10.0, 100)

    k = TACK(d=0, normalized=False, LSigma=LSigma, center=t.mean())
    kn = TACK(d=0, normalized=True, LSigma=LSigma, center=t.mean())

    K = k(t, t)
    Kn = kn(t, t)

    denom = jnp.sqrt(k(t)[:, None] * k(t)[None, :])

    assert jnp.allclose((K / denom), Kn)
