# %%
import equinox as eqx
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel


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


if __name__ == "__main__":
    LSigma = jnp.array([[3.1, 0.3], [0.3, 0.27789]])
    t = jnp.linspace(0.0, 10.0, 100)

    k = TACK(d=0, normalized=False, LSigma=LSigma, center=t.mean())
    kn = TACK(d=0, normalized=True, LSigma=LSigma, center=t.mean())

    K = k(t, t)
    Kn = kn(t, t)

    denom = jnp.sqrt(k(t)[:, None] * k(t)[None, :])

    assert jnp.allclose((K / denom), Kn)
