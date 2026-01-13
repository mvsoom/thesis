# %%
import equinox as eqx
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel

from gfm.ack import ACK


class PACK(Kernel):
    """Stationary PACK kernel with optional per-harmonic sigma_c"""

    d: int = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    J: int = eqx.field(static=True, default=1)  # number of harmonics is 2J

    period: float = 1.0

    sigma_b: float = 1.0
    sigma_c: JAXArray | float = 1.0  # scalar or shape (J,)

    def _sigma_cj(self):
        """Return per-harmonic sigma_c of shape (J,)"""
        sigma_c = self.sigma_c
        if jnp.ndim(sigma_c) == 0:
            return jnp.full((self.J,), sigma_c)
        if sigma_c.shape == (self.J,):
            return sigma_c
        raise ValueError(
            f"sigma_c must be scalar or shape ({self.J},), got {sigma_c.shape}"
        )

    def evaluate(self, t1: JAXArray, t2: JAXArray) -> JAXArray:
        if jnp.ndim(t1) or jnp.ndim(t2):
            raise ValueError("Expected scalar inputs")

        w = 2.0 * jnp.pi / self.period
        j = jnp.arange(1, self.J + 1)

        sigma_b = self.sigma_b
        sigma_cj = self._sigma_cj()

        X1 = jnp.concatenate(
            [
                jnp.array([sigma_b]),
                sigma_cj * jnp.cos(w * j * t1),
                sigma_cj * jnp.sin(w * j * t1),
            ]
        )
        X2 = jnp.concatenate(
            [
                jnp.array([sigma_b]),
                sigma_cj * jnp.cos(w * j * t2),
                sigma_cj * jnp.sin(w * j * t2),
            ]
        )

        ack = ACK(d=self.d, normalized=self.normalized)
        pre = 1.0 if self.normalized else 0.5
        return pre * ack.evaluate(X1, X2)

    def spectrum_via_fft(self, N=8192):
        delta = jnp.linspace(0.0, self.period, N, endpoint=False)
        k = self(delta, jnp.array([0.0])).squeeze()
        K = jnp.fft.rfft(k) / N
        a0 = K[0].real
        am = 2.0 * K[1:].real
        return jnp.concatenate([jnp.array([a0]), am])


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tinygp.gp import GaussianProcess

    from utils.jax import vk

    k = PACK(d=1, normalized=True, J=2, sigma_c=jnp.array([0.5, 1.5]))
    t = jnp.linspace(-1, 2, 1000)

    gp = GaussianProcess(k, t)

    du = gp.sample(vk())
    plt.plot(t, du)
    plt.title("PACK sample")
    plt.show()

    S = k.spectrum_via_fft()
    plt.plot(S)
    plt.yscale("log")
    plt.title("PACK spectrum via FFT")
    plt.show()
