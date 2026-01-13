# %%

import equinox as eqx
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel

from gfm.ack import ACK


class PACK(Kernel):
    d: int = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    J: int = eqx.field(static=True, default=1)  # number of harmonics is 2J

    period: float = 1.0

    sigma_b: float = 1.0
    sigma_c: float = 1.0

    def evaluate(self, t1: JAXArray, t2: JAXArray) -> JAXArray:
        if jnp.ndim(t1) or jnp.ndim(t2):
            raise ValueError("Expected scalar inputs")

        w = 2.0 * jnp.pi / self.period
        j = jnp.arange(1, self.J + 1)

        sigma_b = self.sigma_b
        sigma_c = self.sigma_c

        X1 = jnp.concatenate(
            [
                jnp.array([sigma_b]),
                sigma_c * jnp.cos(w * j * t1),
                sigma_c * jnp.sin(w * j * t1),
            ]
        )
        X2 = jnp.concatenate(
            [
                jnp.array([sigma_b]),
                sigma_c * jnp.cos(w * j * t2),
                sigma_c * jnp.sin(w * j * t2),
            ]
        )

        ack = ACK(d=self.d, normalized=self.normalized)
        pre = 1.0 if self.normalized else 0.5
        K12 = pre * ack.evaluate(X1, X2)
        return K12


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tinygp.gp import GaussianProcess

    from utils.jax import vk

    k = PACK(d=1, normalized=True, J=2)
    t = jnp.linspace(-1, 2, 1000)

    gp = GaussianProcess(k, t)

    du = gp.sample(vk())
    plt.plot(t, du)
