# %%
import equinox as eqx
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel


class PeriodicMatern(Kernel):
    """Stationary periodic Matérn kernel (1D)

    k(Δ) = a0 + 2 * sum_{m>=1} a_m cos(2π m Δ / period)

    with
        a_m ∝ ( (2π m / period)^2 + λ^2 )^{-ν - 1/2}
        λ = sqrt(2ν) / scale
    """

    nu: float = eqx.field(static=True)
    scale: float = 1.0
    period: float = 1.0
    M: int = eqx.field(static=True, default=128)  # number of harmonics

    def _spectrum(self):
        """Return a_m for m = 0..M"""
        m = jnp.arange(0, self.M + 1)
        omega = 2.0 * jnp.pi * m / self.period
        lam = jnp.sqrt(2.0 * self.nu) / self.scale

        a = (omega * omega + lam * lam) ** (-(self.nu + 0.5))

        # normalize so that k(0) = 1
        a0 = a[0] + 2.0 * jnp.sum(a[1:])
        return a / a0

    def evaluate(self, t1: JAXArray, t2: JAXArray) -> JAXArray:
        if jnp.ndim(t1) or jnp.ndim(t2):
            raise ValueError("Expected scalar inputs")

        delta = t1 - t2
        m = jnp.arange(0, self.M + 1)
        a = self._spectrum()

        omega = 2.0 * jnp.pi * m / self.period
        cos_terms = jnp.cos(omega * delta)

        return a[0] + 2.0 * jnp.sum(a[1:] * cos_terms[1:])

    def spectrum_via_fft(self, N=8192):
        """Numerical spectrum from kernel samples, same convention as PACK"""
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

    k = PeriodicMatern(nu=1.5)
    t = jnp.linspace(-1, 2, 1000)

    gp = GaussianProcess(k, t)

    du = gp.sample(vk())
    plt.plot(t, du)
    plt.title(f"Periodic Matern nu={k.nu:.1f} sample")
    plt.show()

    S = k.spectrum_via_fft()
    plt.plot(S)
    plt.yscale("log")
    plt.title("Spectrum via FFT")
    plt.xlim(0, k.M + 20)  # zero beyond M unlike PACK
    plt.show()
