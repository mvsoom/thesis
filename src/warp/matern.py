# %%
from typing import Optional, Sequence

import gpjax as gpx
import jax.numpy as jnp
import jax.scipy as jsp
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import PositiveReal


def _glaguerre_gw(J, alpha):
    n = jnp.arange(J, dtype=jnp.float32)
    diag = 2.0 * n + 1.0 + alpha
    k = jnp.arange(1, J, dtype=jnp.float32)
    off = jnp.sqrt(k * (k + alpha))

    A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    z, V = jnp.linalg.eigh(A)

    mu0 = jnp.exp(jsp.special.gammaln(alpha + 1.0))
    w = mu0 * (V[0, :] ** 2)
    return z, w


class gpxMaternSEMixture(gpx.kernels.AbstractKernel):
    """
    Matérn kernel via squared exponential mixture (Tronarp+ 2018).

    Static:
        J   number of mixture components

    Learnable:
        nu
        scale
        lengthscale
    """

    J: int

    nu: PositiveReal
    scale: PositiveReal
    lengthscale: PositiveReal

    def __init__(
        self,
        *,
        J: int = 16,
        nu: float = 1.5,
        scale: float = 1.0,
        lengthscale: float = 1.0,
        active_dims: Optional[Sequence[int]] = None,
        n_dims: Optional[int] = None,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())

        self.J = J

        self.nu = PositiveReal(jnp.array(nu))
        self.scale = PositiveReal(jnp.array(scale))
        self.lengthscale = PositiveReal(jnp.array(lengthscale))

    def _mixture_params(self):
        nu = self.nu
        rho = self.lengthscale

        alpha = nu - 1.0
        z, w = _glaguerre_gw(self.J, alpha)

        ell2 = z * (rho**2) / nu
        sig2 = (self.scale**2) * w / jnp.exp(jsp.special.gammaln(nu))

        return ell2, sig2

    def __call__(self, x, y):
        x = x.squeeze()
        y = y.squeeze()

        tau2 = jnp.sum((x - y) ** 2)

        ell2, sig2 = self._mixture_params()

        return jnp.sum(sig2 * jnp.exp(-0.5 * tau2 / ell2))


if __name__ == "__main__":
    import numpy as np
    from scipy.special import gamma, kv

    def exact_matern_np(tau, rho, sigma, nu):
        tau = np.asarray(tau)
        arg = np.sqrt(2 * nu) * tau / rho
        factor = (2 ** (1 - nu)) / gamma(nu)
        return sigma**2 * factor * (arg**nu) * kv(nu, arg)

    def test_accuracy():
        taus = np.linspace(1e-4, 5.0, 400)

        for nu in [1.5, 2.5, 4.0]:
            print(f"\nnu = {nu}")
            exact = exact_matern_np(taus, 1.0, 1.0, nu)

            for J in [4, 8, 16, 32]:
                k = gpxMaternSEMixture(J=J, nu=nu, scale=1.0, lengthscale=1.0)

                approx = np.array(
                    [float(k(jnp.array([t]), jnp.array([0.0]))) for t in taus]
                )

                rel = np.max(np.abs(approx - exact) / np.maximum(exact, 1e-12))
                rms = np.sqrt(np.mean((approx - exact) ** 2))

                print(f"  J={J:2d}   max rel err = {rel:.2e}   RMS = {rms:.2e}")

    test_accuracy()


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from utils.jax import vk

    meanf = gpx.mean_functions.Zero()

    J = 16

    kernels = [
        gpxMaternSEMixture(J=J, nu=0.5),
        gpxMaternSEMixture(J=J, nu=2.5),
        gpxMaternSEMixture(J=J, nu=4.0),
    ]

    t = jnp.linspace(-3.0, 3.0, 1024)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

    for k, ax in zip(kernels, axes, strict=False):
        prior = gpx.gps.Prior(mean_function=meanf, kernel=k)
        rv = prior(t[:, None])
        y = rv.sample(key=vk(), sample_shape=(3,))
        ax.plot(t, y.T, alpha=0.7)
        ax.set_title(f"SE-mixture Matérn ν={float(k.nu):.2f}")

    plt.show()

# %%
