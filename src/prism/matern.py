# %%
from typing import Optional, Sequence

import gpjax as gpx
import jax.numpy as jnp
import jax.scipy as jsp
from gpjax.kernels import RBF
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import PositiveReal

from prism.spectral import SGMKernel


class SGMRBF(SGMKernel, RBF):
    def compute_sgm(self):
        # RBF kernel: k(r) = sigma2 * exp(-r^2 / (2 ell^2))
        # Under our Fourier convention, one valid spectral form is:
        #   S(xi) = (2pi) * sigma2 * N(xi | 0, 1/ell^2)
        sigma2 = self.variance
        ell = self.lengthscale

        A = jnp.array([2.0 * jnp.pi * sigma2])
        mu = jnp.array([0.0])
        v = jnp.array([1.0 / (ell * ell)])

        A = A / 2  # avoid double counting as this is centered on zero

        return A, mu, v


# %%
def _glaguerre_gw(J, alpha):
    n = jnp.arange(J)
    diag = 2.0 * n + 1.0 + alpha
    k = jnp.arange(1, J)
    off = jnp.sqrt(k * (k + alpha))

    A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    z, V = jnp.linalg.eigh(A)

    mu0 = jnp.exp(jsp.special.gammaln(alpha + 1.0))
    w = mu0 * (V[0, :] ** 2)
    return z, w


class SGMMatern(SGMKernel):
    """
    Matérn kernel via squared exponential mixture (Tronarp+ 2018).

    Static:
        J   number of mixture components
        nu  smoothness

    Learnable:
        scale
        lengthscale
        (nu) <=== can be optimized in principle
    """

    J: int
    nu: PositiveReal

    lengthscale: PositiveReal
    variance: PositiveReal

    def __init__(
        self,
        *,
        nu: float = 1.5,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        J: int = 16,
        active_dims: Optional[Sequence[int]] = None,
        n_dims: Optional[int] = None,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())

        self.J = J

        self.nu = PositiveReal(jnp.array(nu))
        self.variance = PositiveReal(jnp.array(variance))
        self.lengthscale = PositiveReal(jnp.array(lengthscale))

    def _mixture_params(self):
        nu = self.nu
        rho = self.lengthscale

        z, w = _glaguerre_gw(self.J, nu - 1.0)

        ell2 = z * (rho**2) / nu
        sig2 = self.variance * w / jnp.exp(jsp.special.gammaln(nu))

        return ell2, sig2

    def __call__(self, x, y):
        x = x.squeeze()
        y = y.squeeze()

        tau2 = jnp.sum((x - y) ** 2)

        ell2, sig2 = self._mixture_params()

        return jnp.sum(sig2 * jnp.exp(-0.5 * tau2 / ell2))

    def compute_sgm(self):
        nu = self.nu
        rho = self.lengthscale
        sigma2 = self.variance

        z, w = _glaguerre_gw(self.J, nu - 1.0)

        # mixture parameters
        ell2 = z * (rho * rho) / nu
        sig2 = sigma2 * w / jnp.exp(jsp.special.gammaln(nu))

        # spectral Gaussian parameters
        A = 2.0 * jnp.pi * sig2
        mu = jnp.zeros_like(A)
        v = 1.0 / ell2

        A = A / 2  # avoid double counting as this is centered on zero

        return A, mu, v


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

        rho = 2.4
        var = 1.3
        sigma = np.sqrt(var)

        for nu in [1.5, 2.5, 4.0]:
            print(f"\nnu = {nu}")
            exact = exact_matern_np(taus, rho, sigma, nu)

            for J in [4, 8, 16, 32]:
                k = SGMMatern(J=J, nu=nu, variance=var, lengthscale=rho)

                approx = np.array(
                    [float(k(jnp.array([t]), jnp.array([0.0]))) for t in taus]
                )

                rel = np.max(
                    np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-12)
                )
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
        SGMMatern(J=J, nu=0.5),
        SGMMatern(J=J, nu=2.5),
        SGMMatern(J=J, nu=4.0),
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
