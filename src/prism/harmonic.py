# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
from gpjax.kernels import AbstractKernel, Periodic
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.variational_families import CollapsedVariationalGaussian

from gp.periodic import periodic_se_series_coeffs
from prism.spectral import (
    SGMCollapsedVariationalGaussian,
    complex_to_real_Kuf,
    complex_to_real_Kuu,
)
from prism.svi import (
    svi_basis,
)
from utils.jax import vk


class SHMCollapsedVariationalGaussian(SGMCollapsedVariationalGaussian):
    """Special case of Spectral Gaussian Mixture for line spectrum

    This is the case for periodic kernels:
        1) Line spectrum at harmonics: Gaussian mixture -> variance 0
        2) Input density: Gaussian window -> variance infty

    Due to line spectrum inducing points are fixed at harmonics.
    Period is always set to 1 (just rescale time)
    DC is always included
    """

    M: int  # number of harmonics NOT including DC (GP rank is 2M+1)

    def __init__(self, M, *args, **kwargs):
        kernel = kwargs["posterior"].prior.kernel
        period = kernel.period
        freqs = (1.0 + jnp.arange(M)) / period
        inducing_inputs = freqs[:, None]
        super().__init__(
            *args, inducing_inputs=inducing_inputs, sigma_w=jnp.inf, **kwargs
        )
        self.inducing_inputs = jnp.array(
            self.inducing_inputs.value
        )  # not trainable
        self.M = M

    def compute_Kuu(self):
        kernel = self.posterior.prior.kernel
        A, _ = kernel.compute_shm(self.M)
        Kuu_complex = jnp.diag(A / (2.0 * jnp.pi))
        return complex_to_real_Kuu(Kuu_complex)

    def compute_Kuf(self, t):
        kernel = self.posterior.prior.kernel
        A, mu = kernel.compute_shm(self.M)
        tau = jnp.ravel(t)
        Kuf_complex = (A / (2.0 * jnp.pi))[:, None] * jnp.exp(
            -1j * mu[:, None] * tau[None, :]
        )
        return complex_to_real_Kuf(Kuf_complex)


class SHMKernel(AbstractKernel):
    def compute_shm(self, M):
        """Return (A, mu) for the line spectrum of the periodic kernel

        Here A is the line MASS (ie prefactor of the delta) and mu = (2pi) f (radians per unit time)"""
        raise NotImplementedError


class SHMPeriodic(SHMKernel, Periodic):
    def compute_shm(self, J):
        ell = 2.0 * self.lengthscale  # match tinygp convention
        q2 = periodic_se_series_coeffs(ell, J)

        js = jnp.arange(J + 1, dtype=q2.dtype)
        mu = (2.0 * jnp.pi / self.period) * js  # angular

        # S(omega)=(2pi) q0^2 delta(0) + (pi) sum_{j>=1} qj^2 [delta(+/- omega_j)]
        A = jnp.pi * q2
        A = A.at[0].set(2.0 * jnp.pi * q2[0])  # DC has double mass

        A = A * self.variance

        return A, mu


if __name__ == "__main__":
    import numpy as np

    from gp.periodic import PeriodicSE

    dt = 0.05
    M = 512
    t = np.arange(M) * dt

    J = 24

    variance = 1.21
    period = 4.89
    ell = 2.75

    k_gpjax = SHMPeriodic(variance=variance, lengthscale=ell / 2, period=period)
    k_tinygp = variance * PeriodicSE(ell=jnp.array(ell), period=period, J=J)

    K_gpjax = k_gpjax.gram(t[:, None]).to_dense()
    K_tinygp = k_tinygp(t, t)

    print("Max abs diff:", np.max(np.abs(K_gpjax - K_tinygp)))  # ok

# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    kernel = k_gpjax

    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=len(t))
    posterior = prior * likelihood

    Z = jax.random.choice(vk(), t, (8,))[:, None]

    qsvi = CollapsedVariationalGaussian(posterior=posterior, inducing_inputs=Z)

    from prism.svi import svi_basis

    Phi = jax.vmap(svi_basis, in_axes=(None, 0))(qsvi, t)
    eps = jax.random.normal(vk(), shape=(Phi.shape[1], 2))
    y = Phi @ eps

    plt.plot(t, y)

# %%
if __name__ == "__main__":
    qsvi_vff = SHMCollapsedVariationalGaussian(posterior=posterior, M=J)

    Phi = jax.vmap(svi_basis, in_axes=(None, 0))(qsvi_vff, t)
    eps = jax.random.normal(vk(), shape=(Phi.shape[1], 2))
    y = Phi @ eps

    plt.plot(t, y)
