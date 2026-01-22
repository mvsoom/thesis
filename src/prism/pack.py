# %%
import equinox as eqx
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel

from gfm.ack import ACK


class PACK(Kernel):
    """Stationary PACK kernel with optional per-harmonic sigma_c"""

    d: int = eqx.field(static=True)
    normalized: bool = eqx.field(static=True, default=False)
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

# %%
from typing import Optional, Sequence

import gpjax as gpx
import jax.numpy as jnp
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import PositiveReal


class gpxPACK(gpx.kernels.AbstractKernel):
    # static kernel structure
    d: int
    normalized: bool
    J: int
    period: float

    # learnable parameters
    sigma_b: PositiveReal
    sigma_c: PositiveReal  # scalar; vector handled by broadcasting if needed

    def __init__(
        self,
        d: int,
        J: int = 1,
        normalized: bool = True,
        period: float = 1.0,
        sigma_b: float = 1.0,
        sigma_c: float = 1.0,
        active_dims: Optional[Sequence[int]] = None,
        n_dims: Optional[int] = None,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())

        # static
        self.d = d
        self.normalized = normalized
        self.J = J
        self.period = period

        # GPJax-tracked parameters
        self.sigma_b = PositiveReal(jnp.array(sigma_b))
        self.sigma_c = PositiveReal(jnp.ones(J) * sigma_c)

    def __call__(self, x, y):
        """
        x, y: shape (1 D) == (1, 1)
        """
        t1 = x.squeeze()
        t2 = y.squeeze()

        k = PACK(
            d=self.d,
            normalized=self.normalized,
            J=self.J,
            period=self.period,
            sigma_b=self.sigma_b,
            sigma_c=self.sigma_c,
        )

        return k.evaluate(t1, t2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from utils.jax import vk

    J = 2

    kernels = [gpxPACK(d=d, J=J, normalized=True) for d in [0, 1, 2, 3]]
    meanf = gpx.mean_functions.Zero()

    t = jnp.linspace(-1.0, 2.0, 512)  # .reshape(-1, 1)

    fig, axes = plt.subplots(
        ncols=2, nrows=2, figsize=(7, 6), tight_layout=True
    )

    for k, ax in zip(kernels, axes.ravel(), strict=False):
        prior = gpx.gps.Prior(mean_function=meanf, kernel=k)
        rv = prior(t)
        y = rv.sample(key=vk(), sample_shape=(3,))
        ax.plot(t, y.T, alpha=0.7)
        ax.set_title(k.name)

# %%
import jax.numpy as jnp
from tinygp.helpers import JAXArray

from gfm.ack import compute_Jd


def pack_arccos_kernel(
    t1: JAXArray,
    t2: JAXArray,
    *,
    d: int,
    period: float,
    sigma_b: JAXArray,
    sigma_c: JAXArray,  # shape (J,)
    normalized: bool,
):
    """
    Stable ArcCos(d) kernel for periodic PACK embedding.
    """

    # frequencies
    J = sigma_c.shape[0]
    j = jnp.arange(1, J + 1)
    w = 2.0 * jnp.pi / period

    # Δ = t1 - t2
    delta = t1 - t2

    # squared norm (constant!)
    S = sigma_b**2 + jnp.sum(sigma_c**2)

    # inner product
    dot = sigma_b**2 + jnp.sum(sigma_c**2 * jnp.cos(w * j * delta))

    # cosine of angle
    c = dot / S

    # strictly interior clip for gradients
    eps_c = 1e-7
    c = jnp.clip(c, -1.0 + eps_c, 1.0 - eps_c)

    theta = jnp.arccos(c)
    s = jnp.sin(theta)

    # ArcCos polynomial
    Jd = compute_Jd(d, c, s)

    if normalized:
        Jd0 = compute_Jd(d, 1.0 - eps_c, 0.0)
        return Jd / Jd0
    else:
        # ||x||^d ||x'||^d = S^(d/2) * S^(d/2) = S^d
        return (1.0 / jnp.pi) * (S**d) * Jd


class gpxPACKStable(gpx.kernels.AbstractKernel):
    # static kernel structure
    d: int
    normalized: bool
    J: int
    period: float

    # learnable parameters
    sigma_b: PositiveReal
    sigma_c: PositiveReal  # scalar; vector handled by broadcasting if needed

    def __init__(
        self,
        d: int,
        J: int = 1,
        normalized: bool = True,
        period: float = 1.0,
        sigma_b: float = 1.0,
        sigma_c: float = 1.0,
        active_dims: Optional[Sequence[int]] = None,
        n_dims: Optional[int] = None,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())

        # static
        self.d = d
        self.normalized = normalized
        self.J = J
        self.period = period

        # GPJax-tracked parameters
        self.sigma_b = PositiveReal(jnp.array(sigma_b))
        self.sigma_c = PositiveReal(jnp.ones(J) * sigma_c)

    def __call__(self, x, y):
        """
        x, y: shape (1 D) == (1, 1)
        """
        t1 = x.squeeze()
        t2 = y.squeeze()

        return pack_arccos_kernel(
            t1,
            t2,
            d=self.d,
            period=self.period,
            sigma_b=self.sigma_b,
            sigma_c=self.sigma_c,
            normalized=self.normalized,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from utils.jax import vk

    J = 2

    kernels = [gpxPACKStable(d=d, J=J) for d in [0, 1, 2, 3]]
    meanf = gpx.mean_functions.Zero()

    t = jnp.linspace(-1.0, 2.0, 528)  # .reshape(-1, 1)

    fig, axes = plt.subplots(
        ncols=2, nrows=2, figsize=(7, 6), tight_layout=True
    )

    for k, ax in zip(kernels, axes.ravel(), strict=False):
        prior = gpx.gps.Prior(mean_function=meanf, kernel=k)
        rv = prior(t)
        y = rv.sample(key=vk(), sample_shape=(3,))
        ax.plot(t, y.T, alpha=0.7)
        ax.set_title(f"PACK(d={k.d})")

# %%
if __name__ == "__main__":
    k = gpxPACK(d=2, J=J)
    ks = gpxPACKStable(d=2, J=J)

    x, y = jnp.array([0.8456]), jnp.array([-1.5641])

    k(x, y), ks(x, y)

# %%
from typing import Optional, Sequence

import gpjax as gpx
import jax.numpy as jnp
from gpjax.parameters import PositiveReal


class NormalizedPACK(gpx.kernels.AbstractKernel):
    """
    Normalized PACK ArcCos kernel with explicit amplitude sigma_a.

    Inputs: x, y are 1D time points shaped like (1,) or (1,1) etc.
    Kernel is stationary and periodic with period.

    k(t, t') = sigma_a^2 * J_d(c(Δ), s(Δ)) / J_d(1, 0)
    where
      c(Δ) = (sigma_b^2 + Σ_j sigma_cj^2 cos(ω j Δ)) / (sigma_b^2 + Σ_j sigma_cj^2)
      ω = 2π / period
      Δ = t - t'
      s(Δ) = sin(arccos(c(Δ))) computed stably
    """

    # static
    d: int
    J: int
    period: float

    # learnable
    sigma_a: PositiveReal
    sigma_b: PositiveReal
    sigma_c: PositiveReal  # can be scalar or (J,) depending on init

    def __init__(
        self,
        *,
        d: int,
        J: int = 1,
        period: float = 1.0,
        sigma_a: float = 1.0,
        sigma_b: float = 1.0,
        sigma_c: float | jnp.ndarray = 1.0,
        active_dims: Optional[Sequence[int]] = None,
        n_dims: Optional[int] = None,
        eps_c: float = 1e-7,
    ):
        super().__init__(active_dims, n_dims, DenseKernelComputation())

        self.d = int(d)
        self.J = int(J)
        self.period = float(period)

        self._eps_c = float(eps_c)  # keep c strictly inside (-1, 1)

        self.sigma_a = PositiveReal(jnp.array(sigma_a))
        self.sigma_b = PositiveReal(jnp.array(sigma_b))

        sc = jnp.array(sigma_c)
        if sc.ndim == 0:
            sc = jnp.ones((self.J,)) * sc
        if sc.shape != (self.J,):
            raise ValueError(
                f"sigma_c must be scalar or shape ({self.J},), got {sc.shape}"
            )
        self.sigma_c = PositiveReal(sc)

        # precompute normalization constant J_d(1, 0)
        # use the same eps_c convention to avoid boundary grads
        c0 = 1.0 - self._eps_c
        self._Jd0 = compute_Jd(self.d, c0, 0.0)

    def __call__(self, x, y):
        # Accept (1,), (1,1), (D,) etc but treat as scalar time
        t1 = x.squeeze()
        t2 = y.squeeze()

        J = self.J
        j = jnp.arange(1, J + 1)

        w = 2.0 * jnp.pi / self.period
        delta = t1 - t2

        sb2 = self.sigma_b**2
        sc2 = self.sigma_c**2  # (J,)

        S = sb2 + jnp.sum(sc2)  # constant norm^2 of embedding
        dot = sb2 + jnp.sum(sc2 * jnp.cos(w * j * delta))

        c = dot / S
        eps_c = self._eps_c
        c = jnp.clip(c, -1.0 + eps_c, 1.0 - eps_c)

        theta = jnp.arccos(c)
        s = jnp.sin(theta)

        Jd = compute_Jd(self.d, c, s)
        kshape = Jd / self._Jd0

        return (self.sigma_a**2) * kshape


if __name__ == "__main__":
    k = gpxPACK(d=2, J=J)
    ks = NormalizedPACK(d=2, J=J)

    x, y = jnp.array([0.3]), jnp.array([0.1])

    k(x, y), ks(x, y)