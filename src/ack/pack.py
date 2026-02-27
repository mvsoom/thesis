# %%
import jax.numpy as jnp
from gpjax.gps import Prior
from gpjax.mean_functions import Zero
from gpjax.parameters import PositiveReal  # keep using this

from ack.parameters import Simplex
from gfm.ack import compute_Jd
from prism.harmonic import SHMKernel
from utils.jax import vk


class PACK(SHMKernel):
    """
    Normalized PACK ArcCos kernel with explicit marginal variance and simplex weights.

    Stationary + periodic with period T.

    k(Δ) = variance * J_d(c(Δ), s(Δ)) / J_d(1, 0)

    where
      c(Δ) = w0 + sum_{j=1..J} wj * cos(j * ω * Δ)
      ω = 2π / period
      Δ = t - t'
      s(Δ) = sqrt(1 - c(Δ)^2) (computed via arccos/sin for stability)

    Constraints:
      variance > 0
      w in simplex: wj >= 0, sum_j wj = 1

    Notes:
      - This removes the global scale redundancy of (sigma_b, sigma_c).
      - k(0) = variance exactly (up to eps_c convention in J_d(1,0)).
    """

    d: int
    J: int
    period: float

    variance: PositiveReal
    weights: Simplex  # shape (J+1,)

    def __init__(
        self,
        d: int,
        J: int = 1,
        period: float = 1.0,
        variance: float = 1.0,
        weights: jnp.ndarray | None = None,
        eps=1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d = int(d)
        self.J = int(J)
        self.period = float(period)
        self.eps = float(eps)

        self.variance = PositiveReal(jnp.asarray(variance))

        if weights is None:
            weights = jnp.ones((self.J + 1,))
        else:
            assert weights.shape == (self.J + 1,), (
                f"weights must have shape ({self.J + 1},), got {weights.shape}"
            )

        weights = weights / jnp.sum(weights)

        self.weights = Simplex(weights)

        # precompute normalization constant J_d(1, 0)
        c0 = 1.0 - self.eps
        self._Jd0 = compute_Jd(self.d, c0, 0.0)

    def __call__(self, x, y):
        t1 = x.squeeze()
        t2 = y.squeeze()

        delta = t1 - t2
        w0 = self.weights[0]
        wj = self.weights[1:]  # (J,)

        j = jnp.arange(1, self.J + 1)
        omega0 = 2.0 * jnp.pi / self.period

        c = w0 + jnp.sum(wj * jnp.cos(omega0 * j * delta))
        c = jnp.clip(c, -1.0 + self.eps, 1.0 - self.eps)

        theta = jnp.arccos(c)
        s = jnp.sin(theta)

        Jd = compute_Jd(self.d, c, s)
        kshape = Jd / self._Jd0

        return self.variance * kshape


def weights_to_sigmas(w, scale=1.0):
    sb2 = scale * w[0]
    sc2 = scale * w[1:]
    sigma_b = jnp.sqrt(sb2)
    sigma_c = jnp.sqrt(sc2)
    return sigma_b, sigma_c


if __name__ == "__main__":
    # simple test of the kernel
    import matplotlib.pyplot as plt

    k = PACK(d=1, J=4, period=3.0, variance=2.0)
    t = jnp.linspace(0, 9.0, 512)
    K = k.gram(t).to_dense()

    plt.imshow(K)
    plt.colorbar()
    plt.title("Normalized PACK Kernel Matrix")
    plt.xlabel("t'")
    plt.ylabel("t")
    plt.show()

# %%
if __name__ == "__main__":
    prior = Prior(mean_function=Zero(), kernel=k)
    samples = prior.predict(t).sample(vk(), (1,))
    plt.plot(t, samples.T)
    plt.title("PACK sample")
    plt.show()

# %%
if __name__ == "__main__":
    from prism.pack import NormalizedPACK

    k1 = PACK(d=1, J=4, period=3.0, variance=2.0)

    sb, sc = weights_to_sigmas(k1.weights, scale=1.0)
    k2 = NormalizedPACK(
        d=1, J=4, period=3.0, sigma_a=jnp.sqrt(2.0), sigma_b=(sb), sigma_c=sc
    )

    K1 = k1.gram(t).to_dense()
    K2 = k2.gram(t).to_dense()

    print("Max abs diff:", jnp.max(jnp.abs(K1 - K2)))  # ok

if __name__ == "__main__":
    import jax
    from gpjax.likelihoods import Gaussian

    from prism.harmonic import SHMCollapsedVariationalGaussian, SHMPeriodicFFT
    from prism.svi import svi_basis

    kfft = SHMPeriodicFFT(k)

    prior = Prior(kfft, Zero())
    likelihood = Gaussian(num_datapoints=len(t))
    posterior = prior * likelihood

    M = 16

    qsvi_vff = SHMCollapsedVariationalGaussian(posterior=posterior, M=M)

    Phi = jax.vmap(svi_basis, in_axes=(None, 0))(qsvi_vff, t)
    eps = jax.random.normal(vk(), shape=(Phi.shape[1], 2))
    y = Phi @ eps

    plt.plot(t, y)

    # %%
    A, mu = kfft.compute_shm(M)

    lhs = kfft(jnp.array(0.0), jnp.array(0.0))
    rhs = (1 / (2 * jnp.pi)) * (A[0] + 2 * A[1:].sum())

    print("k(0,0) =", lhs)
    print("k(0,0) from spectrum =", rhs)  # equal? ok

    # %%
    r = t[:, None] - t[None, :]

    for Mtry in [8, 16, 32, 64]:

        def k_from_shm(r):
            return kfft.k_from_shm(Mtry, r)

        K_shm = jax.vmap(k_from_shm)(r)

        print(
            f"Mtry = {Mtry:2} Max abs diff:", jnp.max(jnp.abs(K1 - K_shm))
        )  # ok
