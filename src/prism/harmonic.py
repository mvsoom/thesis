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

    def __init__(self, M, *args, **kwargs):
        kernel = kwargs["posterior"].prior.kernel
        period = kernel.period
        freqs = (1.0 + jnp.arange(M)) / period
        inducing_inputs = freqs[:, None]
        super().__init__(
            *args, inducing_inputs=inducing_inputs, sigma_w=jnp.inf, **kwargs
        )
        self.inducing_inputs = jnp.array(
            self.inducing_inputs[...]
        )  # not trainable

    def compute_Kuu(self):
        kernel = self.posterior.prior.kernel
        A, _ = kernel.compute_shm()
        Kuu_complex = jnp.diag(A / (2.0 * jnp.pi))
        return complex_to_real_Kuu(Kuu_complex)

    def compute_Kuf(self, t):
        kernel = self.posterior.prior.kernel
        A, mu = kernel.compute_shm()
        tau = jnp.ravel(t)
        Kuf_complex = (A / (2.0 * jnp.pi))[:, None] * jnp.exp(
            -1j * mu[:, None] * tau[None, :]
        )
        return complex_to_real_Kuf(Kuf_complex)


class SHMKernel(AbstractKernel):
    def __init__(self, *args, num_harmonics, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_harmonics = num_harmonics

    def compute_shm(self):
        """Return (A, mu) for the line spectrum of the periodic kernel

        A and mu have shape (M+1,) where M is the number of harmonics self.num_harmonics (not including DC).

        Here A is the line MASS (ie prefactor of the delta) and mu = (2pi) f (radians per unit time)
        """
        raise NotImplementedError

    def k_from_shm(self, r):
        """k(r) from cosine series"""
        A, mu = self.compute_shm()
        A0 = A[0]
        Apos = A[1:]
        mup = mu[1:]
        return (A0 / (2.0 * jnp.pi)) + jnp.sum(
            (Apos / jnp.pi) * jnp.cos(mup[None, :] * r[:, None]), axis=1
        )


class SHMPeriodic(SHMKernel, Periodic):
    def compute_shm(self):
        J = self.num_harmonics
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

    k_gpjax = SHMPeriodic(
        variance=variance, lengthscale=ell / 2, period=period, num_harmonics=J
    )
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

# %%
import jax
import jax.numpy as jnp

from prism.harmonic import SHMKernel


class SHMPeriodicFFT(SHMKernel):
    """
    Generic SHM line-spectrum wrapper for a stationary periodic kernel k(t,t').

    Assumptions:
      - k is stationary and periodic with known period T
      - we can evaluate k(delta, 0) for delta in [0, T)

    Returns A, mu such that:
      S(xi) = A0 * delta(xi) + sum_{m>=1} Am * [delta(xi-mu_m)+delta(xi+mu_m)]
      k(r) = (1/2pi) ∫ S(xi) exp(i xi r) dxi.
    """

    def __init__(
        self,
        kernel,
        *,
        N_min: int = 512,
        oversamp: int = 8,
        clip_negative: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.N_min = int(N_min)
        self.oversamp = int(oversamp)
        self.clip_negative = bool(clip_negative)

    @property
    def period(self):
        return self.kernel.period

    def __call__(self, x, y):
        return self.kernel(x, y)

    def _choose_N(self, M: int) -> int:
        M = int(M)
        N = max(self.N_min, self.oversamp * 2 * (M + 1))
        return int(1 << (N - 1).bit_length())

    def compute_shm(self):
        M = self.num_harmonics
        N = self._choose_N(M)

        T = self.kernel.period
        omega0 = (2.0 * jnp.pi) / T

        # sample k(delta) over one period
        delta = jnp.linspace(0.0, T, N, endpoint=False)

        # evaluate k(delta, 0)
        # keep shapes simple: pass (1,) arrays if your kernel expects that
        x = delta[:, None]
        y = jnp.zeros((1,))

        def k0(xi):
            return self.kernel(xi, y)

        k = jax.vmap(k0)(x).reshape(-1)  # (N,)

        # rFFT gives complex Fourier series coeffs for samples on uniform grid
        K = jnp.fft.rfft(k) / N  # (N//2+1,)
        a0 = jnp.real(K[0])
        am = 2.0 * jnp.real(K[1 : M + 1])  # cosine series coefficients

        a = jnp.concatenate([jnp.array([a0]), am], axis=0)  # (M+1,)

        if self.clip_negative:
            a = jnp.clip(a, 0.0)

        mu = omega0 * jnp.arange(M + 1, dtype=a.dtype)  # (M+1,)

        # map cosine coefficients -> line masses under your convention
        A = jnp.pi * a
        A = A.at[0].set(2.0 * jnp.pi * a[0])  # DC double mass

        return A, mu


if __name__ == "__main__":
    d = 1
    kfft = SHMPeriodicFFT(kernel, num_harmonics=16)
    prior = gpx.gps.Prior(kfft, Zero())
    likelihood = Gaussian(num_datapoints=len(t))
    posterior = prior * likelihood

    qsvi_vff = SHMCollapsedVariationalGaussian(posterior=posterior, M=J)

    Phi = jax.vmap(svi_basis, in_axes=(None, 0))(qsvi_vff, t)
    eps = jax.random.normal(vk(), shape=(Phi.shape[1], 2))
    y = Phi @ eps

    plt.plot(t, y)

    # %%
    A, mu = kfft.compute_shm()

    lhs = kfft(jnp.array([[0.0]]), jnp.array([[0.0]]))
    rhs = (1 / (2 * jnp.pi)) * (A[0] + 2 * A[1:].sum())

    print("k(0,0) =", lhs)
    print("k(0,0) from spectrum =", rhs)  # equal? ok

# %%
import jax.numpy as jnp
from gpjax.kernels import AbstractKernel

from prism.harmonic import SHMKernel
from prism.spectral import SGMKernel


class SGMQuasiPeriodic(SGMKernel):
    am: SGMKernel
    periodic: SHMKernel

    def __init__(self, am: SGMKernel, periodic: SHMKernel, **kwargs):
        super().__init__(**kwargs)

        periodic.variance = jnp.asarray(
            periodic.variance[...]
        )  # don't train as this isn't identifiable due to am.variance

        self.am = am
        self.periodic = periodic

    def __call__(self, x, y):
        return self.am(x, y) * self.periodic(x, y)

    def compute_sgm(self):
        # AM spectrum: symmetric by construction in Kuu/Kuf (mu>=0 expected)
        A, mu, v = self.am.compute_sgm()  # (Q,)
        Ap, mup = self.periodic.compute_shm()  # (J+1,) with mup[0]=0

        Q = A.shape[0]
        Jplus = Ap.shape[0]
        assert mup.shape[0] == Jplus

        # Required by Fourier convention:
        # S_product = (1/2π) (S_am * S_per)
        conv_scale = 1.0 / (2.0 * jnp.pi)

        # -------------------------------------------------
        # p = 0 (DC delta): scales AM spectrum
        # -------------------------------------------------
        A0 = Ap[0]  # mass at 0
        A_dc = conv_scale * (A0 * A)
        mu_dc = mu
        v_dc = v

        if Jplus == 1:
            return A_dc, mu_dc, v_dc

        # -------------------------------------------------
        # p >= 1: shift by ±mup
        # -------------------------------------------------
        Ap_pos = Ap[1:]  # (J,)
        mup_pos = mup[1:]  # (J,)

        # broadcast shapes (Q,J)
        AqAp = conv_scale * (A[:, None] * Ap_pos[None, :])  # (Q,J)

        mu_sum = mu[:, None] + mup_pos[None, :]  # (Q,J)
        mu_dif = jnp.abs(mu[:, None] - mup_pos[None, :])  # (Q,J)

        # variances unchanged by delta shift
        v_rep = jnp.broadcast_to(v[:, None], mu_sum.shape)  # (Q,J)

        # flatten and concatenate
        A_all = jnp.concatenate(
            [
                A_dc,
                AqAp.reshape(-1),  # sum branch
                AqAp.reshape(-1),  # abs-diff branch
            ],
            axis=0,
        )

        mu_all = jnp.concatenate(
            [
                mu_dc,
                mu_sum.reshape(-1),
                mu_dif.reshape(-1),
            ],
            axis=0,
        )

        v_all = jnp.concatenate(
            [
                v_dc,
                v_rep.reshape(-1),
                v_rep.reshape(-1),
            ],
            axis=0,
        )
        return A_all, mu_all, v_all


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from gpjax.mean_functions import Zero

    from prism.matern import SGMRBF

    # -------------------------------------------------
    # 1) Build AM and periodic components
    # -------------------------------------------------

    dt = 0.05
    N = 512
    t = np.arange(N) * dt
    t_jax = jnp.asarray(t)

    period = 4.0
    ell = 1.2
    variance_am = 1.5
    variance_per = 1.0

    # Amplitude modulation (single Gaussian, symmetric spectrum)
    am = SGMRBF(variance=variance_am, lengthscale=ell)

    # Periodic (analytic line symmetric spectrum)
    per = SHMPeriodic(
        variance=variance_per,
        lengthscale=ell / 2,
        period=period,
        num_harmonics=16,
    )

    # Quasi-periodic SGM
    k_qp = SGMQuasiPeriodic(am=am, periodic=per)

    # -------------------------------------------------
    # 2) Compare k(0)
    # -------------------------------------------------

    A_qp, mu_qp, v_qp = k_qp.compute_sgm()

    k00_direct = k_qp(jnp.array([[0.0]]), jnp.array([[0.0]]))

    # symmetric SGM convention:
    # k(0) = (1/pi) * sum_q A_q
    k00_spec = (1.0 / jnp.pi) * jnp.sum(A_qp)

    print("k(0) direct:", float(k00_direct))
    print("k(0) spectrum:", float(k00_spec))
    print("abs diff:", float(jnp.abs(k00_direct - k00_spec)))

    # -------------------------------------------------
    # 3) Time-domain reconstruction check
    # -------------------------------------------------

    r = jnp.linspace(0.0, period, 2048)

    # direct product kernel
    def k_dir1(r):
        return k_qp(jnp.array([[r]]), jnp.array([[0.0]]))

    k_dir = jax.vmap(k_dir1)(r)

    # reconstruct from SGM
    # S(xi) = sum A_q [N(+mu_q,v_q)+N(-mu_q,v_q)]
    # k(r) = (1/2pi) ∫ S(xi) e^{i xi r} dxi
    # => closed form for each Gaussian term:
    # k_q(r) = (A_q/pi) * exp(-0.5 v_q r^2) * cos(mu_q r)

    k_rec = jnp.zeros_like(r)

    for Aq, muq, vq in zip(A_qp, mu_qp, v_qp):
        k_rec += (Aq / jnp.pi) * jnp.exp(-0.5 * vq * r**2) * jnp.cos(muq * r)

    err = jnp.max(jnp.abs(k_dir - k_rec))
    print("max time-domain abs error:", float(err))

    plt.figure()
    plt.plot(r, k_dir, label="direct product")
    plt.plot(r, k_rec, "--", label="reconstructed from SGM")
    plt.legend()
    plt.title("Quasi-periodic kernel validation")
    plt.show()