# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
from gpjax.parameters import NonNegativeReal
from gpjax.variational_families import CollapsedVariationalGaussian

from prism.svi import init_Z_grid


class SGMKernel(gpx.kernels.AbstractKernel):
    """Spectral Gaussian Mixture kernel in 1D

    Fourier convention:
        k(r) = (1/2pi) int S(xi) * exp(i xi r) dxi

    S(xi) is a Gaussian mixture:
        S(xi) = sum_q A_q * N(xi | mu_q, v_q)
    where each N(·|mu,v) is unit area, and A_q carries the mass.

    For a real stationary kernel, ensure S(xi)=S(-xi). You can either:
      - return already-symmetric (A, mu, v) with +/- pairs included, or
      - return only mu_q >= 0 and rely on the symmetric formulas below.
    """

    def compute_sgm(self):
        """Return (A, mu, v) each shaped (Q,) for the Q SGM components."""
        raise NotImplementedError


def _sgm_symm_Kuu_complex(A, mu, v, omega, sigma_w):
    """
    Complex Hermitian Kuu for inducing variables

        u_m = ∫ f(t) w(t) e^{-i omega_m t} dt

    with Gaussian window density w(t) = N(t | 0, sigma_w^2).

    Uses symmetric spectrum:
        S(xi) = sum_q A_q [N(xi | +mu_q,v_q) + N(xi | -mu_q,v_q)]

    Parameters
    ----------
    A, mu, v : (Q,)
        Spectral mixture parameters (mu >= 0).
    omega : (M,)
        Angular inducing frequencies.
    sigma_w : float
        Window stddev.

    Returns
    -------
    Kuu : (M,M) complex Hermitian
    """

    b = sigma_w * sigma_w
    omega = omega.reshape(-1)

    # shapes
    om_m = omega[:, None]  # (M,1)
    om_n = omega[None, :]  # (1,M)

    mu_q = mu.reshape(-1, 1, 1)  # (Q,1,1)
    v_q = v.reshape(-1, 1, 1)
    A_q = A.reshape(-1, 1, 1)

    om_m = om_m[None, :, :]  # (1,M,1)
    om_n = om_n[None, :, :]  # (1,1,M)

    # Common quantities
    P2 = (1.0 / v_q) + 2.0 * b  # (Q,1,1)
    pref = 1.0 / jnp.sqrt(1.0 + 2.0 * b * v_q)

    # helper for single sign of mu
    def I_single(mu_signed):
        gamma = mu_signed / v_q  # (Q,1,1)
        beta = b * (om_m + om_n)  # (1,M,M)

        expo = -0.5 * (
            mu_signed * mu_signed / v_q + b * (om_m * om_m + om_n * om_n)
        ) + (gamma + beta) ** 2 / (2.0 * P2)

        return pref * jnp.exp(expo)

    # symmetric sum over ±mu
    I_plus = I_single(+mu_q)
    I_minus = I_single(-mu_q)

    I_pair = I_plus + I_minus  # (Q,M,M)

    Kuu = (1.0 / (2.0 * jnp.pi)) * jnp.sum(A_q * I_pair, axis=0)

    # enforce Hermitian symmetry numerically
    Kuu = 0.5 * (Kuu + jnp.conj(Kuu.T))

    return Kuu


def _sgm_symm_Kuf_complex(A, mu, v, omega, tau, sigma_w):
    """
    Complex Kuf:

        Kuf_{m,n} = cov(u_m, f(tau_n))

    where
        u_m = ∫ f(t) w(t) e^{-i omega_m t} dt

    Uses symmetric spectrum S(xi) = sum_q A_q [N(+mu_q)+N(-mu_q)].

    Returns full complex matrix (M,N).
    """

    b = sigma_w * sigma_w

    omega = omega.reshape(-1)  # (M,)
    tau = jnp.asarray(tau).reshape(-1)  # (N,)

    mu_q = mu.reshape(-1, 1, 1)  # (Q,1,1)
    v_q = v.reshape(-1, 1, 1)
    A_q = A.reshape(-1, 1, 1)

    om = omega.reshape(1, -1, 1)  # (1,M,1)
    t = tau.reshape(1, 1, -1)  # (1,1,N)

    P = (1.0 / v_q) + b
    pref = 1.0 / jnp.sqrt(1.0 + b * v_q)

    def J_single(mu_signed):
        alpha = (mu_signed / v_q) + b * om  # (Q,M,1)

        expo = -0.5 * (mu_signed * mu_signed / v_q + b * (om * om)) + (
            alpha - 1j * t
        ) ** 2 / (2.0 * P)

        return pref * jnp.exp(expo)  # (Q,M,N)

    J_plus = J_single(+mu_q)
    J_minus = J_single(-mu_q)

    J_pair = J_plus + J_minus  # (Q,M,N)

    Kuf = (1.0 / (2.0 * jnp.pi)) * jnp.sum(A_q * J_pair, axis=0)

    return Kuf


def complex_to_real_Kuu(K):
    Re = jnp.real(K)
    Im = jnp.imag(K)
    top = jnp.concatenate([Re, -Im], axis=1)
    bottom = jnp.concatenate([Im, Re], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def complex_to_real_Kuf(K):
    return jnp.concatenate([jnp.real(K), jnp.imag(K)], axis=0)


class SGMCollapsedVariationalGaussian(CollapsedVariationalGaussian):
    """
    Drop-in for CollapsedVariationalGaussian.

    We repurpose inducing_inputs as frequencies in cycles-per-unit-tau (not angular).
    Internally we convert to angular omega = 2pi * inducing_inputs.

    sigma_w is the stddev of the (normalized) Gaussian window density in tau:
        w(tau) = N(tau | 0, sigma_w^2)
    """

    def __init__(self, *args, sigma_w=15.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_w = float(sigma_w)

        self.inducing_inputs = NonNegativeReal(self.inducing_inputs.value)

    def compute_Kuu(self):
        # inducing_inputs are frequencies (cycles per unit tau)
        omega = (2.0 * jnp.pi) * self.inducing_inputs.squeeze()
        kernel = self.posterior.prior.kernel
        A, mu, v = kernel.compute_sgm()

        # kernel returns only mu_q >= 0, these formulas enforce symmetry
        Kuu_complex = _sgm_symm_Kuu_complex(
            A=A, mu=mu, v=v, omega=omega, sigma_w=self.sigma_w
        )
        Kuu = complex_to_real_Kuu(Kuu_complex)
        return Kuu

    def compute_Kuf(self, t):
        omega = (2.0 * jnp.pi) * self.inducing_inputs.squeeze()
        kernel = self.posterior.prior.kernel
        A, mu, v = kernel.compute_sgm()

        # t are tau locations
        Kuf_complex = _sgm_symm_Kuf_complex(
            A=A, mu=mu, v=v, omega=omega, tau=t, sigma_w=self.sigma_w
        )
        Kuf = complex_to_real_Kuf(Kuf_complex)
        return Kuf

    @property
    def num_inducing(self):
        return (
            self.inducing_inputs.shape[0] * 2
        )  # account for real/imaginary split


def init_Z_inverse_ecdf_from_psd(key, M, freqs, Pxx):
    f = jnp.asarray(freqs)
    w = jnp.asarray(Pxx + 1e-12)  # avoid zeros
    w = w / jnp.sum(w)
    cdf = jnp.cumsum(w)
    u = init_Z_grid(key, M).reshape(-1)
    Z = jnp.interp(u, cdf, f)
    return Z[:, None]


def _numeric_sgm_spectrum_symm(A, mu, v, xi):
    """
    Symmetric spectrum:
      S(xi) = sum_q A_q [N(xi | +mu_q,v_q) + N(xi | -mu_q,v_q)]
    A, mu, v: (Q,)
    xi: (L,)
    returns: (L,)
    """
    xi = xi.reshape(1, -1)  # (1,L)
    A = A.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    v = v.reshape(-1, 1)

    def N(x, m, vv):
        return (1.0 / jnp.sqrt(2.0 * jnp.pi * vv)) * jnp.exp(
            -0.5 * ((x - m) ** 2) / vv
        )

    S = jnp.sum(A * (N(xi, +mu, v) + N(xi, -mu, v)), axis=0)
    return S.reshape(-1)


def _numeric_Kuf_complex(A, mu, v, omega, tau, sigma_w, xi_grid):
    """
    Kuf_{m,n} = (1/2pi) int S(xi) e^{-i xi tau_n} exp(-0.5 b (xi-omega_m)^2) dxi
    and then take real part (should be real under symmetry).
    """
    b = sigma_w * sigma_w
    S = _numeric_sgm_spectrum_symm(A, mu, v, xi_grid)  # (L,)
    dxi = xi_grid[1] - xi_grid[0]
    tau = jnp.asarray(tau).reshape(-1)  # (N,)
    omega = jnp.asarray(omega).reshape(-1)  # (M,)

    xi = xi_grid.reshape(1, 1, -1)  # (1,1,L)
    om = omega.reshape(-1, 1, 1)  # (M,1,1)
    t = tau.reshape(1, -1, 1)  # (1,N,1)
    Sx = S.reshape(1, 1, -1)  # (1,1,L)

    integrand = Sx * jnp.exp(-1j * xi * t) * jnp.exp(-0.5 * b * (xi - om) ** 2)
    val = (
        (1.0 / (2.0 * jnp.pi)) * jnp.sum(integrand, axis=-1) * dxi
    )  # (M,N) complex
    return val


def _numeric_Kuu(A, mu, v, omega, sigma_w, xi_grid):
    """
    Kuu_{m,n} = (1/2pi) int S(xi) exp(-0.5 b (xi-om_m)^2) exp(-0.5 b (xi-om_n)^2) dxi
    """
    b = sigma_w * sigma_w
    S = _numeric_sgm_spectrum_symm(A, mu, v, xi_grid)  # (L,)
    dxi = xi_grid[1] - xi_grid[0]
    omega = jnp.asarray(omega).reshape(-1)  # (M,)

    xi = xi_grid.reshape(1, 1, -1)  # (1,1,L)
    om_m = omega.reshape(-1, 1, 1)  # (M,1,1)
    om_n = omega.reshape(1, -1, 1)  # (1,M,1)
    Sx = S.reshape(1, 1, -1)  # (1,1,L)

    integrand = (
        Sx
        * jnp.exp(-0.5 * b * (xi - om_m) ** 2)
        * jnp.exp(-0.5 * b * (xi - om_n) ** 2)
    )
    val = (1.0 / (2.0 * jnp.pi)) * jnp.sum(integrand, axis=-1) * dxi  # (M,M)
    val = 0.5 * (val + val.T)
    return val


if __name__ == "__main__":
    # Smoke test + numeric agreement tests
    from prism.matern import SGMMatern

    key = jax.random.PRNGKey(0)

    # kernel = SGMRBF(lengthscale=0.85, variance=2.7)  # OK
    kernel = SGMMatern(J=32, nu=1.5, variance=4.78, lengthscale=0.85)  # OK
    A, mu, v = kernel.compute_sgm()
    print("SGM parameters:")
    print("A:", A)
    print("mu:", mu)
    print("v:", v)

    sigma_w = 15.0

    # Choose a few inducing freqs in cycles/tau, convert inside formulas
    freqs = jnp.array([0.01, 0.1, 1.0, 2.0, 3.5, 6.0])  # cycles per unit tau
    omega = (2.0 * jnp.pi) * freqs  # angular

    # Query points tau
    tau = jnp.linspace(-1.0, 1.0, 9)

    # Closed-form
    Kuu_cf = _sgm_symm_Kuu_complex(A, mu, v, omega, sigma_w)
    Kuf_cf = _sgm_symm_Kuf_complex(A, mu, v, omega, tau, sigma_w)

    # Numeric integration grid: must cover spectrum mass near 0 and window peaks near omega
    # Window term is narrow in xi around each omega with width ~ 1/sigma_w.
    # So cover [-max|omega|-pad, +max|omega|+pad] with small step.
    max_om = float(jnp.max(jnp.abs(omega)))
    pad = 2.0
    lo, hi = -max_om - pad, +max_om + pad

    # Step small enough to resolve the narrow window (~ 1/sigma_w)
    # Use about 8-12 points per stddev: dx ~ (1/sigma_w)/10
    dx = 1.0 / (sigma_w * 10.0)
    L = int(jnp.ceil((hi - lo) / dx)) + 1
    # Cap L to keep runtime reasonable in a smoke test
    L = min(L, 200_001)
    xi_grid = jnp.linspace(lo, hi, L)

    Kuu_num = _numeric_Kuu(A, mu, v, omega, sigma_w, xi_grid)
    Kuf_num = _numeric_Kuf_complex(A, mu, v, omega, tau, sigma_w, xi_grid)

    # Relative / absolute tolerances: numeric integral truncation dominates for large omega
    atol = 5e-4
    rtol = 5e-4

    print(
        "Kuu closed-form vs numeric max abs:",
        float(jnp.max(jnp.abs(Kuu_cf - Kuu_num))),
    )
    print(
        "Kuf closed-form vs numeric max abs:",
        float(jnp.max(jnp.abs(Kuf_cf - Kuf_num))),
    )

    assert jnp.allclose(Kuu_cf, Kuu_num, rtol=rtol, atol=atol), "Kuu mismatch"
    assert jnp.allclose(Kuf_cf, Kuf_num, rtol=rtol, atol=atol), "Kuf mismatch"

    # PSD-ish sanity (numerical): eigenvalues should be >= -tiny
    evals = jnp.linalg.eigvalsh(Kuu_cf)
    print("Kuu min eig:", float(jnp.min(evals)))
    assert float(jnp.min(evals)) > -1e-6, "Kuu not PSD (numerical tolerance)"
    print("All tests passed.")

    def sgm_k0_from_symm(A):
        # with your symmetric convention S = sum A (N+ + N-):
        # integral mass = 2 * sum A, so k(0) = (1/2pi) * 2 sum A = sum A / pi
        return jnp.sum(A) / jnp.pi

    def kernel_k0(kernel):
        return kernel(jnp.array([0.0]), jnp.array([0.0]))

    k0_kernel = kernel_k0(kernel)
    A, mu, v = kernel.compute_sgm()
    k0_spec = sgm_k0_from_symm(A)

    print("Mass tests:")
    print(" k(0) from kernel:", float(k0_kernel))
    print(" k(0) from spectrum mass:", float(k0_spec))
    print(" ratio spec/kernel:", float(k0_spec / k0_kernel))  # must be 1

    assert jnp.allclose(
        kernel(jnp.array([0.0]), jnp.array([0.0])),
        jnp.sum(A) / jnp.pi,
        rtol=1e-6,
    ), "k(0) mass mismatch"

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def _sample_from_kernel(key, kernel, t, nsamp=3, jitter=1e-6):
    """Direct GP sampling via covariance matrix."""
    t = jnp.asarray(t).reshape(-1, 1)
    N = t.shape[0]

    # build dense covariance
    K = jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(t))(t).squeeze()
    K = K + jitter * jnp.eye(N)

    L = jnp.linalg.cholesky(K)
    z = jax.random.normal(key, shape=(nsamp, N))
    return z @ L.T


def _sample_from_spectrum(key, A, mu, v, t, nfeat=4096):
    """
    Sample GP via random Fourier series

    Using:
        k(r) = (1/2pi) ∫ S(xi) cos(xi r) dxi

    With implicit symmetry, total spectral mass = 2*sum(A).
    Therefore:
        k(0) = sum(A)/pi
    """

    A = jnp.asarray(A).reshape(-1)
    mu = jnp.asarray(mu).reshape(-1)
    v = jnp.asarray(v).reshape(-1)

    # kernel variance
    k0 = jnp.sum(A) / jnp.pi

    # mixture weights
    p = A / jnp.sum(A)

    key_i, key_s, key_z, key_b = jax.random.split(key, 4)

    # choose mixture component
    idx = jax.random.categorical(key_i, jnp.log(p), shape=(nfeat,))

    # choose symmetric sign ±
    sign = jax.random.bernoulli(key_s, 0.5, shape=(nfeat,))
    sign = jnp.where(sign, 1.0, -1.0)

    # sample frequency
    z = jax.random.normal(key_z, shape=(nfeat,))
    xi = sign * mu[idx] + jnp.sqrt(v[idx]) * z

    # random phase
    b = jax.random.uniform(
        key_b, shape=(nfeat,), minval=0.0, maxval=2.0 * jnp.pi
    )

    # random Fourier features
    t = jnp.asarray(t).reshape(-1)
    Phi = jnp.cos(xi[:, None] * t[None, :] + b[:, None])

    f = jnp.sqrt(2.0 * k0 / nfeat) * jnp.sum(Phi, axis=0)
    return f


if __name__ == "__main__":
    from prism.matern import SGMMatern

    key = jax.random.PRNGKey(123)

    kernel = SGMMatern(J=32, nu=1.5, variance=4.78, lengthscale=1.23)

    t = jnp.linspace(-3.0, 3.0, 1024)

    # --- direct GP samples
    key_gp, key_rff = jax.random.split(key)
    y_gp = _sample_from_kernel(key_gp, kernel, t, nsamp=3)

    # --- spectral samples
    A, mu, v = kernel.compute_sgm()
    keys = jax.random.split(key_rff, 3)

    y_spec = jnp.stack(
        [_sample_from_spectrum(k, A, mu, v, t, nfeat=8192) for k in keys],
        axis=0,
    )

    # --- plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), tight_layout=True)

    for i in range(3):
        axes[0].plot(t, y_gp[i], alpha=0.7)
    axes[0].set_title("Direct GP samples")

    for i in range(3):
        axes[1].plot(t, y_spec[i], alpha=0.7)
    axes[1].set_title("Spectrum (RFF) samples")

    plt.show()

    # sanity check: variance
    k00_kernel = float(kernel(jnp.array([0.0]), jnp.array([0.0])))
    k00_spec = float(jnp.sum(A) / jnp.pi)

    print("k(0) kernel:", k00_kernel)
    print("k(0) from spectrum:", k00_spec)
