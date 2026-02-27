# %%
import jax
import jax.numpy as jnp
from gpjax.kernels import AbstractKernel
from gpjax.parameters import PositiveReal
from gpjax.variational_families import CollapsedVariationalGaussian

from prism.svi import init_Z_grid


class SGMKernel(AbstractKernel):
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

    NOTE: we have a even window and even spectrum:

        S(xi) = sum_q A_q [N(xi | +mu_q,v_q) + N(xi | -mu_q,v_q)]

    so this function actually returns real Kuu.
    We keep it complex if these conditions change later.

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
    """
    Convert complex Hermitian Kuu (with omega[0]=0)
    into real block form with DC handled separately.

    Input:
        K : (Mtot, Mtot) complex Hermitian
            where Mtot = M + 1
            index 0 = DC frequency

    Output:
        K_real : (1 + 2M, 1 + 2M) real symmetric
    """

    Mtot = K.shape[0]
    assert Mtot >= 1

    M = Mtot - 1  # number of nonzero freqs

    Re = jnp.real(K)
    Im = jnp.imag(K)

    K_dc = Re[0:1, 0:1]  # (1,1)

    Re_rest = Re[1:, 1:]  # (M,M)
    Im_rest = Im[1:, 1:]  # (M,M)

    # Real block for nonzero freqs
    K_rest = jnp.block(
        [
            [Re_rest, -Im_rest],
            [Im_rest, Re_rest],
        ]
    )  # (2M, 2M)

    # Cross term (DC x others)
    Re_cross = Re[1:, 0:1]
    Im_cross = Im[1:, 0:1]

    K_cross_top = jnp.concatenate([Re_cross.T, Im_cross.T], axis=1)

    K_cross = jnp.concatenate([Re_cross, Im_cross], axis=0)

    K_real = jnp.block(
        [
            [K_dc, K_cross_top],
            [K_cross, K_rest],
        ]
    )

    return K_real

def complex_to_real_Kuf(K):
    """
    Convert complex Kuf (with omega[0]=0)
    into real block form.

    Input:
        K : (Mtot, N) complex
            Mtot = M + 1

    Output:
        K_real : (1 + 2M, N) real
    """
    Mtot, N = K.shape
    M = Mtot - 1

    K_dc = jnp.real(K[0:1, :])  # (1, N)

    K_rest = K[1:, :]  # (M, N)

    Re = jnp.real(K_rest)
    Im = jnp.imag(K_rest)

    K_real = jnp.concatenate([K_dc, Re, Im], axis=0)

    return K_real


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

        # inducing_inputs are frequencies (cycles per unit tau)
        self.inducing_inputs = PositiveReal(self.inducing_inputs.value)

    def compute_Kuu(self):
        kernel = self.posterior.prior.kernel
        A, mu, v = kernel.compute_sgm()

        # kernel returns only mu_q >= 0, these formulas enforce symmetry
        Kuu_complex = _sgm_symm_Kuu_complex(
            A=A, mu=mu, v=v, omega=self.omega, sigma_w=self.sigma_w
        )
        Kuu = complex_to_real_Kuu(Kuu_complex)
        return Kuu

    def compute_Kuf(self, t):
        kernel = self.posterior.prior.kernel
        A, mu, v = kernel.compute_sgm()

        # t are tau locations
        Kuf_complex = _sgm_symm_Kuf_complex(
            A=A, mu=mu, v=v, omega=self.omega, tau=t, sigma_w=self.sigma_w
        )
        Kuf = complex_to_real_Kuf(Kuf_complex)
        return Kuf

    @property
    def omega(self):
        freq = jnp.concatenate(
            [jnp.array([0.0]), self.inducing_inputs.squeeze()]
        )  # explicit DC
        return (2.0 * jnp.pi) * freq

    @property
    def num_inducing(self):
        return (
            self.inducing_inputs.shape[0] * 2 + 1
        )  # account for real/imaginary split + DC


def init_Z_inverse_ecdf_from_psd(key, M, freqs, Pxx):
    f = jnp.asarray(freqs)
    w = jnp.asarray(Pxx + 1e-12)  # avoid zeros
    w = w / jnp.sum(w)
    cdf = jnp.cumsum(w)
    u = init_Z_grid(key, M)
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
