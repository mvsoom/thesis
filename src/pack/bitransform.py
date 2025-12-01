"""Calculate the SPACK bitransform k_tilde(omega1, omega2)"""
# %%

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from gfm.ack import STACK
from pack import zonal

N_QUAD = 256
LEGGAUSS_TABLE = np.polynomial.legendre.leggauss(N_QUAD)


def make_legendre_quadrature(t1, t2):
    """Gauss-Legendre nodes/weights on [t1,t2]"""
    nodes_np, weights_np = LEGGAUSS_TABLE
    x = 0.5 * (t2 - t1) * nodes_np + 0.5 * (t2 + t1)
    w = 0.5 * (t2 - t1) * weights_np
    return jnp.asarray(x), jnp.asarray(w)


def compute_I(m, omega, t1, t2):
    """Compute I_m(omega)

        I_m(omega) = ∫ (1 + x^2)^(-1) exp(i m arctan x - i omega x) dx

    Note: we use Legendre-Gauss quadrature for simplicity, but Filon methods are more accurate (due to oscillatory integrand) given same amount of nodes.
    """
    x_nodes, w_nodes = make_legendre_quadrature(t1, t2)

    theta = jnp.arctan(x_nodes)
    base = w_nodes / (1.0 + x_nodes**2)

    phase = jnp.exp(1j * (m * theta - omega * x_nodes))
    return jnp.sum(base * phase, axis=-1)  # ()


def compute_boundary_term(m, d, theta, omega):
    sec = 1.0 / jnp.cos(theta)
    return (
        (-1.0)
        / (1j * omega)
        * sec**d
        * jnp.exp(1j * m * theta - 1j * omega * jnp.tan(theta))
    )  # ()


def compute_H_from_I(I_vector, omega, D, M, t1, t2):
    """Compute H(d, m; omega) from recurrence on I

    NOTE: the recurrence leaves some entries of H undefined when |m| < d (ie col < row).
    These are just padded with zeros and don't influence downstream calculations of the coefficients. [Before these were NaNs and we trimmed them but this is cleaner and equally fast.]
    """
    # Allocate table to be filled by recurrence
    size = 2 * M + 1
    assert size == len(I_vector)
    m = jnp.arange(-M, M + 1)
    H_table = jnp.zeros((D + 1, size), dtype=I_vector.dtype)

    # Set up boundary term differences
    theta1 = jnp.arctan(t1)
    theta2 = jnp.arctan(t2)

    def boundary_diff(m, d):
        return compute_boundary_term(
            m, d, theta2, omega
        ) - compute_boundary_term(m, d, theta1, omega)

    boundary_diff = jax.vmap(boundary_diff, in_axes=(0, None))

    # Calculate d = 0 "up" from I_vector
    diff0 = boundary_diff(m, 0)
    H0 = diff0 + (m / omega) * I_vector
    H_table = H_table.at[0].set(H0)

    # Continue going "up" for d >= 1
    for d in range(1, D + 1):
        idx = jnp.arange(d, size - d)
        m_sub = m[idx]

        diff = boundary_diff(m_sub, d)

        H_prev = H_table[d - 1]
        H_p = H_prev[idx + 1]
        H_m = H_prev[idx - 1]

        recur = 0.5 / omega * ((m_sub - d) * H_p + (m_sub + d) * H_m)
        H_table = H_table.at[d, idx].set(diff + recur)

    return H_table  # (D+1, 2*M+1)


def compute_H_from_quad(omega, d, m, t1, t2, tol=1e-12):
    """Reference integral for debugging"""

    def integrand_real(x):
        return (1.0 + x * x) ** (d / 2.0) * np.cos(m * np.arctan(x) - omega * x)

    def integrand_imag(x):
        return (1.0 + x * x) ** (d / 2.0) * np.sin(m * np.arctan(x) - omega * x)

    re, _ = quad(integrand_real, t1, t2, epsabs=tol, epsrel=tol)
    im, _ = quad(integrand_imag, t1, t2, epsabs=tol, epsrel=tol)
    return re + 1j * im


def sort_interval(t1, t2):
    swap = t2 < t1
    t1n = jnp.where(swap, t2, t1)
    t2n = jnp.where(swap, t1, t2)
    sign = jnp.where(swap, -1.0, 1.0)
    return t1n, t2n, sign


def compute_H_table(omega, D, M, t1, t2):
    """Build the full H(d,m; omega) table for d = 0..D, m = -M..M"""
    t1, t2, sign = sort_interval(t1, t2)

    def compute_I_vector(m):
        return compute_I(m, omega, t1, t2)

    m = jnp.arange(-M, M + 1)
    I_vector = jax.vmap(compute_I_vector)(m)  # shape (2*M+1,)
    H_table = compute_H_from_I(I_vector, omega, D, M, t1, t2)

    return (
        m,  # (2*M+1,)
        sign * H_table,  # (D+1, 2*M+1)
    )


if __name__ == "__main__":
    M = 100

    def test_one(omega, d, m, t1=-3.0, t2=4.0):
        # compute via fast recurrence
        m_vals, H = compute_H_table(omega, d, M, t1, t2)
        approx = H[d, m + M]

        # reference
        exact = compute_H_from_quad(omega, d, m, t1, t2)
        rel = abs(approx - exact) / max(1e-16, abs(exact))

        print(
            f"rel.err={rel:.3e}   "
            f"[omega={omega:4.1f}, d={d}, m={m:4d}]   "
            f"recurrence={approx:.6e}   quad={exact:.6e}"
        )

    # a handful of representative test points
    test_cases = [
        (0.5, 0, 20),
        (0.5, 2, -12),
        (1.0, 3, 5),
        (3.0, 2, -31),
        (10.0, 1, 47),
        (17.0, 3, -59),
    ]

    for omega, d, m in test_cases:
        test_one(omega, d, m)  # all ok


# %%
def compute_phi_tilde(omega, d, M, t1, t2):
    """Compute the factors of the Mercer series of k_tilde

    Note: the weights are baked in the phi factors.

        k_tilde(omega1, omega2) = <phi_tilde(omega1), phi_tilde(omega2)>

    where this is the conjugate dot product.
    """
    m, H_table = compute_H_table(omega, d, M, t1, t2)
    H = H_table[d]  # (2*M+1,)

    # Pair the entries of H with the correct Fourier coefficients from the degree-d zonal expansion
    # NOTE: these "complex_coeffs" are actually real-valued due to evenness of the zonal kernel
    c = zonal.complex_coeffs(d, M)  # (2*M+1,)
    assert len(c) == len(m)
    del m  # these match the indices of c

    phi_tilde = jnp.sqrt(c / (2 * jnp.pi)) * H
    return phi_tilde  # (2*M+1,)


def compute_k_tilde(omega1, omega2, d, M, t1, t2):
    """Compute the hermitian kernel k_tilde(omega1, omega2)"""
    phi_tilde1 = compute_phi_tilde(omega1, d, M, t1, t2)
    phi_tilde2 = compute_phi_tilde(omega2, d, M, t1, t2)
    k_tilde = jnp.vdot(phi_tilde2, phi_tilde1)  # conjugate dot product
    return k_tilde  # ()


def compute_k_tilde_quad(omega1, omega2, d, t1, t2):
    x, w = make_legendre_quadrature(t1, t2)

    W = w[:, None] * w[None, :]
    K = STACK(d)(x, x)

    phase = jnp.exp(-1j * omega1 * x[:, None] + 1j * omega2 * x[None, :])
    integrand = K * phase

    return jnp.sum(W * integrand)


if __name__ == "__main__":
    M = 128  # DONT INCREASE BLINDLY: SEE BELOW
    t1, t2 = -3.0, 4.0

    def test_pair(omega1, omega2, d):
        approx = compute_k_tilde(omega1, omega2, d, M, t1, t2)
        exact = compute_k_tilde_quad(omega1, omega2, d, t1, t2)
        rel = abs(approx - exact) / max(1e-16, abs(exact))

        print(
            f"rel.err={rel:.3e}   "
            f"[w1={omega1:4.1f}, w2={omega2:4.1f}, d={d}]   "
            f"approx={approx:.6e}   exact={exact:.6e}"
        )

    test_cases = [
        (0.5, 1.0, 0),
        (0.5, 0.9, 2),
        (1.0, 3.0, 1),
        (3.0, 10.0, 2),
        (10.0, 7.3, 3),
        (34.5, 12.3, 0),
        (223.4, 98.7, 1),
    ]

    for w1, w2, d in test_cases:
        test_pair(w1, w2, d)

# %%

"""
NOTE ABOUT SPECTRAL TRUNCATION (THE CHOICE OF M)

The recurrence for H(d,m;ω) is numerically stable only for moderate m.
Beyond a few hundred modes the upward recursion (in d) amplifies noise
at a rate that eventually dominates the true signal. This happens even
though the underlying Mercer series (the zonal J_d^ext expansion) has
excellent m^{-2d-2} decay. The instability is entirely a property of
the H-recurrence itself, not of the kernel.

What we observe in practice:

    • For |m| <= ~100-150 the recurrence matches high-accuracy
      quadrature extremely well for all relevant d and ω.

    • For larger m the recurrence over-resolves the boundary term,
      and roundoff is propagated and amplified across levels d=0→1→2→3.
      Past M≈200 the computed H(d,m;ω) can blow up by orders of magnitude.

    • Increasing M never improves accuracy of k_tilde; in fact it
      monotonically worsens it once the unstable tail begins. This
      is visible even with very high quadrature order.

Why small ω are the hardest:

    The I_m(ω) integrals are least oscillatory when ω is small,
    so cancellation is weakest and the recurrence becomes the
    dominant numerical mechanism. High-ω cases actually behave 
    *better* because the quadrature is naturally stabilised by 
    the oscillatory phase.

Operationally:

    • The domain of interest (speech acoustics) gives harmonics
      up to roughly ω ~ 2π f with f in the few-hundred-Hz range.
      For fs=16kHz and typical glottal cycles we only ever see
      ω ≲ 200.

    • For ω <= 200 and d <= 3, choosing M = 128 is sufficient for
      machine-precision accuracy when checked against direct
      quadrature of the kernel (TACK(d)).

    • Going beyond M≈128 buys nothing: the zonal coefficients
      already decay rapidly (m^{-2}, m^{-4}, m^{-6}, m^{-8}), so
      the extra modes contribute negligibly to the true kernel
      but *massively* to numerical instability of the recurrence.

Summary:

    USE M = 128 (or smaller). DEFINITELY do not increase M.
    Accuracy improves downwards with M until ~128 and then
    collapses if pushed further. This is the stable operating
    regime for the H-recurrence and is entirely adequate for
    the omega range used in glottal modelling.

    # This issue can be solved by using normalized TACK kernels or just
    # always doing quadrature, but for now we just cap M at 128.
"""


# %%
def build_k_tilde_matrix(omega_vals, d, M, t1, t2):
    """Return K(w_i, w_j) for all omega_vals, shape (W,W).

    Uses:
        phi_tilde(w) : (2*M+1,)
        K(w_i,w_j) = <phi_tilde(w_i), phi_tilde(w_j)>
    """

    # phi(w_i) stacked: Φ has shape (W, 2*M+1)
    phi = jax.vmap(lambda w: compute_phi_tilde(w, d, M, t1, t2))(
        omega_vals
    )  # (W, 2*M+1)

    # inner vmap: fix phi_i, vary phi_j
    def k_row(phi_i):
        return jax.vmap(lambda phi_j: jnp.vdot(phi_j, phi_i))(phi)
        # returns row: (W,)

    # outer vmap sweeps φ_i
    K = jax.vmap(k_row)(phi)  # (W, W)

    return K


if __name__ == "__main__":
    fs = 16000.0  # Hz
    T = 5.0  # msec period
    f0 = 1000.0 / T  # Hz
    num_harmonics = int(fs // (2 * f0))

    omega_vals = 2 * jnp.pi * (jnp.arange(1, num_harmonics + 1) * f0)

    D = 0  # or 1,2,3
    M = 128  # chosen stable spectral truncation
    t1, t2 = -3.0, 4.0

    K = build_k_tilde_matrix(omega_vals, D, M, t1, t2)
    print(K.shape)  # (num_harmonics, num_harmonics)
