# %%
import jax.numpy as jnp
import numpy as np

from gfm.ack import compute_Jd_theta_even


def _coeffs_d0(m_max):
    m = np.arange(0, m_max + 1, dtype=float)
    a = np.zeros_like(m)
    a[0] = np.pi
    odd = np.arange(0, m_max + 1) % 2 == 1
    idx = np.nonzero(odd)[0]
    if idx.size > 0:
        a[idx] = 4.0 / (np.pi * (idx.astype(float) ** 2))
    return a


def _coeffs_d1(m_max):
    m = np.arange(0, m_max + 1, dtype=float)
    a = np.zeros_like(m)
    a[0] = 8.0 / np.pi
    if m_max >= 1:
        a[1] = np.pi / 2.0
    # even m >= 2
    even = np.arange(0, m_max + 1) % 2 == 0
    idx = np.nonzero(even & (m >= 2))[0]
    if idx.size > 0:
        mm = idx.astype(float)
        a[idx] = 8.0 / (np.pi * (mm**2 - 1.0) ** 2)
    return a


def _coeffs_d2(m_max):
    m = np.arange(0, m_max + 1, dtype=float)
    a = np.zeros_like(m)
    a[0] = 2.0 * np.pi
    if m_max >= 2:
        a[2] = np.pi / 2.0
    # odd m
    idx = np.nonzero(np.arange(0, m_max + 1) % 2 == 1)[0]
    if idx.size > 0:
        mm = idx.astype(float)
        a[idx] = 128.0 / (np.pi * mm**2 * (mm**2 - 4.0) ** 2)
    return a


def _coeffs_d3(m_max):
    m = np.arange(0, m_max + 1, dtype=float)
    a = np.zeros_like(m)
    a[0] = 256.0 / (3.0 * np.pi)
    if m_max >= 1:
        a[1] = 27.0 * np.pi / 4.0
    if m_max >= 3:
        a[3] = 3.0 * np.pi / 4.0
    even = np.arange(0, m_max + 1) % 2 == 0
    idx = np.nonzero(even)[0]
    if idx.size > 0:
        mm = idx.astype(float)
        # avoid division by zero for m=0,2? we already set m=0 separately,
        # m=2 is safe: (4-1)*(4-9) != 0
        a[idx] = 6912.0 / (np.pi * (mm**2 - 1.0) ** 2 * (mm**2 - 9.0) ** 2)
        # re-instate the hand-set special cases (to be safe numerically)
        a[0] = 256.0 / (3.0 * np.pi)
        if m_max >= 1:
            a[1] = 27.0 * np.pi / 4.0
        if m_max >= 3:
            a[3] = 3.0 * np.pi / 4.0
    return a


def _build_j_coeffs(d, m_max):
    if d == 0:
        return _coeffs_d0(m_max)
    if d == 1:
        return _coeffs_d1(m_max)
    if d == 2:
        return _coeffs_d2(m_max)
    if d == 3:
        return _coeffs_d3(m_max)
    raise ValueError("d must be 0, 1, 2 or 3")


def cosine_coeffs(d, m_max):
    """Cosine-series coefficients a_m^{(d)} for J_d^ext on [-pi,pi]"""
    return _build_j_coeffs(d, m_max)  # (m_max+1,)


def complex_coeffs(d, m_max):
    """
    Complex Fourier coefficients c_m^{(d)} with

        J_d^ext(theta) = sum_{m in Z} c_m^{(d)} e^{i m theta},

    for m >= 0. Here c_m = a_m / 2, and c_{-m} = c_m.
    """
    a = 0.5 * cosine_coeffs(d, m_max)
    c = np.concat([a[::-1][:-1], a])
    return c  # (2*m_max+1,)


def reconstruct_jd_ext(d, theta, m_max):
    """
    Reconstruct J_d^ext(theta) from its complex Fourier coefficients
    up to order m_max.
    """
    c = complex_coeffs(d, m_max)
    exps = jnp.exp(1j * jnp.arange(-m_max, m_max + 1)[:, None] * theta[None, :])
    recon = jnp.sum(c[:, None] * exps, axis=0)
    return recon


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = 1
    m_max = 500

    theta = jnp.linspace(-jnp.pi, jnp.pi, 1000)
    recon = reconstruct_jd_ext(d, theta, m_max)
    true = compute_Jd_theta_even(d, theta)

    plt.plot(theta, recon.real, label="real part")
    plt.plot(theta, recon.imag, label="imag part")
    plt.plot(theta, true, "--", label="true")

    plt.title(f"Reconstruction of J_{d}^ext with m_max={m_max}")
    plt.xlabel("theta")
    plt.ylabel("J_d^ext(theta)")
    plt.legend()
    plt.show()

    # show error
    plt.plot(theta, jnp.abs(recon.real - true))
    plt.title(f"Reconstruction error of J_{d}^ext with m_max={m_max}")
    plt.xlabel("theta")
    plt.ylabel("|recon - true|")
    plt.show()
