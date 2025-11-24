# %%
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from pack.spectral_factors import build_pack_factors
from utils.jax import vk


def sample_fourier_coeffs(key, d, omega_vals, t1, t2, m_max, n_quad=128):
    """
    Sample complex Fourier coeffs c_k (k=1..W) for degree d.

    Returns:
        c : shape (W,), complex
    """
    # B: (M_trim, W)
    _, B = build_pack_factors(d, omega_vals, t1, t2, m_max, n_quad)
    M, W = B.shape

    # z ~ CN(0, I_M)
    key_re, key_im = jax.random.split(key)
    z_re = jax.random.normal(key_re, (M,))
    z_im = jax.random.normal(key_im, (M,))
    z = (z_re + 1j * z_im) / jnp.sqrt(2.0)

    # c = B^H z
    c = jnp.conj(B).T @ z  # (W,)
    return c


def reconstruct_periodic_sample(c, omega_vals, t_grid):
    """
    Reconstruct a real-valued periodic sample u'_T(t)
    from positive-frequency coeffs c_k with cov K_ff.

    Args:
        c          : (W,), complex
        omega_vals : (W,), rad / time_units
        t_grid     : (N_t,), same time units as omega_vals^-1

    Returns:
        u_t : (N_t,), real
    """
    # shape (N_t, W)
    phase = jnp.exp(1j * omega_vals[None, :] * t_grid[:, None])
    # 2 * Re sum_k c_k e^{i omega_k t}
    u_t_complex = jnp.sum(c[None, :] * phase, axis=1)
    u_t = 2.0 * jnp.real(u_t_complex)
    return u_t


# your existing setup
t1 = -5
t2 = 5


T = 20.0  # ms
fs = 16000.0  # Hz, not directly used here
F0 = 1000.0 / T  # Hz equivalent
num_harmonics = int(np.floor((fs / F0) / 2)) * 100

f_vals = jnp.arange(1, num_harmonics + 1) / T  # cycles / ms
omega_vals = 2.0 * jnp.pi * f_vals  # rad / ms

# sample coeffs and synthesize
d = 0
m_max = 100

c = sample_fourier_coeffs(vk(), d, omega_vals, t1, t2, m_max)

# time grid over a few periods [0,T]
num_periods = 2

N_t = int(T / 1000 * fs) * num_periods
t = jnp.linspace(-T, T, N_t, endpoint=False)  # ms

du = (1 / T) * reconstruct_periodic_sample(c, omega_vals, t)
dt = t[1] - t[0]


u = jnp.cumsum(du) * dt

plt.plot(t, du, label="u'_T(t)")
plt.plot(t, u, label="u_T(t)")
plt.xlabel("time (ms)")
plt.legend()
plt.title("Sampled periodic function and its integral")

# soft grey vlines on t1 and t2
plt.axvline(-t1, color="grey", linestyle="--", alpha=0.5)
plt.axvline(-t2, color="grey", linestyle="--", alpha=0.5)

# integrate over [t1, t2] and see if we hit zero
# something is wrong with t1, t2 and minus signs
where = (t >= -t2) & (t <= -t1)
jnp.sum(du[where] * dt)

plt.plot(t, where)

# https://chatgpt.com/g/g-p-68f9d6b4a46c81919b645b342ba50e41-ongoing/c/69236ac4-94c0-8331-af6e-2dc3edb7dcd8
