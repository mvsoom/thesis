# %%
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from tinygp.gp import GaussianProcess
from tinygp.kernels import ExpSquared

from gp.blr import blr_from_mercer
from gp.periodic import SPACK
from utils.jax import vk

t1, t2 = -3.0, 4.0
d = 0

T = 2.0  # ms
fs = 16000.0  # Hz
num_periods = 6

N = int((T / 1000) * fs) * num_periods
t, dt = jnp.linspace(
    -T * num_periods / 2, T * num_periods / 2, N * 10, retstep=True
)

F0 = 1000.0 / T  # Hz
num_harmonics = int(np.floor((fs / F0) / 2))
f_vals = jnp.arange(1, num_harmonics + 1) / T  # cycles/ms
omega_vals = 2.0 * jnp.pi * f_vals  # rad/ms

k = SPACK(d, T, num_harmonics, t1, t2)

gp = blr_from_mercer(k, t)

# use faster Mercer sampling
du = gp.sample(vk())
u = jnp.cumsum(du) * dt

# plot
plt.plot(t, du, label="u'_T(t)")
plt.plot(t, u, label="u_T(t)")
plt.legend()
plt.title("Sampled periodic function and its integral")

# grey markers for open phase [t1,t2]
plt.axvline(t1, color="grey", ls="--", alpha=0.5)
plt.axvline(t2, color="grey", ls="--", alpha=0.5)

plt.show()

# check integral over [t1,t2] is approx zero
where = (t >= t1) & (t <= t2)
print("∫ u'_T = ", jnp.sum(du[where] * dt))

# %%
gp.log_probability(du)

# %%
# QPACK proof of concept
k = SPACK(d, T, num_harmonics, t1, t2) * ExpSquared(scale=5 * T)

gp = GaussianProcess(k, t)

# sampling is slow (does not use Mercer features)
du = gp.sample(vk())
u = jnp.cumsum(du) * dt

plt.plot(t, du, label="u'_T(t)")
plt.plot(t, u, label="u_T(t)")
plt.legend()
plt.title("Sampled periodic function and its integral")

plt.axvline(t1, color="grey", ls="--", alpha=0.5)
plt.axvline(t2, color="grey", ls="--", alpha=0.5)

# %%
