# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

from iklp.mercer import psd_svd
from iklp.util import _periodic_kernel_batch

key = jax.random.PRNGKey(123)
B, M = 5, 50
Ts = jax.random.normal(key, (B,)) * 5 + 10
K = _periodic_kernel_batch(Ts, M)

Phi = psd_svd(K, noise_floor_db=-60.0)
print("Phi.shape:", Phi.shape)

K_approx = jnp.matmul(Phi, jnp.swapaxes(Phi, -1, -2))
err = jnp.max(jnp.abs(K - K_approx), axis=[-2, -1])
print("reconstruction error:", err)
print("Phi dtype:", Phi.dtype)
