# %%
import jax

from iklp.periodic import (
    periodic_kernel,
    periodic_kernel_phi,
    periodic_mock_data,
)

f0, K = periodic_kernel(I=10)
f0_2, Phi = periodic_kernel_phi(I=10, batch_size=3, noise_floor_db=-90.0)

assert jax.numpy.allclose(f0, f0_2), "F0 series do not match!"
print("F0 series:", f0)

K_approx = Phi @ jax.numpy.swapaxes(Phi, -1, -2)
err = jax.numpy.max(jax.numpy.abs(K - K_approx))
print("Max absolute reconstruction error:", err)

# sample mock data
f0i, x = periodic_mock_data(jax.random.PRNGKey(1234), f0, Phi)
print("sampled f0:", f0i)
