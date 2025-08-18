# %%
import jax

jax.config.update("jax_enable_x64", False)

from iklp.hyperparams import random_periodic_kernel_hyperparams

key = jax.random.PRNGKey(0)

h = random_periodic_kernel_hyperparams(key)
print(h)

print("Phi shape:", h.Phi.shape)
print(h.Phi.device, jax.config.jax_enable_x64)
