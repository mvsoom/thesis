# %%
from iklp.hyperparams import random_periodic_kernel_hyperparams
from iklp.state import (
    init_test_stable_state,
    sample_x_from_z,
    sample_z_from_prior,
    sample_z_from_q,
)
from utils.jax import vk

h = random_periodic_kernel_hyperparams(vk())


z = sample_z_from_prior(vk(), h)
print("Sample theta mean (prior):", z.theta.mean())

x = sample_x_from_z(vk(), z, h)
print("x mean (prior):", x.mean())

state = init_test_stable_state(vk(), h)

zs = sample_z_from_q(vk(), state.xi, h)
print("Sample theta mean (q):", zs.theta.mean())

xs = sample_x_from_z(vk(), zs, h)
print("x mean (q):", xs.mean())  # stable
