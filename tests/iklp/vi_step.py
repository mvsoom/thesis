# %%
import jax.numpy as jnp

from iklp import vi
from iklp.hyperparams import random_periodic_kernel_hyperparams
from iklp.state import init_test_stable_state
from utils.jax import vk

h = random_periodic_kernel_hyperparams(vk(), I=20, M=128, rank=8)
state0 = init_test_stable_state(vk(), h)

state_fast = vi.vi_step(state0)
state_debug = vi.vi_step_debug(state0)

for name in [
    "rho_theta",
    "tau_theta",
    "rho_w",
    "tau_w",
    "rho_e",
    "tau_e",
    "delta_a",
]:
    a = getattr(state_fast.xi, name)
    b = getattr(state_debug.xi, name)
    assert jnp.allclose(a, b), f"Mismatch in xi.{name}"

elbo_fast = vi.compute_elbo_bound(state_fast)
elbo_debug = vi.compute_elbo_bound(state_debug)
assert jnp.allclose(elbo_fast, elbo_debug), "Mismatch in ELBO"

print("vi_step matches vi_step_debug")
