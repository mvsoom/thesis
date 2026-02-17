# %%
import jax

from iklp import vi
from iklp.hyperparams import random_periodic_kernel_hyperparams
from iklp.state import init_test_stable_state
from iklp.vi import compute_elbo_bound
from utils.jax import vk

jax.config.update("jax_debug_nans", True)


# Small test run
h = random_periodic_kernel_hyperparams(vk(), I=20, M=128)
state = init_test_stable_state(vk(), h)

state = vi.update_delta_a(state)
print("Initial ELBO:", compute_elbo_bound(state))