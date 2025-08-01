# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from iklp.hyperparams import Hyperparams
from iklp.periodic import periodic_kernel_phi, periodic_mock_data
from iklp.state import init_state, init_test_stable_state
from iklp.vi import (
    compute_elbo_bound,
    compute_expectations,
    update_delta_a,
    vi_step,
)
from utils import plot_power_spectrum_db, top_p_indices

update_delta_a = jax.jit(update_delta_a)
vi_step = jax.jit(vi_step)
compute_elbo_bound = jax.jit(compute_elbo_bound)
compute_expectations = jax.jit(compute_expectations)

# %%
f0, Phi = periodic_kernel_phi(I=400, M=2048, fs=16000)

# %%
# Using periodic mock data, there are always one or two very clear resonances in the data, and they are typically recovered well with an AR filter ~= constant
# When P is too high, the model will sometimes explain the resonance as a very low frequency content through a nontrivial AR filter (not caused by accidental high initialization value of low frequency thetas, the model actually seeks this out and sometimes retains the higher frequency hypotheses)
# This is solved by setting P to 1 or any other low value
h = Hyperparams(Phi, P=30)
key = jax.random.PRNGKey(303341)

if False:
    state = init_test_stable_state(key, h, remove_dc=True)
else:
    k1, k2 = jax.random.split(key, 2)
    f0_star, x = periodic_mock_data(k1, f0, Phi)
    state = init_state(k2, x, h)

# %%
# Warmup
update_delta_a(state)
vi_step(state)
compute_elbo_bound(state)
compute_expectations(state)

# %%
fig, ax = plt.subplots()
cmap = plt.get_cmap("coolwarm")

# Updating q(a) = delta(a* - a) as the very first update
# is known to yield better convergence
# as it is initalized to zeroes
state = update_delta_a(state)

score = -jnp.inf
criterion = 0.001
n_iter = 10

for i in range(n_iter):
    E = compute_expectations(state)
    print(f"argmax_i(theta) = {top_p_indices(E.theta)[:5]}...")

    state = vi_step(state)

    color = cmap(i / n_iter)
    ax.plot(f0, E.theta, color=color, alpha=0.8, label=f"iter {i}")

    lastscore = score
    score = compute_elbo_bound(state)

    if i == 0:
        improvement = 1.0
    else:
        improvement = (score - lastscore) / jnp.abs(lastscore)

    print(
        "iteration {}: bound = {:.2f} ({:+.5f} improvement)".format(
            i, score, improvement
        )
    )
    if improvement < 0.0:
        print("Diverged")
        break
    if improvement < criterion:
        print("Converged")
        break
    if jnp.isnan(improvement) and i > 0:
        print("NaN")
        break

ax.set_xlabel("F0")
ax.set_ylabel("E[theta]")
ax.set_yscale("log")
ax.legend()

# %%
plt.plot(E.theta)
plt.axvline(np.argmax(E.theta), color="red", linestyle="--")

# %%
fig, ax = plt.subplots()
plot_power_spectrum_db(state.data.x, fs=16000, ax=ax)

ax.axvline(f0[np.argmax(E.theta)], color="red", linestyle="--", alpha=0.5)
ax.set_xlim(100, 400)

# %%
plt.plot(state.xi.delta_a)

# %%
