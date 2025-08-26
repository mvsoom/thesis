"""Show theta waterfall plot during VI optimization for a (P=0) model"""

# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from matplotlib import pyplot as plt

from iklp.hyperparams import (
    active_components,
    pi_kappa_hyperparameters,
)
from iklp.mercer import psd_svd
from iklp.metrics import (
    compute_power_distibution,
    compute_state_power_distribution,
)
from iklp.run import CriterionState, print_progress, vi_run_criterion
from iklp.state import (
    compute_expectations,
    sample_x_from_z,
    sample_z_from_prior,
)
from utils.jax import maybe32, vk

# %%
# Get some kernels
kernels = [
    gpx.kernels.Matern12(),
    gpx.kernels.Matern32(),
    gpx.kernels.Matern52(),
    gpx.kernels.RBF(),
]

t = jnp.linspace(-3.0, 3.0, num=200).reshape(-1, 1)

I = len(kernels)
K = jnp.stack([k.gram(t).to_dense() for k in kernels], axis=0)
Phi = psd_svd(K)

# %%
# Setup stuff
CMAP = plt.get_cmap("coolwarm")

# WARNING: need to jit any function used in onstate() callback, otherwise trigers recompilation at every iteration
compute_expectations = jax.jit(compute_expectations)
compute_state_power_distribution = jax.jit(compute_state_power_distribution)


def onstate(cs: CriterionState, every=1):
    print_progress(cs)

    i = cs.i
    if i % every == 0:
        power_distribution = compute_state_power_distribution(cs.state)
        plt.plot(power_distribution, linewidth=1.0, color=CMAP(i / 100))


vi_run_criterion = jax.jit(vi_run_criterion, static_argnames=("callback",))

# %%
# Define hyperparameters and sample from prior
alpha = 0.1  # solve_for_alpha(I) => ensure one component dominates
pi = 0.95
kappa = 1.0

h = pi_kappa_hyperparameters(Phi, alpha=maybe32(alpha), pi=pi, kappa=kappa, P=0)
z = sample_z_from_prior(vk(), h)
x = sample_x_from_z(vk(), z, h)

print("nu_w", z.nu_w)
print("nu_e", z.nu_e)
print("sum(theta) = ", z.theta.sum())
print("pitchedness = ", z.nu_w / (z.nu_w + z.nu_e))
print("I_eff =", active_components(z.theta))


# Show sampled timeseries x
plt.figure()
plt.plot(x)
plt.xlabel("x")
plt.title("Sampled x")
plt.show()

power = jnp.mean(x**2)
print("power(x)/(nu_w + nu_e) = ", power / (z.nu_w + z.nu_e))


# Show power waterfall plot
plt.figure()

# This plots on plt via onstate() function
cs = vi_run_criterion(vk(), x, h, callback=onstate)

true_power_distribution = compute_power_distibution(z)

plt.stem(
    true_power_distribution,
    linefmt="r-",
    markerfmt="ro",
    label="true power distribution",
    basefmt=" ",
)

labels = ["noise"] + [str(i) for i in range(I)]

plt.xticks(ticks=np.arange(I + 1), labels=labels)
plt.xlabel("kernel index $i$")
plt.ylabel("relative power")
plt.title("Power distribution waterfall through VI and ground truth")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()


inferred_power_distribution = compute_state_power_distribution(cs.state)

score = np.exp(
    scipy.stats.entropy(true_power_distribution, inferred_power_distribution)
)

print(
    "Score of inferred power distribution (lower is better, 1.0 is perfect): ",
    score,
)
