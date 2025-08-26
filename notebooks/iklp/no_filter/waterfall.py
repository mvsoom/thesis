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
from iklp.mercer_op import sample_parts
from iklp.metrics import (
    StateMetrics,
    compute_metrics_power_distribution,
    compute_power_distibution,
)
from iklp.run import print_progress, vi_run_criterion
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

# WARNING: need to jit any function used in on_metrics() callback, otherwise trigers recompilation at every iteration
compute_expectations = jax.jit(compute_expectations)
compute_metrics_power_distribution = jax.jit(compute_metrics_power_distribution)

collected_metrics = []

def on_metrics(metrics: StateMetrics):
    print_progress(metrics)

    global collected_metrics
    collected_metrics.append(metrics)


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
power = jnp.mean(x**2)
print("power(x)/(nu_w + nu_e) = ", power / (z.nu_w + z.nu_e))

# Show sampled timeseries x
plt.figure()
plt.plot(x)
plt.xlabel("x")
plt.title("Sampled x")
plt.show()


# Show power waterfall plot
plt.figure()

# This plots on plt via on_metrics() function
state, metrics = vi_run_criterion(vk(), x, h, callback=on_metrics)

for i, ms in enumerate(collected_metrics):
    power_distribution = compute_metrics_power_distribution(ms)
    plt.plot(
        power_distribution,
        linewidth=1.0,
        color=CMAP(i / len(collected_metrics)),
    )


inferred_power_distribution = compute_metrics_power_distribution(metrics)

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
plt.legend(loc="best")
plt.show()

# Calculate performance scores
score = np.exp(
    scipy.stats.entropy(true_power_distribution, inferred_power_distribution)
)

print(
    "Score(DKL) of inferred power distribution (lower is better, 1.0 is perfect): ",
    score,
)

wasserstein = scipy.stats.wasserstein_distance(
    np.arange(I + 1),
    np.arange(I + 1),
    true_power_distribution,
    inferred_power_distribution,
)  # symmetric

print(
    "Score(Wasserstein) between true and inferred power distribution (lower is better, 0.0 is perfect): ",
    wasserstein,
)

# %%

from iklp.state import compute_auxiliaries

aux = compute_auxiliaries(state)

op = aux.Omega

signal, noise = sample_parts(op, vk())

plt.plot(x, label="x")
plt.plot(signal, label="signal")
plt.plot(noise, label="noise")
plt.title("Sampled signal and noise parts | (E(z),)")
plt.legend()

# %%
plt.plot(x, label="x")
plt.plot(metrics.signals.T, label="signal")

noises = x - metrics.signals

plt.title("Sampled signal and noise parts | (E(z), x)")
plt.legend()
# %%
