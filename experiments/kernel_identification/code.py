# %% [markdown]
# # Single effective lengthscale
#
# Given a sparse mixture { (k_j, ℓ_j) } with weights w_j = θ_j / Σ_i θ_i, define per-component correlation times
#
# $$
# \tau_j = c_{k_j}\,\ell_j, \quad
# c_{\mathrm{m12}}=1,\; c_{\mathrm{m32}}=\frac{2}{\sqrt{3}},\;
# c_{\mathrm{m52}}=\frac{8}{3\sqrt{5}},\; c_{\mathrm{rbf}}=\sqrt{\frac{\pi}{2}}.
# $$
#
# Use the geometric mean as a single scalar statistic:
#
# $$
# \ell_{\mathrm{eff}} \equiv \exp\!\Big(\sum_j w_j \log \tau_j\Big).
# $$
#
# With M points and m samples per lengthscale: Δt = ℓ_eff / m, T = (M-1)Δt, domain [-L, L] with L = T/2.
#
# # Weighted rank r (shared across kernels; Hilbert on [-L, L])
#
# Fix a spectral tolerance ε (e.g., ε=10^{-2}). For each kernel j, solve S_j(ω_c)/S_j(0)=ε. Closed forms (angular frequency):
#
# $$
# \begin{aligned}
# \text{m12: } & \omega_c = \tfrac{1}{\ell}\sqrt{\varepsilon^{-1}-1},\\
# \text{m32: } & \omega_c = \tfrac{\sqrt{3}}{\ell}\sqrt{\varepsilon^{-1/2}-1},\\
# \text{m52: } & \omega_c = \tfrac{\sqrt{5}}{\ell}\sqrt{\varepsilon^{-1/3}-1},\\
# \text{rbf: } & \omega_c = \sqrt{2\ln(1/\varepsilon)}\,/\,\ell.
# \end{aligned}
# $$
#
# On [-L, L], Laplacian eigenfrequencies are \( s_n = n\pi/(2L) \). Per-kernel mode count:
#
# $$
# r_j = \Big\lceil \frac{2L}{\pi}\,\omega_{c,j} \Big\rceil.
# $$
#
# Aggregate to a single rank with mixture weights and a small safety factor γ (e.g., γ=1.1):
#
# $$
# r = \left\lceil \gamma \sum_j w_j\, r_j \right\rceil
# \quad \text{(or use a weighted quantile if you prefer robustness).}
# $$
#
# # Hilbert kernel construction (for reference)
#
# Use sine basis on [-L, L]:
#
# $$
# \phi_n(x)=\sqrt{\tfrac{1}{L}}\;\sin\!\Big(\tfrac{n\pi}{2L}\,(x+L)\Big),\quad
# s_n=\tfrac{n\pi}{2L},\ n=1,\dots,r,
# $$
#
# with diagonal weights from the base PSD:
#
# $$
# \lambda_n = S\!\big(s_n\big),\qquad
# k(x,x') \approx \sum_{n=1}^r \lambda_n\,\phi_n(x)\,\phi_n(x').
# $$
#

# %%
seed = 0


batch_size = 1
num_metrics_samples = 1

N_kernels = 1
M_features = 128

N_ell = 100
N_data = 128

alpha_scale = 1.0


assert N_kernels <= 4

# %%
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from matplotlib import pyplot as plt

from gp.hilbert import Hilbert
from gp.spectral import ExpSquared, Matern12, Matern32, Matern52
from iklp.hyperparams import (
    active_components,
    pi_kappa_hyperparameters,
    solve_for_alpha,
)
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
master_key = jax.random.PRNGKey(seed)


# %%
def integrated_correlation_time(self):
    """1D stationary kernel characteristic lengthscale"""
    if isinstance(self, Matern12):
        return self.scale * 1.0
    elif isinstance(self, Matern32):
        return self.scale * 2 / jnp.sqrt(3)
    elif isinstance(self, Matern52):
        return self.scale * 8 / (3 * jnp.sqrt(5))
    elif isinstance(self, ExpSquared):
        return self.scale * jnp.sqrt(jnp.pi / 2)
    else:
        raise ValueError("Unknown kernel type")


kernel_types = [
    ExpSquared,
    Matern52,
    Matern32,
    Matern12,
]

for k in kernel_types:
    k.integrated_correlation_time = integrated_correlation_time

ells = jnp.logspace(-1, 1, num=N_ell)

kernels = []
for k in kernel_types[:N_kernels]:
    for ell in ells:
        kernels.append(k(scale=ell))


# %%
# Sample ground true theta to determine which kernels are active, and what is the effective lengthscale
# The latter is used to sample data indices, as they must span a few effective lengthscales to be informative about the kernel mixture
I = len(kernels)

alpha = solve_for_alpha(I) * alpha_scale
pi = 0.95
kappa = 1.0

Phi_dummy = jnp.empty((I, M_features, 0))

h = pi_kappa_hyperparameters(
    Phi_dummy, alpha=maybe32(alpha), pi=pi, kappa=kappa, P=0
)

master_key, key = jax.random.split(master_key)
z_true = sample_z_from_prior(key, h)
theta_true = z_true.theta

# %%
weights = theta_true / theta_true.sum()

integrated_correlation_times = jnp.array(
    [k.integrated_correlation_time() for k in kernels]
)

l_eff = jnp.exp(jnp.sum(weights * jnp.log(integrated_correlation_times)))

plt.plot(theta_true)
plt.xlabel("kernel index $i$")
plt.ylabel("weight $\Theta_i$")
plt.show()

print(f"nu_w        = {z_true.nu_w:.4f}")
print(f"nu_e        = {z_true.nu_e:.4f}")
print(f"sum(theta)  = {z_true.theta.sum():.4f}")
print(f"pitchedness = {(z_true.nu_w / (z_true.nu_w + z_true.nu_e)):.4f}")
print(f"I_eff       = {active_components(z_true.theta):.4f}")
print(f"l_eff       = {l_eff:.4f}")

# %%
# Sample data indices and data themselves according to ground truth priors
t_support = l_eff * 3.0
hilbert_support = t_support * 1.25
t = jnp.linspace(-t_support, t_support, num=N_data)


def compute_phi(k):
    h = Hilbert(k, M=M_features, L=hilbert_support)
    Phi = jax.vmap(h.compute_phi)(t)  # (N_data, M)
    L = h.compute_weights_root()  # (M, M)
    return Phi @ L  # (N_data, M)


Phi = jnp.stack([compute_phi(k) for k in kernels], axis=0)
h = h.replace(Phi=Phi)

master_key, key = jax.random.split(master_key)
x = sample_x_from_z(key, z_true, h)

plt.plot(t, x)
plt.xlabel("t")
plt.title("Sampled data x")
plt.show()

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
alpha = solve_for_alpha(I) * alpha_scale
pi = 0.95
kappa = 1.0

Phi_dummy = jnp.empty((I, M, 0))  # Dummy, not used when P=0

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
