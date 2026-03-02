# %% tags=["parameters", "export"]
kernelname = "pack:0"
M = 64  # Number of PRISM basis functions
J = 8
iteration = 1
seed = 2455473317

# %%
import jax

jax.config.update("jax_enable_x64", False)

# %%
import os

import jax.numpy as jnp
import numpy as np
import plotly.express as px
from gpjax.dataset import Dataset
from gpjax.gps import Prior
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero

from ack.pack import PACK
from commonvoice.data import get_data
from prism.harmonic import (
    SHMCollapsedVariationalGaussian,
    SHMPeriodic,
    SHMPeriodicFFT,
    harmonic_null_model,
)
from prism.svi import (
    do_prism,
    reconstruct_waveforms,
    surrogate_log_evidence_on_test,
    svi_basis,
)
from utils import dump_egg, time_this
from utils.jax import JDTYPE, vk

master_key = jax.random.key(seed)

# %%
# Number of independent waveforms to process train/test
N_TRAIN = 15000
N_TEST = 3000

WIDTH = 8192

# %%
X, y, meta = get_data(width=WIDTH, with_metadata=True)

X = jnp.array(X, dtype=JDTYPE)
y = jnp.array(y, dtype=JDTYPE)

train_data = Dataset(X=X[:N_TRAIN], y=y[:N_TRAIN])
test_data = Dataset(
    X=X[N_TRAIN : N_TRAIN + N_TEST], y=y[N_TRAIN : N_TRAIN + N_TEST]
)

oq = np.array([np.mean(m["oq"]) for m in meta])
n_eff = int(np.sum(~np.isnan(X), axis=1).mean())

print("Number of training waveforms:", N_TRAIN)
print("Average number samples per waveform:", n_eff)
print("Padding width (max waveform length):", WIDTH)

occupancy = (~np.isnan(y)).sum() / np.prod(y.shape)
print(
    f"Data occupancy after padding at WIDTH={WIDTH}: {100 * (1 - occupancy):.2f}%"
)

qs = np.quantile(np.nanmax(X, axis=1), [0.25, 0.5, 0.75, 0.95])
print(f"Quantiles of covered cycles at WIDTH={WIDTH}:", qs)


# %%
def init_kernel(key=vk()):
    if "pack" in kernelname:
        d = kernelname.split(":")[1]
        lz = jax.random.lognormal(key, shape=(J + 2,))
        variance, weights = lz[0], lz[1:]
        return SHMPeriodicFFT(
            PACK(d=d, J=J, period=1.0, variance=variance, weights=weights),
            num_harmonics=M,
        )
    elif kernelname == "periodic":
        lz = jax.random.lognormal(key, shape=(2,))
        variance, lengthscale = lz[0], lz[1]
        return SHMPeriodic(
            period=1.0,
            variance=variance,
            lengthscale=lengthscale,
            num_harmonics=M,
        )
    else:
        raise ValueError(f"Unknown kernel name: {kernelname}")


def get_lengthscale_or_weights(qsvi):
    if kernelname == "periodic":
        return np.atleast_1d(qsvi.posterior.prior.kernel.lengthscale)
    elif kernelname.startswith("pack"):
        return np.array(qsvi.posterior.prior.kernel.kernel.weights)
    else:
        raise ValueError(f"Unknown kernel name: {kernelname}")


# %%
# Do PRISM-VFF
batch_size = 64
num_iters = 2000
lr = 1e-3
jitter = 1e-4


def collapsed_svi(key=vk()):
    kernel = init_kernel(key)
    prior = Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=WIDTH)
    posterior = prior * likelihood
    return SHMCollapsedVariationalGaussian(posterior=posterior, M=M)


master_key, subkey = jax.random.split(master_key)

from prism.svi import optimize as optimize_svi

with time_this() as svi_timer:
    qsvi, history = optimize_svi(
        subkey,
        collapsed_svi,
        train_data,
        lr,
        batch_size,
        num_iters,
    )

# %%
px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
).show()

print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)
print("Learned kernel variance:", qsvi.posterior.prior.kernel.variance)
print(
    "Learned kernel lengthscale or weights:", get_lengthscale_or_weights(qsvi)
)


# %%
# Define and inspect the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


tau_test = jnp.linspace(0, 3, 512)
Psi_test = jax.vmap(psi)(tau_test)  # test indices

master_key, subkey = jax.random.split(master_key)

eps = jax.random.normal(subkey, shape=(2 * M + 1, 4))
y = Psi_test @ eps

px.line(y).update_traces(x=tau_test).update_layout(
    xaxis_title=r"$\tau$ (cycles)",
    yaxis_title=r"$u'(t)$",
    title="Prior samples of learned latent function distribution",
).show()

px.line(Psi_test).update_traces(x=tau_test).update_layout(
    xaxis_title=r"$\tau$ (cycles)",
    yaxis_title=r"$\psi_m(t)$",
    title=r"Learned basis functions $\psi_m(t)$",
).show()

# %%
# Can we reconstruct waveforms from TRAINING/TEST data?
chosen_indices = jnp.array([10, 100, 250, 500])

for data in [train_data, test_data]:
    chosen_data = Dataset(data.X[chosen_indices], data.y[chosen_indices])
    mu_eps, Sigma_eps = do_prism(qsvi, chosen_data)
    reconstruct_waveforms(
        mu_eps, qsvi, chosen_data, np.arange(len(chosen_indices)), tau_test=None
    ).show()

# %%
# Compute normalized log likelihood of test data under model via MC
neff = np.sum(~np.isnan(test_data.y), axis=1)
logp = surrogate_log_evidence_on_test(qsvi, test_data)

x = logp / neff
x_mean = np.mean(x)
x_std = np.std(x)

print(
    f"Average log likelihood per effective data point on test set: {x_mean:.4f} +/- {x_std:.4f}"
)

# %%
# Calculate test score of a local "harmonic" null model
# This means: keep variance and inducing points, but an essentially flat spectrum
qsvi_null, centered_test_data = harmonic_null_model(qsvi, test_data)

logp_null = surrogate_log_evidence_on_test(qsvi_null, centered_test_data)

x_null = logp_null / neff
x_null_mean = np.mean(x_null)
x_null_std = np.std(x_null)

print(
    f"Average log likelihood per effective data point on test set under NULL: {x_null_mean:.4f} +/- {x_null_std:.4f}"
)

# %%
payload = {
    "qsvi": qsvi,
}

dump_egg(payload, os.getenv("EXPERIMENT_NOTEBOOK_REL"))

# %% tags=["export"]
svi_walltime = svi_timer.walltime
svi_obs_std = float(qsvi.posterior.likelihood.obs_stddev)
svi_lengthscale_or_weights = get_lengthscale_or_weights(qsvi)

mean_loglike_test = x_mean
std_loglike_test = x_std
mean_loglike_test_null = x_null_mean
std_loglike_test_null = x_null_std
