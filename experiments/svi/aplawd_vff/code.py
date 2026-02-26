# %%
# parameters, export
M = 16  # Number of inducing frequencies (per real/imaginary part, so total 2M)
seed = 3164879


# %%
import os
from functools import partial

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
from gpjax.dataset import Dataset
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.parameters import Parameter

from aplawd.data import get_data_periods, get_whitener
from prism.matern import SGMMatern
from prism.spectral import init_Z_inverse_ecdf_from_psd
from prism.svi import (
    optimize_restarts_scan,
    pick_best,
    svi_basis,
)
from prism.t_svi import t_SGMCollapsedVariationalGaussian
from utils import dump_egg, time_this
from utils.jax import vk

master_key = jax.random.key(seed)


# %%
# Number of independent waveforms to process train/test
N_TRAIN = 5_000
N_TEST = 1_000


# %%
width = 320  # cutoff at 99% quantile
X, y = get_data_periods(width=width)
X = jnp.array(X, dtype=jnp.float64)
y = jnp.array(y, dtype=jnp.float64)

# take care of input density
X = X - jnp.nanmean(X, axis=1)[:, None]
input_density_std = jnp.nanmean(jnp.nanstd(X, axis=1))

y = jnp.log10(y)

whiten_y, unwhiten_y = get_whitener(y)
y = whiten_y(y)

train_data = Dataset(X=X[:N_TRAIN], y=y[:N_TRAIN])
test_data = Dataset(
    X=X[N_TRAIN : N_TRAIN + N_TEST], y=y[N_TRAIN : N_TRAIN + N_TEST]
)

_, WIDTH_TRAIN = X.shape
n_eff = int(np.sum(~np.isnan(X), axis=1).mean())

print("Number of training waveforms:", N_TRAIN)
print("Average number samples per waveform:", n_eff)
print("Padding width (max waveform length):", WIDTH_TRAIN)
print("Input density X mean, std:", float(jnp.nanmean(X)), input_density_std)

# %%
# Compute data periodogram
from matplotlib import pyplot as plt
from scipy.signal import periodogram

pg_idx = jnp.sum(~jnp.isnan(train_data.y), axis=1) == 50
pg_y = train_data.y[pg_idx][:, :50]

dtau = 1.0

freqs, Pxx = periodogram(
    pg_y, fs=1 / dtau, detrend=False, window="cosine", scaling="spectrum"
)

Pxx = jnp.mean(Pxx, axis=0)  # average over waveforms

# jax sample frequencies from Pxx
subkey, master_key = jax.random.split(master_key)
Z = init_Z_inverse_ecdf_from_psd(subkey, M, freqs, Pxx)

# plot with sampled frequencies
plt.figure(figsize=(8, 4))
plt.plot(freqs, Pxx, label="Power Spectral Density")
plt.scatter(
    Z, Pxx[jnp.searchsorted(freqs, Z)], color="red", label="Sampled Frequencies"
)


plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.title("Data Periodogram with Sampled Frequencies")
plt.legend()

# %%
##############################################################
# STAGE 1: t-PRISM (COLLAPSED SVI)
# Learn a global basis for the variably sized data
# which then defines a map for the latent space of the BGPLVM
# Secret sauce: "batching" complete waveforms via masking
# ELBO factorizes over independent waveforms
# Secret sauce 2: robust to spike noise
##############################################################
batch_size = 256
num_iters = (
    5000  # 800 suffices if we init lengthscale to 10, but this also works
)
lr = 1e-2
jitter = 1e-4
num_restarts = 1


def trainable(path, v):
    if not isinstance(v, Parameter):
        return False
    # path is usually a tuple of names; make this robust
    leaf = path[-1] if isinstance(path, (tuple, list)) and path else str(path)
    # FIXME: TEST LEARNING VARIANCE
    if leaf == "variance":
        return False
    return True


def t_collapsed_svi(key=vk(), M=M, nu=1, num_inner=3):
    Z = init_Z_inverse_ecdf_from_psd(key, M, freqs, Pxx)

    kernel = SGMMatern(nu=2.5, lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=WIDTH_TRAIN, obs_stddev=0.1)

    posterior = prior * likelihood

    return t_SGMCollapsedVariationalGaussian(
        posterior=posterior,
        inducing_inputs=Z,
        nu=nu,
        num_inner=num_inner,
        jitter=jitter,
        sigma_w=input_density_std,
    )


master_key, subkey = jax.random.split(master_key)

from prism.svi import optimize as optimize_svi

optimize_svi = partial(
    optimize_svi,
    model=t_collapsed_svi,
    dataset=train_data,
    lr=lr,
    batch_size=batch_size,
    num_iters=num_iters,
    trainable=trainable,
)

with time_this() as svi_timer:
    states, elbo_histories = optimize_restarts_scan(
        optimize_svi, num_restarts, subkey
    )

qsvi, history = pick_best(states, elbo_histories, t_collapsed_svi())

px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
).show()


print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)
print("Learned lengthscales:", qsvi.posterior.prior.kernel.lengthscale)
print("Learned variance:", qsvi.posterior.prior.kernel.variance)

# %%
# FIXME
plt.figure(figsize=(8, 4))
plt.plot(freqs, Pxx, label="Power Spectral Density")
plt.scatter(
    Z, Pxx[jnp.searchsorted(freqs, Z)], color="red", label="Sampled Frequencies"
)


plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.title("Data Periodogram with Sampled Frequencies")
plt.legend()
plt.scatter(qsvi.inducing_inputs, jnp.zeros(M), color="blue")


# %%
# Define and inspect the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


tau_test = jnp.linspace(-n_eff, n_eff, 1000)
Psi_test = jax.vmap(psi)(tau_test)  # test indices

master_key, subkey = jax.random.split(master_key)

eps = jax.random.normal(subkey, shape=(qsvi.num_inducing, 5))
y = Psi_test @ eps

px.line(y).update_traces(x=tau_test).update_layout(
    xaxis_title="tau",
    yaxis_title="u'(t)",
    title="Prior samples of learned latent function distribution",
).show()
# This is a prior draw from the learned RKHS subspace, not data-like yet.
# It answers: What does a typical GP draw look like under the learned kernel?
# expected to look generic and smooth

px.line(Psi_test).update_traces(x=tau_test).update_layout(
    xaxis_title="tau",
    yaxis_title="psi_m(t)",
    title="Learned basis functions psi_m(t)",
).show()


# %%
from prism.t_svi import do_t_prism

mu_eps, Sigma_eps, w = do_t_prism(qsvi, train_data)


# %%
# Can we reconstruct waveforms from the SVI latent space?
from prism.svi import reconstruct_waveforms

test_indices = jnp.array([1, 10, 25, 50])

reconstruct_waveforms(mu_eps, qsvi, train_data, test_indices, tau_test).show()


# %%
# Compute normalized log likelihood of test data under model via MC
from prism.t_svi import t_log_evidence_is_on_test

master_key, mc_key = jax.random.split(master_key)

neff = np.sum(~np.isnan(test_data.y), axis=1)
logp = t_log_evidence_is_on_test(qsvi, test_data, mc_key)

x = logp / neff
x_mean = np.mean(x)
x_std = np.std(x)

print(
    "Average log likelihood per effective data point on test set:",
    x_mean,
    "+/-",
    x_std,
)


# %%
# Can we reconstruct test waveforms?
mu_eps, Sigma_eps, w = do_t_prism(qsvi, test_data)

reconstruct_waveforms(
    mu_eps, qsvi, test_data, test_indices, tau_test, weights=w
).show()


# %%
# Calculate local null model: kernel replaced by white noise, everything else (including SVI approx) same
from prism.svi import as_null_model

master_key, subkey = jax.random.split(master_key)

qsvi_null = as_null_model(qsvi)

logp_null = t_log_evidence_is_on_test(
    qsvi_null, test_data, mc_key
)  # reuse same key: reduces variance

x_null = logp_null / neff
x_null_mean = np.mean(x_null)
x_null_std = np.std(x_null)

print(
    "Average log likelihood per effective data point on test set under NULL:",
    x_null_mean,
    "+/-",
    x_null_std,
)

# %%
payload = {
    "whiten": whiten_y,
    "unwhiten": unwhiten_y,
    "qsvi": qsvi,
}

dump_egg(payload, os.getenv("EXPERIMENT_NOTEBOOK_REL"))


# %%
# export
svi_walltime = svi_timer.walltime
svi_obs_std = float(qsvi.posterior.likelihood.obs_stddev)
svi_lengthscale = float(qsvi.posterior.prior.kernel.lengthscale)
svi_alpha = float(qsvi.posterior.prior.kernel.alpha)

mean_loglike_test = x_mean
std_loglike_test = x_std
mean_loglike_test_null = x_null_mean
std_loglike_test_null = x_null_std
