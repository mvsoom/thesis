# %% tags=["parameters", "export"]
carriername = "pack:1"
M = 32  # Number of PRISM basis functions
J = 16
iteration = 1
seed = 2455473317

# %%
import jax

jax.config.update("jax_enable_x64", False)

# %%
import os

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
from gpjax.dataset import Dataset
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero

from ack.pack import PACK
from commonvoice.data import get_data
from prism.harmonic import (
    SGMQuasiPeriodic,
    SHMPeriodic,
    SHMPeriodicFFT,
    quasiharmonic_null_model,
)
from prism.matern import SGMRBF, SGMMatern
from prism.plot import (
    compute_mean_yw_power,
    mean_spectrum,
    plot_average_psd_db,
    plot_basis_functions,
    plot_elbo_history,
    plot_prior_samples,
    plot_spectra_after_vi,
    plot_spectral_initialization,
)
from prism.spectral import SGMCollapsedVariationalGaussian
from prism.svi import (
    do_prism,
    reconstruct_waveforms,
    surrogate_log_evidence_on_test,
    svi_basis,
)
from utils import dump_egg, time_this
from utils.jax import (
    JDTYPE,
    nnx_copy,
    normalize_density,
    quantile_sample,
    vk,
)

master_key = jax.random.key(seed)

# %%
# Number of independent waveforms to process train/test
N_TRAIN = 15000
N_TEST = 1800

WIDTH = 4096

# %%
X, y = get_data(width=WIDTH)

X = jnp.array(X, dtype=JDTYPE)
y = jnp.array(y, dtype=JDTYPE)

# take care of input density: play along with Gaussian input density
X = X - jnp.nanmean(X, axis=1)[:, None]
input_density_std = jnp.nanmean(jnp.nanstd(X, axis=1))

train_data = Dataset(X=X[:N_TRAIN], y=y[:N_TRAIN])
test_data = Dataset(
    X=X[N_TRAIN : N_TRAIN + N_TEST], y=y[N_TRAIN : N_TRAIN + N_TEST]
)

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
# Compute DFT of data
sigma_w = (
    10 * input_density_std
)  # has no influence on empirical spectrum, but maybe for inferring AM lengthscale. EDIT: has huge influence because this modulates all basis functions, so set plenty large

dtau = jnp.nanmean(jnp.abs(jnp.diff(X, axis=1)))
print(f"Average sampling interval dtau: {dtau:.4f}")

freqs = jnp.linspace(0, 0.5 / dtau, 8_000)

average_psd = compute_mean_yw_power(train_data.X, train_data.y, freqs, sigma_w)

plot_average_psd_db(freqs, average_psd).show()


# %%
def init_carrier_kernel(key=vk(), num_harmonics=32):
    if "pack" in carriername:
        d = carriername.split(":")[1]
        lz = jax.random.lognormal(key, shape=(J + 2,))
        variance, weights = lz[0], lz[1:]
        k = SHMPeriodicFFT(
            PACK(d=d, J=J, period=1.0, variance=variance, weights=weights),
            num_harmonics=num_harmonics,
        )
        k.kernel.variance = jnp.asarray(
            1.0
        )  # don't train as this isn't identifiable (carrier variance vs am variance)
        return k
    elif carriername == "periodic":
        lz = jax.random.lognormal(key, shape=(2,))
        variance, lengthscale = lz[0], lz[1]
        k = SHMPeriodic(
            period=1.0,
            variance=variance,
            lengthscale=lengthscale,
            num_harmonics=num_harmonics,
        )
        k.variance = jnp.asarray(
            1.0
        )  # don't train as this isn't identifiable (carrier variance vs am variance)
        return k
    else:
        raise ValueError(f"Unknown kernel name: {carriername}")


def get_carrier_lengthscale_or_weights(qsvi):
    if carriername == "periodic":
        return np.atleast_1d(qsvi.posterior.prior.kernel.carrier.lengthscale)
    elif carriername.startswith("pack"):
        return np.array(qsvi.posterior.prior.kernel.carrier.kernel.weights)
    else:
        raise ValueError(f"Unknown kernel name: {carriername}")


def init_am_kernel(key=vk()):
    lz = jax.random.lognormal(key, shape=(2,))
    return SGMRBF(variance=lz[0], lengthscale=lz[1])


def init_baseline_kernel(key=vk(), J=16):
    lz = jax.random.lognormal(key, shape=(3,))
    return SGMMatern(nu=lz[0], variance=lz[1], lengthscale=lz[2], J=J)


# %%
alpha = 1 / 4


def collapsed_svi(key=vk(), M=M, sigma_w=sigma_w, freqs=freqs, alpha=alpha):
    k1, k2, k3 = jax.random.split(key, 3)

    am = init_am_kernel(k1)
    carrier = init_carrier_kernel(k2)
    baseline = init_baseline_kernel(k3)
    kernel = SGMQuasiPeriodic(am, carrier, baseline)

    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=WIDTH)
    posterior = prior * likelihood

    S = kernel.bochner_spectrum(freqs)
    D = average_psd
    score = S * D
    tempered_score = score**alpha
    tempered_score_norm = normalize_density(freqs, tempered_score)
    samples = quantile_sample(freqs, tempered_score_norm, M)

    inducing_inputs = samples[:, None]

    return SGMCollapsedVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        sigma_w=sigma_w,
        freeze_inducing_inputs=True,
    )


n_samples = 1000
keys = jax.random.split(vk(), n_samples)
S_mean = mean_spectrum(collapsed_svi, keys, freqs)

prior_qsvi = collapsed_svi()

prior_qsvi.chunk = 512  # for block streaming ELBO computation

plot_spectral_initialization(
    freqs=freqs,
    average_psd=average_psd,
    prior_qsvi=prior_qsvi,
    S_mean=S_mean,
    M=M,
    alpha=alpha,
).show()

# %%
from prism.svi import collapsed_elbo_masked_streamed

collapsed_elbo_masked_streamed(
    prior_qsvi, X[0], y[0]
)  # test if inducing points aren't too close


# %%
# Do PRISM-VFF
batch_size = 32
microbatch = None
num_iters = 6000
lr = 1e-3
jitter = 1e-4

master_key, subkey = jax.random.split(master_key)

from prism.svi import optimize as optimize_svi

with time_this() as svi_timer:
    qsvi, history = optimize_svi(
        subkey,
        nnx_copy(prior_qsvi),
        train_data,
        lr,
        batch_size,
        num_iters,
        prior_model=True,
        microbatch=microbatch,
    )

# %%
plot_elbo_history(history).show()

print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)

print("AM lengthscale:", qsvi.posterior.prior.kernel.am.lengthscale)

# %%
# Compare spectra after VI
plot_spectra_after_vi(freqs, average_psd, prior_qsvi, qsvi).show()


# %%
# Define and inspect the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


tau_test = jnp.linspace(0, 3, 512)
Psi_test = jax.vmap(psi)(tau_test)  # test indices

master_key, subkey = jax.random.split(master_key)

eps = jax.random.normal(subkey, shape=(2 * M + 1, 4))
y = Psi_test @ eps

plot_prior_samples(tau_test, y).show()

plot_basis_functions(tau_test, Psi_test).show()

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
qsvi_null, centered_test_data = quasiharmonic_null_model(
    qsvi, test_data, sigma_w=sigma_w
)

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
svi_am_variance = float(qsvi.posterior.prior.kernel.am.variance)
svi_am_lengthscale = float(qsvi.posterior.prior.kernel.am.lengthscale)
svi_baseline_nu = float(qsvi.posterior.prior.kernel.baseline.nu)
svi_baseline_variance = float(qsvi.posterior.prior.kernel.baseline.variance)
svi_baseline_lengthscale = float(
    qsvi.posterior.prior.kernel.baseline.lengthscale
)
svi_lengthscale_or_weights = get_carrier_lengthscale_or_weights(qsvi)

mean_loglike_test = x_mean
std_loglike_test = x_std
mean_loglike_test_null = x_null_mean
std_loglike_test_null = x_null_std
