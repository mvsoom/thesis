# %% tags=["parameters", "export"]

prism = "iteration~0_M~64_J~16_kernelname~pack:1".replace(
    "~", chr(61)
)  # avoid papermill bug with "=" in parameter values
Q = 6  # Latent dimensionality of BGPLVM
K = 3  # Number of GMM components in latent space of BGPLVM
seed = 2455473317

# %%
import jax

jax.config.update("jax_enable_x64", False)

# %%
import os
from functools import partial
from itertools import combinations

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
from gpjax.dataset import Dataset

from egifa.data import get_data
from lvm.bgplvm import BayesianGPLVM
from lvm.plots import (
    oq_sensitivity_spearman,
    pair_plots_oq,
    plot_cluster_means_in_data_space,
    plot_cluster_samples_in_data_space,
    plot_oq_vs_loglik,
    sample_latent_gmm_pointwise,
    single_plot_oq,
    xd_gmm_plots,
)
from lvm.qgpvlm import make_qgpvlm, surrogate_mixture_log_evidence_on_test
from lvm.xdgmm import fit_xdgmm
from prism.svi import (
    do_prism,
    make_whitener,
    offdiag_energy_fraction,
    optimize_restarts_scan,
    pick_best,
    svi_basis,
)
from utils import dump_egg, load_egg, time_this
from utils.jax import JDTYPE, pca_reduce, vk

master_key = jax.random.key(seed)

# %%
egg = f"prism-vff/egifa/periodic/runs/{prism}.ipynb"
payload = load_egg(egg)
qsvi = payload["qsvi"]

# %%
# Number of independent waveforms to process train/test
N_TRAIN = 2500
N_TEST = 700

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
oq_train = oq[:N_TRAIN]
oq_test = oq[N_TRAIN : N_TRAIN + N_TEST]
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
# Embed the training data via the learned SVI basis (PRISM)
# This is the learned SVI basis space
mu_eps, Sigma_eps = do_prism(qsvi, train_data, batch_size=128)

# %%
# Now prepare to do dimensionality reduction in the learned SVI basis space
# Secret sauce: noisy data via *diagonal* covariances only
# So do a global whitening transform to get near-diagonal matrices for our modified BGPLVM algorithm
whiten, unwhiten = make_whitener(mu_eps, Sigma_eps)

mu_eps_whitened, Sigma_eps_whitened = whiten(mu_eps, Sigma_eps)
diag_eps_whitened = jnp.diagonal(Sigma_eps_whitened, axis1=1, axis2=2)

offdiag = offdiag_energy_fraction(Sigma_eps_whitened)
print("Whitened offdiag energy fraction:", offdiag)

# extract diagonal only
diag_eps_whitened = jnp.diagonal(
    Sigma_eps_whitened, axis1=1, axis2=2
)  # (N, num_inducing_svi)

# just a hack to get means and vars to model via Dataset
dataset_bgplvm = Dataset(X=diag_eps_whitened, y=mu_eps_whitened)

# %%
# Initialize means via PCA, rest is inited randomly
X_mean_init = pca_reduce(mu_eps_whitened, Q)


def bayesian_gplvm(key=vk(), num_inducing_bgplvm=64, Q=Q, jitter=1e-2):
    k1, k2, k3 = jax.random.split(key, 3)

    lengthscale = jax.random.lognormal(k1, shape=(Q,))
    kernel = gpx.kernels.RBF(lengthscale=lengthscale)

    permutation = jax.random.permutation(k2, X_mean_init.shape[0])
    Z = X_mean_init[permutation[:num_inducing_bgplvm]]

    X_var_init = jax.random.lognormal(k3, shape=(mu_eps_whitened.shape[0], Q))

    return BayesianGPLVM(
        kernel, X_mu=X_mean_init, X_var=X_var_init, Z=Z, jitter=jitter
    )


# Try different inits for some variaty, ELBO curves are very stable
num_iters = 4_000
lr = 5e-2
num_restarts = 5

master_key, subkey = jax.random.split(master_key)

from lvm.bgplvm import optimize as optimize_bgplvm

optimize_bgplvm = partial(
    optimize_bgplvm,
    model=bayesian_gplvm,
    dataset=dataset_bgplvm,
    lr=lr,
    num_iters=num_iters,
)

with time_this() as lvm_timer:
    states, elbo_histories = optimize_restarts_scan(
        optimize_bgplvm, num_restarts, subkey
    )

qlvm, history = pick_best(states, elbo_histories, bayesian_gplvm())

# %%
px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
).show()

# %% tags=["export"]
lvm_walltime = lvm_timer.walltime
lvm_obs_std = np.sqrt(qlvm.sigma2)

inverse_lengthscale = 1.0 / np.array(qlvm.kernel.lengthscale)

print("Learned noise std:", lvm_obs_std)
print("Average data std:", np.std(dataset_bgplvm.y, axis=0).mean())
print("Inverse lengthscales:", inverse_lengthscale)

# %%
px.bar(
    x=np.arange(Q),
    y=inverse_lengthscale,
    title="Inverse lengthscales by latent dimension",
    labels={"x": "Latent dimension", "y": "Inverse lengthscale"},
).show()

print("Inferred sqrt(variance) of random point:")
print(np.sqrt(qlvm.X_var[0, :]))

# %%
top3 = np.argsort(-inverse_lengthscale)[:3]

pairs = list(combinations(top3, 2))

showdensity = False
showscatter = True

if Q >= 2:
    pair_plots_oq(qlvm, pairs, showdensity, showscatter, oq_train).show()

if Q >= 3:
    single_plot_oq(qlvm, top3, oq_train).show()

# %%
# Do GMM in latent space of BGPLVM
master_key, subkey = jax.random.split(master_key)

m = qlvm.X_mu
S = jax.vmap(jnp.diag)(qlvm.X_var)

gmm, history = fit_xdgmm(m, S, K, verbose=True, n_iter=500)

for fig in xd_gmm_plots(gmm, history, qlvm, top3):
    fig.show()

# Draw samples from the latent GMM and push **pointwise** through the BGPLVM map to data space
# works for any grid, any resolution, any duration
plvm = qlvm.build_posterior(dataset_bgplvm.y)


def psi(t):
    return svi_basis(qsvi, t)


tau_test = jnp.linspace(0, 3, 1024)

sample_latent_gmm_pointwise(gmm, plvm, psi, tau_test, unwhiten).show()

# %%
# Push GMM components via linearized BGPLVM map to get a mixture surrogate model

qgp = make_qgpvlm(gmm, plvm, psi, whiten, unwhiten)

plot_cluster_means_in_data_space(qgp, tau_test).show()

plot_cluster_samples_in_data_space(subkey, qgp, tau_test, nsamples=3).show()


# %%
# Evaluate on test set

neff = np.sum(~np.isnan(test_data.y), axis=1)
log_prob_gmm = surrogate_mixture_log_evidence_on_test(qgp, qsvi, test_data)
lp = log_prob_gmm / neff

# %%
# Check correlation between OQ and surrogate log likelihood on test set

plot_oq_vs_loglik(oq_test, lp).show()
spearman = oq_sensitivity_spearman(oq_test, log_prob_gmm)

# %% tags=["export"]
mean_loglike_test = np.mean(lp)
std_loglike_test = np.std(lp)
oq_sensitivity = spearman["oq_sensitivity"]
oq_sensitivity_p = spearman["oq_sensitivity_p"]

# %%
print(
    f"Average log likelihood per effective data point on test set: {mean_loglike_test} +/- {std_loglike_test}"
)

print("OQ sensitivity (Spearman):", spearman["oq_sensitivity"])

# %%
payload = {
    "qsvi": qsvi,
    "qgp": qgp,
}

dump_egg(payload, os.getenv("EXPERIMENT_NOTEBOOK_REL"))
