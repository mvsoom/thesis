# %%
# parameters, export
M = 128  # Number of PRISM basis functions
Q = 9  # Latent dimensionality of qBGPLVM
iteration = 1
seed = 2455473317
d = 1
am = "rbf"


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
from gpjax.kernels import RBF, ProductKernel, RationalQuadratic
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.parameters import NonNegativeReal, Parameter
from gpjax.variational_families import CollapsedVariationalGaussian

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
from lvm.qgpvlm import loglikelihood_on_test, make_qgpvlm
from prism.pack import NormalizedPACK
from prism.svi import (
    init_Z_inverse_ecdf,
    offdiag_energy_fraction,
    optimize_restarts_scan,
    pick_best,
    svi_basis,
)
from utils import dump_egg, time_this
from utils.constants import NOISE_FLOOR_POWER
from utils.jax import pca_reduce, vk

master_key = jax.random.key(seed)


# %%
# Number of independent waveforms to process train/test
N_TRAIN = 2500
N_TEST = 500
N_HELD = 210

WIDTH = 8192


# %%
X, y, meta = get_data(n=N_TRAIN, width=WIDTH, with_metadata=True)
X = jnp.array(X, dtype=jnp.float64)
y = jnp.array(y, dtype=jnp.float64)
train_data = Dataset(X=X, y=y)

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
##############################################################
# STAGE 1: PRISM (COLLAPSED SVI)
# Learn a global basis for the variably sized data
# which then defines a map for the latent space of the BGPLVM
# Secret sauce: "batching" complete waveforms via masking
# ELBO factorizes over independent waveforms
##############################################################
batch_size = 32
num_iters = 10_000
lr = 1e-3
jitter = 1e-4


def trainable(path, v):
    if not isinstance(v, Parameter):
        return False
    # path is usually a tuple of names; make this robust
    leaf = path[-1] if isinstance(path, (tuple, list)) and path else str(path)
    if leaf == "sigma_a":
        return False
    return True


def collapsed_svi(key=vk(), d=d, J=1, M=M):
    """Memory cost is O(B M W) where W == WIDTH"""
    Z = init_Z_inverse_ecdf(key, M, X)

    if am == "rbf":
        modulation = RBF()
    elif am == "rationalquadratic":
        modulation = RationalQuadratic(variance=1.0, alpha=NonNegativeReal(1.0))
    else:
        raise ValueError(f"Unknown amplitude modulation kernel: {am}")

    kernel = ProductKernel(
        [
            modulation,
            NormalizedPACK(d=d, J=J, sigma_a=1.0),
        ]
    )

    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=WIDTH)
    posterior = prior * likelihood

    return CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=Z, jitter=jitter
    )


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
        trainable=trainable,
    )


# %%
px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
).show()

print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)
print(
    "Learned AM lengthscale:",
    qsvi.posterior.prior.kernel.kernels[0].lengthscale,
)


# %%
# Define and inspect the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


tau_test = jnp.linspace(0, 7, 512)
Psi_test = jax.vmap(psi)(tau_test)  # test indices

master_key, subkey = jax.random.split(master_key)

eps = jax.random.normal(subkey, shape=(M, 5))
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
# Embed the training data via the learned SVI basis (PRISM)
from prism.svi import do_prism_scan

mu_eps, Sigma_eps = do_prism_scan(qsvi, train_data)


# %%
# Can we reconstruct waveforms from the SVI latent space?
from prism.svi import reconstruct_waveforms

test_indices = jnp.array([10, 100, 250, 500])

reconstruct_waveforms(mu_eps, qsvi, train_data, test_indices, tau_test).show()


# %%
#########################################################
# STAGE 2: B-GP-LVM
# Dimensionality reduction in the learned SVI basis space
# Secret sauce: noisy data via diagonal covariances only
#########################################################

# Global whitening transform to get near-diagonal matrices for our modified BGPLVM algorithm
from prism.svi import make_whitener

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
# Initialize via PCA
X_mean_init = pca_reduce(mu_eps_whitened, Q)
X_var_init = np.ones((mu_eps_whitened.shape[0], Q))


def bayesian_gplvm(key=vk(), num_inducing_bgplvm=32, Q=Q, jitter=1e-4):
    lengthscale = jnp.ones((Q,))
    kernel = gpx.kernels.RBF(lengthscale=lengthscale)

    permutation = jax.random.permutation(key, X_mean_init.shape[0])
    Z = X_mean_init[permutation[:num_inducing_bgplvm]]

    return BayesianGPLVM(
        kernel, X_mu=X_mean_init, X_var=X_var_init, Z=Z, jitter=jitter
    )


# Can get trapped early so restarts are needed here (no batching so no noise; restarts just init positions of inducing inputs)
# Because of OOM and time pressure we keep restarts at the meta "iteration" level

num_iters = 15_000
lr = 5e-2
num_restarts = 1

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


# %%
# export
svi_walltime = svi_timer.walltime
svi_obs_std = float(qsvi.posterior.likelihood.obs_stddev)
svi_am_lengthscale = float(qsvi.posterior.prior.kernel.kernels[0].lengthscale)

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
    pair_plots_oq(qlvm, pairs, showdensity, showscatter, oq).show()

if Q >= 3:
    single_plot_oq(qlvm, top3, oq).show()


# %%
#########################################################
# STAGE 3: GMM in latent space of BGPLVM
# Learn a density mode via local full-covariance Gaussians
# Secret sauce: XD-GMM handles input uncertainties
# Secret sauce #2: background component handles outliers
#########################################################
from lvm.xdgmm import fit_xdgmm

# Load test data for STAGE 5
X_test, Y_test, meta_test = get_data(
    n=N_TEST, offset=N_TRAIN, width=WIDTH, with_metadata=True
)

oq_test = np.array([np.mean(m["oq"]) for m in meta_test])

with jax.default_device(jax.devices("cpu")[0]):
    # Map through PRISM
    Psi_test = np.array(jax.vmap(jax.vmap(psi))(X_test))
    mask_test = ~np.isnan(X_test)

    f_list = []
    Psi_list = []

    for mask, Y, Psi in zip(mask_test, Y_test, Psi_test):
        f_list.append(Y[mask])
        Psi_list.append(Psi[mask])


# %%
def process_K(K, key=vk()):
    print("**************************************")
    print(f"qGPLVM with K={K} components")
    print("**************************************")

    m = qlvm.X_mu
    S = jax.vmap(jnp.diag)(qlvm.X_var)

    gmm, history = fit_xdgmm(m, S, K, verbose=True, n_iter=500)

    for fig in xd_gmm_plots(gmm, history, qlvm, top3):
        fig.show()

    # Draw samples from the latent GMM and push **pointwise** through the BGPLVM map to data space
    # works for any grid, any resolution, any duration
    plvm = qlvm.build_posterior(dataset_bgplvm.y)

    sample_latent_gmm_pointwise(gmm, plvm, psi, tau_test, unwhiten).show()

    #########################################################
    # STAGE 4: Push GMM components via linearized BGPLVM map
    # to get extremely low rank GMM in data space; each component
    # defines a low-rank GP learned from data
    #########################################################
    qgp = make_qgpvlm(gmm, plvm, psi, whiten, unwhiten)

    plot_cluster_means_in_data_space(qgp, tau_test).show()

    plot_cluster_samples_in_data_space(key, qgp, tau_test, nsamples=9).show()

    #########################################################
    # STAGE 5: Evaluate surrogate GMM likelihood on test set
    # p(f | tau) = sum_k pi_k N(f | Psi(tau) mu_k, Psi Sigma_k Psi^T + sigma_obs^2 I)
    #########################################################
    neff = np.sum(~np.isnan(Y_test), axis=1)

    log_prob_gmm = loglikelihood_on_test(
        qgp,
        f_list=f_list,
        Psi_list=Psi_list,
        obs_std=qsvi.posterior.likelihood.obs_stddev,
        noise_floor=np.sqrt(NOISE_FLOOR_POWER),
    )

    lp = log_prob_gmm / neff
    mean_loglike_test = np.mean(lp)
    std_loglike_test = np.std(lp)

    plot_oq_vs_loglik(oq_test, lp).show()
    spearman = oq_sensitivity_spearman(oq_test, log_prob_gmm)

    print(
        f"[K={K}] Average log likelihood per effective data point on test set: {mean_loglike_test} +/- {std_loglike_test}"
    )

    print(
        f"[K={K}] OQ sensitivity (Spearman):",
        spearman["oq_sensitivity"],
    )

    return {
        "qgp": qgp,
        "K": K,
        "mean_loglike_test": mean_loglike_test,
        "std_loglike_test": std_loglike_test,
        "oq_sensitivity": spearman["oq_sensitivity"],
        "oq_sensitivity_p": spearman["oq_sensitivity_p"],
    }


def dump_models(results):
    payload = {
        "qsvi": qsvi,
        "qgp": {r["K"]: r.pop("qgp") for r in results},
    }
    dump_egg(payload, os.getenv("EXPERIMENT_NOTEBOOK_REL"))
    return results


KS = [1, 2, 4, 8, 16]


# %%
# export
results = dump_models(
    [
        process_K(K, subkey)
        for K, subkey in zip(KS, jax.random.split(master_key, len(KS)))
    ]
)



