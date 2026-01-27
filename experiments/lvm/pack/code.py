# %%
# parameters, export
M = 32  # Number of PRISM basis functions
Q = 3  # Latent dimensionality of qBGPLVM
iteration = 1
seed = 997512


# %%
from functools import partial
from itertools import combinations

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
from gpjax.dataset import Dataset
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.variational_families import CollapsedVariationalGaussian

from lvm.bgplvm import BayesianGPLVM
from lvm.plots import (
    pair_plots_oq,
    plot_cluster_means_in_data_space,
    plot_cluster_samples_in_data_space,
    plot_logl_histogram,
    sample_latent_gmm_pointwise,
    single_plot_oq,
    xd_gmm_plots,
)
from lvm.qgpvlm import loglikelihood_on_test, make_qgpvlm
from prism.pack import NormalizedPACK
from prism.svi import (
    do_prism,
    offdiag_energy_fraction,
    optimize_restarts_scan,
    pick_best,
    svi_basis,
)
from surrogate.prism import get_test_data, get_train_data
from utils import nats_to_ban, time_this
from utils.constants import NOISE_FLOOR_POWER
from utils.jax import pca_reduce, vk

jax.config.update("jax_enable_checks", False)

master_key = jax.random.key(seed)

# %%
# Number of independent waveforms to process train/test
N_TRAIN = 3000
N_TEST = 500

# %%
X, y, oq = get_train_data(n=N_TRAIN)
train_data = Dataset(X=X, y=y)

_, WIDTH_TRAIN = X.shape
n_eff = int(np.sum(~np.isnan(X), axis=1).mean())

print("Number of training waveforms:", N_TRAIN)
print("Average number samples per waveform:", n_eff)
print("Padding width (max waveform length):", WIDTH_TRAIN)


# %%
##############################################################
# STAGE 1: PRISM (COLLAPSED SVI)
# Learn a global basis for the variably sized data
# which then defines a map for the latent space of the BGPLVM
# Secret sauce: "batching" complete waveforms via masking
# ELBO factorizes over independent waveforms
##############################################################
batch_size = 256
num_iters = 100  # 5000 #FIXME
lr = 1e-3


def collapsed_svi(key=vk(), d=1, J=1, M=M):
    Z = jax.random.uniform(key, shape=(M, 1), minval=0.0, maxval=1.0)

    kernel = NormalizedPACK(d=d, J=J)
    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=WIDTH_TRAIN)
    posterior = prior * likelihood

    return CollapsedVariationalGaussian(posterior=posterior, inducing_inputs=Z)


master_key, subkey = jax.random.split(master_key)

from prism.svi import optimize as optimize_svi

with time_this() as svi_timer:
    qsvi, history = optimize_svi(
        subkey, collapsed_svi, train_data, lr, batch_size, num_iters
    )

# %%
px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
).show()

print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)


# %%
# Define and inspect the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


tau_test = jnp.linspace(0, 2, 500)
Psi_test = jax.vmap(psi)(tau_test)  # test indices

eps = jax.random.normal(vk(), shape=(M, 5))
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
mu_eps, Sigma_eps = do_prism(qsvi, train_data)

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
X_var_init = np.ones((N_TRAIN, Q))


def bayesian_gplvm(key=vk(), num_inducing_bgplvm=32, Q=Q):
    lengthscale = jnp.ones((Q,))
    kernel = gpx.kernels.RBF(lengthscale=lengthscale)

    permutation = jax.random.permutation(key, X_mean_init.shape[0])
    Z = X_mean_init[permutation[:num_inducing_bgplvm]]

    return BayesianGPLVM(kernel, X_mu=X_mean_init, X_var=X_var_init, Z=Z)


# Can get trapped early so restarts are needed here (no batching so no noise; restarts just init positions of inducing inputs)

num_iters = 100  # 10_000 # FIXME
lr = 1e-3
num_restarts = 2

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
X_test, Y_test, log_prob_u = get_test_data(n=N_TEST, offset=N_TRAIN)

# Map through PRISM
Psi_test = np.array(jax.vmap(jax.vmap(psi))(X_test))
mask_test = ~np.isnan(X_test)

f_list = []
Psi_list = []

for mask, Y, Psi in zip(mask_test, Y_test, Psi_test):
    f_list.append(Y[mask])
    Psi_list.append(Psi[mask])

# %%


def process_K(K):
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

    plot_cluster_samples_in_data_space(vk(), qgp, tau_test, nsamples=6).show()

    #########################################################
    # STAGE 5: Evaluate surrogate GMM likelihood on test set
    # p(f | tau) = sum_k pi_k N(f | Psi(tau) mu_k, Psi Sigma_k Psi^T + sigma_obs^2 I)
    #########################################################
    log_prob_gmm = loglikelihood_on_test(
        qgp,
        f_list=f_list,
        Psi_list=Psi_list,
        obs_std=qsvi.posterior.likelihood.obs_stddev,
        noise_floor=np.sqrt(NOISE_FLOOR_POWER),
    )

    plot_logl_histogram(
        log_prob_gmm,
        log_prob_u,
        n_eff,
        K,
    ).show()

    mean_gmm_loglikelihood = log_prob_gmm.mean()
    mean_lf_loglikelihood = log_prob_u.mean()

    D_KL = mean_lf_loglikelihood - mean_gmm_loglikelihood

    print(
        f"[K={K}] Average log likelihood per sample (LF model):",
        mean_lf_loglikelihood / n_eff,
    )
    print(
        f"[K={K}] Average log likelihood per sample (GMM model)",
        mean_gmm_loglikelihood / n_eff,
    )

    print(f"[K={K}] D_KL (nats):", D_KL)
    print(f"[K={K}] D_KL (bans):", nats_to_ban(D_KL))
    print(f"[K={K}] D_KL (bans/sample):", nats_to_ban(D_KL) / n_eff)

    return {
        "K": K,
        "mean_gmm_loglikelihood": mean_gmm_loglikelihood,
        "mean_lf_loglikelihood": mean_lf_loglikelihood,
        "D_KL_bans_per_sample": nats_to_ban(D_KL) / n_eff,
    }


# %%
# export
results = [process_K(K=k) for k in [1, 2, 4, 8, 16, 32, 64]]
