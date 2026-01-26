# %%
# parameters, export
d = 1
J = 8
num_inducing_svi = 16
latent_dim_bgplvm = 3
iteration = 1
seed = 997562


# %%
from itertools import combinations

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax as ox
import plotly.express as px
from flax import nnx
from gpjax.dataset import Dataset
from gpjax.parameters import Parameter

from prism.bgplvm import BayesianGPLVM
from prism.pack import NormalizedPACK
from prism.svi import (
    batch_collapsed_elbo_masked,
    do_prism,
    get_test_data,
    get_train_data,
    gp_posterior_mean_from_eps,
    latent_pair_density,
    offdiag_energy_fraction,
    pick_best,
    svi_basis,
)
from prism.xdgmm import gmm_data_loglikelihoods
from utils import nats_to_ban, time_this
from utils.constants import NOISE_FLOOR_POWER
from utils.jax import pca_reduce, vk

jax.config.update("jax_enable_checks", False)

master_key = jax.random.key(seed)


# %%
N_TRAIN = 5000

X, y, oq = get_train_data(n=N_TRAIN)
N_TRAIN, WIDTH_TRAIN = (
    X.shape
)  # Number of waveforms in dataset, max waveform length

dataset = Dataset(X=X, y=y)

print("Data shape:", X.shape, y.shape)


# %%
##############################################################
# STAGE 1: PRISM (COLLAPSED SVI)
# Learn a global basis for the variably sized data
# which then defines a map for the latent space of the BGPLVM
# Secret sauce: "batching" complete waveforms via masking
# ELBO factorizes over independent waveforms
##############################################################
def get_variational_svi_model(key=jax.random.PRNGKey(0)):
    Z = jax.random.uniform(
        key, shape=(num_inducing_svi, 1), minval=0.0, maxval=1.0
    )

    kernel = NormalizedPACK(d=d, J=J)
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=WIDTH_TRAIN)

    posterior = prior * likelihood

    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=Z
    )

    return q


def optimize_svi(key):
    key, subkey = jax.random.split(key)

    q = get_variational_svi_model(subkey)

    batch_size = 256
    num_iters = 5000
    lr = 1e-3

    def cost(q, d):
        return -batch_collapsed_elbo_masked(q, d, N_TRAIN)

    opt_posterior, cost_history = gpx.fit(
        model=q,
        objective=cost,
        train_data=dataset,
        optim=ox.adam(learning_rate=lr),
        num_iters=num_iters,
        key=key,
        batch_size=batch_size,
        trainable=Parameter,
    )

    history = -cost_history
    return nnx.state(opt_posterior), history


# Restarts here don't make sense due to noisy objective; we do this on the config level so we test on held out data directly instead
num_restarts = 1

master_key, subkey = jax.random.split(master_key)
subkeys = jax.random.split(subkey, num_restarts)

with time_this() as timer:
    states, histories = jax.vmap(optimize_svi)(subkeys)


# %%
fig = px.line(
    np.array(histories).T,
    title="ELBO runs during training",
    labels={"x": "Iteration", "y": "ELBO"},
)
fig.show()

# %%
# pick best run
qsvi, history = pick_best(states, histories, get_variational_svi_model())

fig = px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
)
fig.show()

print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)


# %%
# Define the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


test_tau = jnp.linspace(0, 2, 500)
Psi = jax.vmap(psi)(test_tau)  # test indices

eps = jax.random.normal(vk(), shape=(num_inducing_svi, 5))
y = Psi @ eps

px.line(y).update_traces(x=test_tau).update_layout(
    xaxis_title="tau",
    yaxis_title="f(t)",
    title="Samples of learned latent function distribution",
).show()
# This is a prior draw from the learned RKHS subspace, not data-like yet.
# It answers: What does a typical GP draw look like under the learned kernel?
# expected to look generic and smooth


# %%
px.line(Psi).update_traces(x=test_tau).update_layout(
    xaxis_title="tau",
    yaxis_title="psi_m(t)",
    title="Learned basis functions psi_m(t)",
).show()

# %%
# Embed the training data via the learned SVI basis (PRISM)
mu_eps, Sigma_eps = do_prism(qsvi, dataset)


# %%
# Now we test if the learned RKHS is rich enough to reconstruct some test waveforms
test_indices = jnp.array([10, 100, 250, 500])

f_means = jax.vmap(lambda eps: gp_posterior_mean_from_eps(qsvi, test_tau, eps))(
    mu_eps[test_indices]
)

plot_rows = []
panel_order = []
for idx, f_mean in zip(test_indices, f_means):
    idx_int = int(idx)
    panel_label = f"test_index={idx_int}"
    panel_order.append(panel_label)

    x_data = np.array(dataset.X[idx_int])
    y_data = np.array(dataset.y[idx_int])
    for x_val, y_val in zip(x_data, y_data):
        plot_rows.append(
            {
                "tau": x_val,
                "value": y_val,
                "series": "Data",
                "panel": panel_label,
            }
        )

    x_pred = np.array(test_tau)
    y_pred = np.array(f_mean)
    for x_val, y_val in zip(x_pred, y_pred):
        plot_rows.append(
            {
                "tau": x_val,
                "value": y_val,
                "series": "Posterior mean",
                "panel": panel_label,
            }
        )

fig = px.line(
    plot_rows,
    x="tau",
    y="value",
    color="series",
    facet_col="panel",
    facet_col_wrap=2,
    category_orders={"panel": panel_order},
    title="Posterior mean vs data (selected test indices)",
    labels={"tau": "tau", "value": "f(t)", "series": ""},
)
fig.show()


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
X_mean_init = pca_reduce(mu_eps_whitened, latent_dim_bgplvm)
X_var_init = np.ones((N_TRAIN, latent_dim_bgplvm))


def get_variational_bgplvm_model(key=jax.random.PRNGKey(0)):
    num_inducing_bgplvm = 50

    lengthscale = jnp.ones((latent_dim_bgplvm,))
    kernel = gpx.kernels.RBF(lengthscale=lengthscale)

    permutation = jax.random.permutation(key, X_mean_init.shape[0])
    Z = X_mean_init[permutation[:num_inducing_bgplvm]]

    q = BayesianGPLVM(
        kernel,
        X_mu=X_mean_init,
        X_var=X_var_init,
        Z=Z,
    )

    return q


def optimize_bgplvm(key):
    q = get_variational_bgplvm_model(key)

    def cost(q, d):
        return -q.elbo(d.y, obs_var_diag=d.X)

    lr = 1e-3
    num_iters = 10_000

    bgplvm, cost_history = gpx.fit(
        model=q,
        objective=cost,
        train_data=dataset_bgplvm,
        optim=ox.adam(learning_rate=lr),
        num_iters=num_iters,
    )

    history = -cost_history
    return nnx.state(bgplvm), history


# Can get trapped early so restarts are needed here (no batching so no noise; restarts just init positions of inducing inputs)
num_restarts = 1  # FIXME: set to 5

master_key, subkey = jax.random.split(master_key)
subkeys = jax.random.split(subkey, num_restarts)


with time_this() as timer:
    # FIXME: do this in experimentation -- prevents OOM but no error bar => change num_restarts too
    states, histories = jax.vmap(optimize_bgplvm)(subkeys)
    # states, histories = jax.lax.map(optimize_bgplvm, subkeys)

walltime = timer.walltime


# %%
px.line(histories.T).update_traces(
    x=np.arange(histories.shape[1])
).update_layout(
    xaxis_title="iteration",
    yaxis_title="ELBO",
    title="ELBO runs during BGPLVM training",
).show()


# %%
# pick best run
qlvm, history = pick_best(states, histories, get_variational_bgplvm_model())

px.line(
    history,
    title="ELBO during training (best run)",
    labels={"x": "Iteration", "y": "ELBO"},
).show()


# %%
noise_std = np.sqrt(qlvm.sigma2)

print("Learned noise std:", noise_std)
print("Average data std:", np.std(dataset_bgplvm.y, axis=0).mean())


# %%
inverse_lengthscale = 1.0 / qlvm.kernel.lengthscale
print(inverse_lengthscale)

# plot inverse lengthscales
px.bar(
    x=np.arange(latent_dim_bgplvm),
    y=inverse_lengthscale,
    title="Inverse lengthscales by latent dimension",
    labels={"x": "Latent dimension", "y": "Inverse lengthscale"},
).show()

# %%
print("Inferred sqrt(variance) of random point:")
print(np.sqrt(qlvm.X_var[0, :]))


# %%
top3 = np.argsort(-inverse_lengthscale)[:3]

pairs = list(combinations(top3, 2))

X_mu = qlvm.X_mu
X_var = qlvm.X_var

showdensity = False
showscatter = True  # False#True

if showdensity:
    for pair in pairs:
        dens, extent = latent_pair_density(X_mu, X_var, pair)

        i, j = pair
        x_vals = np.linspace(extent[0], extent[1], dens.shape[1])
        y_vals = np.linspace(extent[2], extent[3], dens.shape[0])

        fig = px.imshow(
            np.array(dens),
            x=x_vals,
            y=y_vals,
            color_continuous_scale=px.colors.sequential.Gray,
            title=f"Latent pair density (latent {i} vs latent {j})",
            labels={"x": f"latent {i}", "y": f"latent {j}", "color": "density"},
            aspect="auto",
        )

        if showscatter:
            fig.add_scatter(
                x=np.array(X_mu[:, i]),
                y=np.array(X_mu[:, j]),
                mode="markers",
                marker=dict(
                    size=6,
                    color=np.array(oq),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="oq"),
                ),
                name="oq",
            )
            fig.update_traces(showscale=False, selector=dict(type="heatmap"))

        fig.show()
elif showscatter:
    scatter_rows = []
    panel_order = []
    for pair in pairs:
        i, j = pair
        panel_label = f"latent {i} vs latent {j}"
        panel_order.append(panel_label)
        for x_val, y_val, color_val in zip(
            np.array(X_mu[:, i]), np.array(X_mu[:, j]), np.array(oq)
        ):
            scatter_rows.append(
                {
                    "x": x_val,
                    "y": y_val,
                    "oq": color_val,
                    "panel": panel_label,
                }
            )

    fig = px.scatter(
        scatter_rows,
        x="x",
        y="y",
        color="oq",
        facet_col="panel",
        facet_col_wrap=2,
        category_orders={"panel": panel_order},
        color_continuous_scale="Viridis",
        title="Latent pair projections colored by oq",
        labels={"x": "latent value", "y": "latent value", "oq": "oq"},
    )
    fig.show()


# %%
i, j, k = top3

fig = px.scatter_3d(
    x=X_mu[:, i],
    y=X_mu[:, j],
    z=X_mu[:, k],
    color=oq,
    color_continuous_scale="Viridis",
    opacity=0.7,
)

fig.update_traces(marker=dict(size=2))
fig.update_layout(
    scene=dict(
        xaxis_title=f"latent {i}",
        yaxis_title=f"latent {j}",
        zaxis_title=f"latent {k}",
    ),
    title="Latent space (top 3 dims) colored by oq",
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()


# %%
#########################################################
# STAGE 3: GMM in latent space of BGPLVM
# Learn a density mode via local full-covariance Gaussians
# Secret sauce: XD-GMM handles input uncertainties
# Secret sauce #2: background component handles outliers
#########################################################
from prism.xdgmm import e_step, fit_xdgmm

m = qlvm.X_mu
S = jax.vmap(jnp.diag)(qlvm.X_var)

K = 8

params, history, (mu0, cov0) = fit_xdgmm(
    m, S, K, alpha_quantile=1 - 1e-6, verbose=True
)

r, *_ = e_step(m, S, params, mu0, cov0, jitter=1e-6)

fig = px.line(
    y=np.array(history),
    title="XD-GMM log likelihood during EM",
    labels={"x": "Iteration", "y": "Log likelihood"},
)
fig.show()


# %%
# plot cumulative histogram of background responsibilities
r_bg = r[:, 0]
fig = px.histogram(
    x=np.array(r_bg),
    nbins=100,
    cumulative=True,
    histnorm="probability",
    title="Cumulative histogram of background responsibilities",
    labels={
        "x": "Responsibility of background component",
        "y": "Cumulative density",
    },
)
fig.show()

is_outlier = r_bg > 0.95

labels = np.argmax(
    r[:, 1:], axis=1
)  # cluster index 0..K-1 (ignoring background)

labels = np.array(labels)

fig = px.bar(
    x=np.arange(r.shape[1]),
    y=np.array(r.sum(axis=0) / r.sum()),
    title="Component responsibility mass",
    labels={"x": "Component index", "y": "Responsibility mass"},
)
fig.show()


# %%
i, j, k = top3

# Colors: clusters, but gray outliers
plot_color = labels.astype(str)
plot_color = np.where(is_outlier, "outlier", plot_color)

fig = px.scatter_3d(
    x=X_mu[:, i],
    y=X_mu[:, j],
    z=X_mu[:, k],
    color=plot_color,
    opacity=0.7,
)

# Make outliers visually obvious
fig.update_traces(
    selector=dict(name="outlier"),
    marker=dict(size=4, color="black", symbol="x"),
)

fig.update_traces(marker=dict(size=3))

fig.update_layout(
    scene=dict(
        xaxis_title=f"latent {i}",
        yaxis_title=f"latent {j}",
        zaxis_title=f"latent {k}",
    ),
    title="Latent space clusters with outliers",
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()


# %%
import plotly.graph_objects as go
from scipy.stats import chi2

# Use the same top3 dims
i, j, k = top3
dims = (i, j, k)

# 99% chi-square radius in 3D
radius = np.sqrt(chi2.ppf(0.95, 3))

# Color map reused from clusters
unique_labels = np.unique(labels)
palette = px.colors.qualitative.Dark24
color_map = {str(l): palette[l % len(palette)] for l in unique_labels}

fig = go.Figure()

# Optional: faint points for context
fig.add_trace(
    go.Scatter3d(
        x=X_mu[:, i],
        y=X_mu[:, j],
        z=X_mu[:, k],
        mode="markers",
        marker=dict(
            size=2,
            color=[color_map[str(l)] for l in labels],
            opacity=0.12,
        ),
        showlegend=False,
    )
)

# Parametric unit sphere grid
Nu, Nv = 50, 25
u = np.linspace(0, 2 * np.pi, Nu)
v = np.linspace(0, np.pi, Nv)
uu, vv = np.meshgrid(u, v)

# Unit sphere (3, Nv, Nu)
sphere = np.stack(
    [
        np.cos(uu) * np.sin(vv),
        np.sin(uu) * np.sin(vv),
        np.cos(vv),
    ],
    axis=0,
)

# Plot each cluster ellipsoid (skip background)
for k_idx in range(params.mu.shape[0]):
    mu = np.array(params.mu)[k_idx][list(dims)]
    cov = np.array(params.cov)[k_idx][np.ix_(dims, dims)]

    # Eigen-decomposition
    w, V = np.linalg.eigh(cov)

    # Transform unit sphere -> ellipsoid
    A = V @ np.diag(np.sqrt(w)) * radius

    # Apply transform, keeping grid
    ell = mu[:, None, None] + np.einsum("ij,jmn->imn", A, sphere)

    X = ell[0]
    Y = ell[1]
    Z = ell[2]

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            opacity=0.25,
            showscale=False,
            surfacecolor=np.zeros_like(X),
            colorscale=[[0, color_map[str(k_idx)]], [1, color_map[str(k_idx)]]],
            name=f"cluster {k_idx}",
        )
    )

fig.update_layout(
    scene=dict(
        xaxis_title=f"latent {i}",
        yaxis_title=f"latent {j}",
        zaxis_title=f"latent {k}",
    ),
    title="Cluster ellipsoids in latent space",
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()


# %%
plvm = qlvm.build_posterior(dataset_bgplvm.y)

pi_free = np.exp(np.array(params.logits))[1:]
pi_free = pi_free / pi_free.sum()

# %%
k = np.random.choice(len(pi_free), p=pi_free)  # component index 0..K-1

mu_k = np.array(params.mu)[k]
cov_k = np.array(params.cov)[k]

z = np.random.multivariate_normal(mu_k, cov_k)  # (1, latent_dim_bgplvm)

mu_y, diag_y = plvm.predict_f_meanvar_batch(z, z * 0)
Sigma_y = jax.vmap(jnp.diag)(diag_y[0])

mu_eps_sample, mu_Sigma_sample = unwhiten(mu_y, Sigma_y)

Psi = jax.vmap(psi)(test_tau)  # test indices

f_sample = Psi @ mu_eps_sample.squeeze()

fig = px.line(
    x=np.array(test_tau),
    y=np.array(np.cumsum(f_sample)),
    title=f"Sample from GMM cluster {k + 1}",
    labels={"x": "tau", "y": "f(t)"},
)
fig.show()

# works for any grid, any resolution, any duration

# %%
#########################################################
# STAGE 4: Push GMM components via linearized BGPLVM map
# to get extremely low rank GMM in data space; each component
# defines a low-rank GP learned from data
#########################################################

y_pi, y_mu, y_cov = plvm.forward_x_gmm(pi_free, params.mu, params.cov)


eps_mu, eps_cov = unwhiten(y_mu, y_cov)

# %%
# Plot cluster means in data space
Psi = jax.vmap(psi)(test_tau)  # test indices
means = Psi @ eps_mu.T

px.line(means).update_traces(x=test_tau).update_layout(
    xaxis_title="tau",
    yaxis_title="u'(tau)",
    title="Learned means of GMM components in data space",
).show()

# %%
k = np.random.choice(len(y_pi), p=y_pi)  # component index 0..K-1

eps_sample = jax.random.multivariate_normal(vk(), eps_mu[k], eps_cov[k])

f_sample = Psi @ eps_sample

px.line(
    x=np.array(test_tau),
    y=np.array(f_sample),
    title=f"Sample from GMM cluster {k + 1}",
    labels={"x": "tau", "y": "u'(t)"},
).show()

# %%
#########################################################
# STAGE 5: Evaluate surrogate GMM likelihood on test set
# p(f | tau) = sum_k pi_k N(f | Psi(tau) mu_k, Psi Sigma_k Psi^T + sigma_obs^2 I)
#########################################################

# Calculate likelihood of test data
N_TEST = 1000

X_test, Y_test, log_prob_u = get_test_data(n=N_TEST, offset=N_TRAIN)

Psi_test = np.array(jax.vmap(jax.vmap(psi))(X_test))
mask_test = ~np.isnan(X_test)

# %%
f_list = []
Psi_list = []

ii = 0
for mask, Y, Psi in zip(mask_test, Y_test, Psi_test):
    f_list.append(Y[mask])
    Psi_list.append(Psi[mask])

# %%
log_prob_gmm = gmm_data_loglikelihoods(
    f_list=f_list,
    Psi_list=Psi_list,
    pi=y_pi,
    mu_k=eps_mu,
    Sigma_k=eps_cov,
    obs_std=qsvi.posterior.likelihood.obs_stddev,
    noise_floor=np.sqrt(NOISE_FLOOR_POWER),
)

mean_gmm_loglikelihood = log_prob_gmm.mean()
mean_lf_loglikelihood = log_prob_u.mean()

D_KL = mean_lf_loglikelihood - mean_gmm_loglikelihood
effective_samples = mask_test.sum(axis=1).mean()

print(
    "Average log likelihood per sample (LF model):",
    mean_lf_loglikelihood / effective_samples,
)
print(
    "Average log likelihood per sample (GMM model)",
    mean_gmm_loglikelihood / effective_samples,
)

print("D_KL (nats):", D_KL)
print("D_KL (bans):", nats_to_ban(D_KL))
print("D_KL (bans/sample):", nats_to_ban(D_KL) / effective_samples)

# %%
# plot histogram of normalized per-sample log likelihoods
gmm = (log_prob_gmm / effective_samples).ravel()
lf = (log_prob_u / effective_samples).ravel()

import pandas as pd

df = pd.DataFrame(
    {
        "loglik": np.concatenate([gmm, lf]),
        "model": (["GMM"] * gmm.size) + (["LF"] * lf.size),
    }
)

px.histogram(
    df,
    x="loglik",
    color="model",
    nbins=100,
    barmode="overlay",
    opacity=0.5,
    title="Normalized log likelihoods per sample",
    labels={"loglik": "Log likelihood per sample", "count": "Count"},
).show()

# %%
