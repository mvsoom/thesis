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
from matplotlib import pyplot as plt

from prism.bgplvm import BayesianGPLVM
from prism.pack import NormalizedPACK
from prism.svi import (
    batch_collapsed_elbo_masked,
    do_prism,
    get_data,
    gp_posterior_mean_from_eps,
    latent_pair_density,
    offdiag_energy_fraction,
    pick_best,
    svi_basis,
    whiten,
)
from utils import time_this
from utils.jax import pca_reduce, vk

jax.config.update("jax_enable_checks", False)

master_key = jax.random.key(seed)

# %%
N_TRAIN = 5000
N_TEST = 1000

X, y, oq = get_data(n=N_TRAIN)
N_TRAIN, WIDTH = X.shape  # Number of waveforms in dataset, max waveform length

dataset = Dataset(X=X, y=y)

print("Data shape:", X.shape, y.shape)


# %%
##############################################
# STAGE 1: COLLAPSED SVI
##############################################
def get_variational_svi_model(key=jax.random.PRNGKey(0)):
    Z = jax.random.uniform(
        key, shape=(num_inducing_svi, 1), minval=0.0, maxval=1.0
    )

    kernel = NormalizedPACK(d=d, J=J)
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=WIDTH)

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

plt.plot(histories.T)
plt.title("ELBO runs during training")
plt.show()

# %%
# pick best run
qsvi, history = pick_best(states, histories, get_variational_svi_model())

plt.plot(history)
plt.title("ELBO during training (best run)")
plt.show()

print("Observation sigma_noise:", qsvi.posterior.likelihood.obs_stddev)


# %%
# Define the global SVI basis found
def psi(t):
    return svi_basis(qsvi, t)


test_tau = jnp.linspace(0, 2, 500)
Psi = jax.vmap(psi)(test_tau)  # test indices

eps = jax.random.normal(vk(), shape=(num_inducing_svi, 5))
y = Psi @ eps
plt.plot(test_tau, y)
plt.title("Samples of learned latent function distribution")
plt.xlabel("tau")
plt.show()
# This is a prior draw from the learned RKHS subspace, not data-like yet.
# It answers: What does a typical GP draw look like under the learned kernel?
# expected to look generic and smooth

# %%
plt.plot(test_tau, Psi)
plt.title("Learned basis functions psi_m(t)")
plt.xlabel("tau")
plt.show()

# %%
# Embed the training data via the learned SVI basis (PRISM)
mu_eps, Sigma_eps = do_prism(qsvi, dataset)

# %%
# Now we test if the learned RKHS is rich enough to reconstruct some test waveforms
test_indices = jnp.array([10, 100, 250, 500])

f_means = jax.vmap(lambda eps: gp_posterior_mean_from_eps(qsvi, test_tau, eps))(
    mu_eps[test_indices]
)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
axes = axes.ravel()

for ax, idx, f_mean in zip(axes, test_indices, f_means):
    ax.plot(dataset.X[int(idx)], dataset.y[int(idx)], label="Data")
    ax.plot(test_tau, f_mean, label="Posterior mean")
    ax.set_title(f"test_index={int(idx)}")

axes[0].legend()
plt.tight_layout()
plt.show()

# %%
##############################################
# STAGE 2: B-GP-LVM
##############################################

# %%
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

plt.plot(histories.T)
plt.title("ELBO runs during BGPLVM training")
plt.show()

# %%
# pick best run
qlvm, history = pick_best(states, histories, get_variational_bgplvm_model())

plt.plot(history)
plt.title("ELBO during training (best run)")
plt.show()

# %%
noise_std = np.sqrt(qlvm.sigma2)

print("Learned noise std:", noise_std)
print("Average data std:", np.std(dataset_bgplvm.y, axis=0).mean())

# %%
inverse_lengthscale = 1.0 / qlvm.kernel.lengthscale
print(inverse_lengthscale)

# plot inverse lengthscales
plt.bar(range(latent_dim_bgplvm), inverse_lengthscale)
plt.xlabel("Latent dimension")
plt.ylabel("Inverse lengthscale")
plt.show()

# %%
print("Inferred sqrt(variance) of random point:")
print(np.sqrt(qlvm.X_var[0, :]))


# %%
top3 = np.argsort(-inverse_lengthscale)[:3]

pairs = list(combinations(top3, 2))

X_mu = qlvm.X_mu
X_var = qlvm.X_var

fig, ax = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))

showdensity = False
showscatter = True  # False#True

for a, pair in zip(np.atleast_1d(ax), pairs):
    if showdensity:
        dens, extent = latent_pair_density(X_mu, X_var, pair)

        # dens = np.log10(dens + 1e-12)

        a.imshow(
            dens,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=dens.min(),
            vmax=dens.max(),
            aspect="auto",
        )

    i, j = pair

    if showscatter:
        sc = a.scatter(X_mu[:, i], X_mu[:, j], c=oq, cmap="viridis", s=10)

    a.set_xlabel(f"latent {i}")
    a.set_ylabel(f"latent {j}")

if showscatter:
    fig.colorbar(sc, ax=ax, label="oq")
plt.show()

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
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()

# %%
from prism.xdgmm import e_step, fit_xdgmm

m = qlvm.X_mu
S = jax.vmap(jnp.diag)(qlvm.X_var)

K = 8

params, history, (mu0, cov0) = fit_xdgmm(
    m, S, K, alpha_quantile=1 - 1e-6, verbose=True
)

r, *_ = e_step(m, S, params, mu0, cov0, jitter=1e-6)

plt.plot(history)
plt.title("XD-GMM log likelihood during EM")
plt.show()

# %%

# plot cumulative histogram of background responsibilities
r_bg = r[:, 0]
plt.hist(r_bg, bins=100, cumulative=True, density=True)
plt.title("Cumulative histogram of background responsibilities")
plt.xlabel("Responsibility of background component")
plt.ylabel("Cumulative density")
plt.show()

is_outlier = r_bg > 0.95

labels = np.argmax(
    r[:, 1:], axis=1
)  # cluster index 0..K-1 (ignoring background)

labels = np.array(labels)

plt.bar(range(r.shape[1]), r.sum(axis=0) / r.sum())

# %%
import numpy as np
import plotly.express as px

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
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()

# %%
import numpy as np
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
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()

# %%
plvm = qlvm.build_posterior(dataset_bgplvm.y)

# %%
pi_free = np.exp(np.array(params.logits))[1:]
pi_free = pi_free / pi_free.sum()

# %%
k = np.random.choice(len(pi_free), p=pi_free)  # component index 0..K-1

mu_k = np.array(params.mu)[k]
cov_k = np.array(params.cov)[k]

z = np.random.multivariate_normal(mu_k, cov_k)

mu_y, diag_y = plvm.predict_f_meanvar(z, z * 0)
Sigma_y = jax.vmap(jnp.diag)(diag_y[0])

mu_eps_sample, mu_Sigma_sample = unwhiten(mu_y, Sigma_y)

f_sample = Psi @ mu_eps_sample.squeeze()

plt.figure(figsize=(6, 3))
plt.plot(test_tau, np.cumsum(f_sample), lw=2)
plt.title(f"Sample from GMM cluster {k + 1}")
plt.xlabel("tau")
plt.ylabel("f(t)")
plt.tight_layout()
plt.show()

# works for any grid, any resolution, any duration
