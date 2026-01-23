# %%
from itertools import combinations

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax as ox
from gpjax.dataset import Dataset
from matplotlib import pyplot as plt

from prism.bgplvm import BayesianGPLVM
from utils import time_this
from utils.jax import pca_reduce

# %%
data = np.load("./data/mu_eps_gplvm.npz")

Y = np.asarray(data["mu_eps_std"])
Yvar = data["diag_eps_std"]
oq = data["oq"]

ndata = None

Y = Y[:ndata]
oq = oq[:ndata]
print(
    "Number of points: {} and Number of dimensions: {}".format(
        Y.shape[0], Y.shape[1]
    )
)

# %%
latent_dim = 8  # number of latent dimensions
num_inducing = 50  # number of inducing pts
num_data = Y.shape[0]  # number of data points


# Initialize via PCA:

# %%
X_mean_init = pca_reduce(Y, latent_dim)
X_var_init = np.ones((num_data, latent_dim))


# Pick inducing inputs randomly from dataset initialization:

# %%
np.random.seed(0)  # for reproducibility
inducing_variable = np.random.permutation(X_mean_init)[:num_inducing]

# %%
lengthscales = np.array([1.0] * latent_dim)

N = Y.shape[0]

kernel = gpx.kernels.RBF(lengthscale=jnp.array(lengthscales))

new_model = BayesianGPLVM(
    kernel,
    X_mu=jnp.array(X_mean_init),
    X_var=jnp.array(X_var_init),
    Z=jnp.array(inducing_variable),
)

elbo3 = new_model.elbo(jnp.array(Y))
print("Initial ELBO (new jax): ", elbo3)
# Initial ELBO:  -29403677.636068583

D = Dataset(X=Yvar, y=jnp.array(Y))  # just a hack to get Yvar data to model


def objective(model, data):
    return -model.elbo(data.y, obs_var_diag=data.X)


# %%
from gpjax.parameters import Parameter

with time_this() as timer:
    opt_model, history = gpx.fit(
        model=new_model,
        objective=objective,
        train_data=D,
        optim=ox.adam(1e-2),
        num_iters=10_000,
        key=jax.random.PRNGKey(0),
        batch_size=-1,
        trainable=Parameter,
    )

walltime = timer.walltime

# %%
plt.plot(-history)
plt.title("ELBO over iterations")
plt.show()

print("Optimized ELBO: ", -history[-1])

# %%
noise_std = np.sqrt(opt_model.sigma2)

print("Learned noise std:", noise_std)
print("Average data std:", np.std(Y, axis=0).mean())

# %%
inverse_lengthscale = 1.0 / opt_model.kernel.lengthscale
print(inverse_lengthscale)

# plot inverse lengthscales
plt.bar(range(latent_dim), inverse_lengthscale)
plt.xlabel("Latent dimension")
plt.ylabel("Inverse lengthscale")
plt.show()

# %%
print("Inferred sqrt(variance) of random point:")
print(np.sqrt(opt_model.X_var[0, :]))


# %%
# calculate empirical density
def latent_pair_density(Xmu, Xvar, pair, nx=200, ny=200, pad=0.5):
    i, j = pair
    mu = Xmu[:, [i, j]]  # (N,2)
    var = Xvar[:, [i, j]]  # (N,2)

    xmin, xmax = mu[:, 0].min() - pad, mu[:, 0].max() + pad
    ymin, ymax = mu[:, 1].min() - pad, mu[:, 1].max() + pad

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    logp = np.full(grid.shape[0], -np.inf)

    for n in range(mu.shape[0]):
        diff = grid - mu[n]
        invvar = 1.0 / var[n]
        quad = diff[:, 0] ** 2 * invvar[0] + diff[:, 1] ** 2 * invvar[1]
        logdet = np.log(var[n]).sum()
        lp = -0.5 * (quad + logdet + 2 * np.log(2 * np.pi))
        logp = np.logaddexp(logp, lp)

    logp -= np.log(mu.shape[0])
    dens = np.exp(logp).reshape(ny, nx)

    extent = [xmin, xmax, ymin, ymax]
    return dens, extent


# %%

top3 = np.argsort(-inverse_lengthscale)[:3]

pairs = list(combinations(top3, 2))

X_mu = opt_model.X_mu
X_var = opt_model.X_var

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
import numpy as np

for q in range(X_mu.shape[1]):
    corr = np.corrcoef(X_mu[:, q], oq)[0, 1]
    print(q, corr)


# %%
import plotly.express as px

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


# %%
# UPNEXT

# TODO:
# done: refind good one
# done: whiten!
# done: jax
# next: test ingesting data uncertainty => YES
#  * observation noise sigma from 0.5 to 1.5
#  * lengthscales identicalS
#  * X_var in main directions inflated by factor ~5!
#  * more outliers in latent space => solve by background Gaussian [https://chatgpt.com/s/t_6970f5efada881918d37b741a59e241f]
# next: restarts etc => YES: needed
# next: sampling to data => must embed this in SVI script
#  * fit 32 GMMs in quantization manner; throw away points in clusters with very low responsibilities

# %%
