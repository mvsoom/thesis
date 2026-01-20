# %%
from itertools import combinations

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.config import (
    default_float,
    set_default_float,
    set_default_summary_fmt,
)
from gpflow.utilities import ops, print_summary

set_default_float(np.float64)
set_default_summary_fmt("notebook")

# %%
data = np.load("./data/mu_eps_gplvm.npz")


# Following the GPflow notation we assume this dataset has a shape of `[num_data, output_dim]`
Y = tf.convert_to_tensor(data["mu_eps_std"], dtype=default_float())
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
X_mean_init = ops.pca_reduce(Y, latent_dim)
X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())


# Pick inducing inputs randomly from dataset initialization:

# %%
np.random.seed(0)  # for reproducibility
tf.random.set_seed(0)

inducing_variable = tf.convert_to_tensor(
    np.random.permutation(X_mean_init.numpy())[:num_inducing],
    dtype=default_float(),
)


# We construct a Squared Exponential (SE) kernel operating on the two-dimensional latent space.
# The `ARD` parameter stands for Automatic Relevance Determination, which in practice means that
# we learn a different lengthscale for each of the input dimensions. See [Manipulating kernels](../advanced/kernels.ipynb) for more information.

# %%
lengthscales = tf.convert_to_tensor([1.0] * latent_dim, dtype=default_float())
kernel = gpflow.kernels.RBF(lengthscales=lengthscales)


# We have all the necessary ingredients to construct the model. GPflow contains an implementation of the Bayesian GPLVM:

# %%
gplvm = gpflow.models.BayesianGPLVM(
    Y,
    X_data_mean=X_mean_init,
    X_data_var=X_var_init,
    kernel=kernel,
    inducing_variable=inducing_variable,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def step():
    with tf.GradientTape() as tape:
        loss = gplvm.training_loss()
    grads = tape.gradient(loss, gplvm.trainable_variables)
    optimizer.apply_gradients(zip(grads, gplvm.trainable_variables))
    return loss


history = []
for it in range(10_000):
    loss = step()
    history.append(loss.numpy())
    if it % 100 == 0:
        print(it, loss.numpy())

# %%
plt.plot(history)


# ## Model analysis
# GPflow allows you to inspect the learned model hyperparameters.

# %%
print_summary(gplvm)

# %%
print(gplvm.kernel.lengthscales)

# plot inverse lengthscales
plt.bar(range(latent_dim), 1.0 / gplvm.kernel.lengthscales.numpy())
plt.xlabel("Latent dimension")
plt.ylabel("Inverse lengthscale")
plt.show()

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


X = gplvm.X_data_mean.numpy()

pairs = list(combinations(range(latent_dim), 2))
pairs = [(0, 1), (1, 5), (0, 5)]

Xmu = gplvm.X_data_mean.numpy()
Xvar = gplvm.X_data_var.numpy()

fig, ax = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))

showdensity = True
showscatter = False  # False#True

for a, pair in zip(np.atleast_1d(ax), pairs):
    if showdensity:
        dens, extent = latent_pair_density(Xmu, Xvar, pair)

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
        sc = a.scatter(Xmu[:, i], Xmu[:, j], c=oq, cmap="viridis", s=10)

    a.set_xlabel(f"latent {i}")
    a.set_ylabel(f"latent {j}")

if showscatter:
    fig.colorbar(sc, ax=ax, label="oq")
plt.show()

# %%
import numpy as np

for q in range(X.shape[1]):
    corr = np.corrcoef(X[:, q], oq)[0, 1]
    print(q, corr)


# %%
import plotly.express as px

Xmu = gplvm.X_data_mean.numpy()

i, j, k = 0, 1, 5  # choose any triple

fig = px.scatter_3d(
    x=Xmu[:, i],
    y=Xmu[:, j],
    z=Xmu[:, k],
    color=oq,  # same array you used in matplotlib
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
# UPNEXT

# realizing this is a ~3d manifold
# https://chatgpt.com/c/696bb8b9-fb38-8333-9ce0-b39f0760ffaa

# B-GP-LVM and jax implementation
# https://chatgpt.com/c/696bd63f-2888-8330-854f-675fa7c2fc00
# Psi stats for kernels.SquaredExponential are implemented by gpflow!

# nice imgs are in ./lvmm_q=2.png and ./lvmm_q=3.png

# next: refind good one
# next: whiten!
# next: ingest data uncertainty
# next: jax: restarts etc
# next: sampling to data
