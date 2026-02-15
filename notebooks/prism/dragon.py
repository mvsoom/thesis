# %%
# Create dragon plot and other stuff
# See .chat:[PRocess-Induced Surrogate Modeling/PRISM]
import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax as ox
from flax import nnx
from gpjax.dataset import Dataset
from gpjax.fit import get_batch
from gpjax.parameters import Parameter
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from prism.pack import NormalizedPACK
from prism.svi import (
    batch_collapsed_elbo_masked,
    gp_posterior_mean_from_eps,
    pick_best,
)
from surrogate.prism import get_train_data
from utils import time_this
from utils.jax import vk

master_key = jax.random.PRNGKey(0)

# %%
num_train_samples = 5000

X, y, oq = get_train_data(n=num_train_samples)
N, WIDTH = X.shape  # Number of waveforms in dataset, max waveform length

dataset = Dataset(X=X, y=y)

# %%
test_batch = get_batch(dataset, 3, vk())
lengths = jnp.sum(~jnp.isnan(test_batch.y), axis=1)
plt.plot(test_batch.X.T, test_batch.y.T, marker="o")
plt.title(f"Random batch with lengths = {lengths}")
plt.xlabel("tau")
plt.ylabel("du")
plt.show()

# %%
kernel = NormalizedPACK(d=1)
NUM_INDUCING = 16

meanf = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=WIDTH)
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
posterior = prior * likelihood


def get_variational_model(z=None, num_inducing_svi=NUM_INDUCING):
    if z is None:
        z = jnp.zeros((num_inducing_svi, 1))

    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=z
    )
    return q


batch_size = 256
num_iters = 5000
num_restarts = 1

lr = 1e-3

master_key, subkey = jax.random.split(master_key)
keys = jax.random.split(subkey, num_restarts)


def optimize(key, num_inducing_svi=NUM_INDUCING):
    key, subkey = jax.random.split(key)

    z = jax.random.uniform(
        subkey, shape=(num_inducing_svi, 1), minval=0.0, maxval=1.0
    )

    q = get_variational_model(z)

    opt_posterior, history = gpx.fit(
        model=q,
        objective=lambda p, d: -batch_collapsed_elbo_masked(p, d, N),
        train_data=dataset,
        optim=ox.adam(learning_rate=lr),
        num_iters=num_iters,
        key=key,
        batch_size=batch_size,
        trainable=Parameter,
    )

    return nnx.state(opt_posterior), history


with time_this() as timer:
    states, histories = jax.vmap(optimize)(keys)

plt.plot(-histories.T)
plt.title("ELBO runs during training")
plt.show()

# %%
# pick best run
opt_posterior, history = pick_best(states, histories, get_variational_model())

plt.plot(-history)
plt.title("ELBO during training (best run)")
plt.show()

# %%
print(
    "Observation sigma_noise:",
    opt_posterior.posterior.likelihood.obs_stddev[...],
)

# %%
from utils.jax import safe_cholesky

Z = opt_posterior.inducing_inputs
Kmm = opt_posterior.posterior.prior.kernel.gram(Z).to_dense()
Lzz = safe_cholesky(Kmm)  # Kmm = Lzz @ Lzz.T


def psi(t):
    x = jnp.asarray(t).reshape(1, -1)
    Kxz = kernel.cross_covariance(x, Z)
    return jax.scipy.linalg.solve_triangular(
        Lzz, Kxz.T, lower=True
    ).squeeze()  # psi = Lzz^{-1} Kzx


t = jnp.linspace(0, 1, 500)
Psi = jax.vmap(psi)(t)

eps = jax.random.normal(vk(), shape=(NUM_INDUCING, 5))
y = Psi @ eps
plt.plot(t, y)
plt.title("Samples of learned latent function distribution")
plt.xlabel("tau")
plt.show()
# This is a prior draw from the learned RKHS subspace, not data-like yet.
# It answers: What does a typical GP draw look like under the learned kernel?
# expected to look generic and smooth

# %%
plt.plot(t, Psi)
plt.title("Learned basis functions psi_m(t)")
plt.xlabel("tau")
plt.show()

# %%
# Now we test if the learned RKHS is rich enough to reconstruct some test waveforms
from prism.svi import infer_eps_posterior_single

test_index = 321

mu_eps, Sigma_eps = infer_eps_posterior_single(
    opt_posterior, dataset.X[test_index], dataset.y[test_index]
)


f_mean = gp_posterior_mean_from_eps(
    opt_posterior,
    t,
    mu_eps,
)


plt.plot(dataset.X[test_index], dataset.y[test_index], label="Data")
plt.plot(t, f_mean, label="Posterior mean")
plt.show()

# %%
# Save GPU memory
with jax.default_device(jax.devices("cpu")[0]):
    mu_eps, Sigma_eps = jax.vmap(
        infer_eps_posterior_single,
        in_axes=(None, 0, 0),
    )(opt_posterior, dataset.X, dataset.y)


# %%
X_latent = StandardScaler().fit_transform(mu_eps)

X_2d = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0,
).fit_transform(X_latent)

# scatter and color by Oq
plt.scatter(*X_2d.T, c=oq, cmap="viridis")
plt.colorbar(label="OQ (open quotient)")
plt.title("t-SNE of inferred latent amplitudes colored by OQ")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.show()

# %%
# Uncertaintly confirms non-degenerate uncertainty, so MoPPCA with uncertainty makes sense
plt.hist(jnp.trace(Sigma_eps.T))
plt.title("Histogram of trace of posterior covariance Sigma_eps")

# %%
# Let us first get a scree plot of data resampled to same grid to get an idea of underlying linear dimension
# resample all samples to a common tau grid
from tqdm import tqdm

from surrogate import source
from surrogate.prism import extract_train_data

lf_samples = source.get_lf_samples(10_000)[:num_train_samples]

N_tau = WIDTH
tau_grid = np.linspace(0.0, 1.0, N_tau)

du_tau = []
for s_tau, s_du, _ in tqdm(list(extract_train_data(lf_samples))):
    du_tau.append(np.interp(tau_grid, s_tau, s_du))

# %%
# do PCA on resampled data
# We can see manifold gradient as expected: OQ varies smoothly
from sklearn.decomposition import PCA

du_tau = np.stack(du_tau)
pca = PCA().fit(du_tau)

# now do cumulative scree plot
explained_variance_ratio_cumsum = jnp.cumsum(pca.explained_variance_ratio_)
plt.plot(
    jnp.arange(1, len(explained_variance_ratio_cumsum) + 1),
    explained_variance_ratio_cumsum,
)
plt.xscale("log")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Scree plot of LF waveform data")
plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
plt.legend()
plt.show()
