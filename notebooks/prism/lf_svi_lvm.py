# %%
# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891/c/696a3d30-510c-8332-8075-7046937ecb61
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax as ox
from flax import nnx
from gpjax.dataset import Dataset
from gpjax.fit import get_batch
from gpjax.parameters import Parameter
from matplotlib import pyplot as plt

from prism.pack import NormalizedPACK
from prism.svi import batch_collapsed_elbo_masked, get_data
from utils.jax import vk

# %%
X, y = get_data(n=2000)
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
p = prior * likelihood

batch_size = 32
num_iters = 2000
num_restarts = 10

lr = 0.001

key = vk()
keys = jax.random.split(key, num_restarts)


def optimize(key):
    key, subkey = jax.random.split(key)

    z = jax.random.uniform(
        subkey, shape=(NUM_INDUCING, 1), minval=0.0, maxval=1.0
    )

    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p, inducing_inputs=z
    )

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


states, histories = jax.vmap(optimize)(keys)

# %%
plt.plot(histories.T)

# %%
# pick one posterior (e.g. best idx) and rebuild

i = int(jnp.nanargmin(jnp.array([h[-1] for h in histories])))  # best elbo

graphdef, _ = nnx.split(
    gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p,
        inducing_inputs=jnp.zeros((NUM_INDUCING, 1)),
    )
)

opt_posterior = nnx.merge(graphdef, jax.tree.map(lambda x: x[i], states))
history = histories[i]

plt.plot(history)

# %%
print(
    "Observation sigma_noise:",
    opt_posterior.posterior.likelihood.obs_stddev.value,
)

# %%
from utils.jax import safe_cholesky

Z = opt_posterior.inducing_inputs.value
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

test_index = 3210

mu_eps, Sigma_eps = infer_eps_posterior_single(
    opt_posterior, dataset.X[test_index], dataset.y[test_index]
)


def gp_posterior_mean_from_eps(q, t_star, mu_eps, jitter=1e-6):
    """
    GP posterior mean at t_star for ONE waveform,
    given inferred mu_eps.

    t_star : [T] query points
    mu_eps : [M]
    """

    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs

    # Kzz and its Cholesky
    Kzz = kernel.gram(Z).to_dense()
    Kzz = Kzz + jitter * jnp.eye(Z.shape[0])
    Lzz = jnp.linalg.cholesky(Kzz)

    # E[u | y]
    u_mean = Lzz @ mu_eps  # [M]

    # K_{t*Z}
    t_star = t_star[:, None]
    KtZ = kernel.cross_covariance(t_star, Z)  # [T,M]

    # posterior mean
    f_mean = KtZ @ jsp.linalg.solve(Kzz, u_mean)

    return f_mean.squeeze()


f_mean = gp_posterior_mean_from_eps(
    opt_posterior,
    t,
    mu_eps,
)


plt.plot(dataset.X[test_index], dataset.y[test_index], label="Data")
plt.plot(t, f_mean, label="Posterior mean")
plt.show()

# %%
mu_eps, Sigma_eps = jax.vmap(
    infer_eps_posterior_single,
    in_axes=(None, 0, 0),
)(opt_posterior, dataset.X, dataset.y)

# %%
from surrogate import source

lf_samples = source.get_lf_samples()[:2000]

Oqs = [s["p"]["Oq"] for s in lf_samples]

# %%

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X_latent = StandardScaler().fit_transform(mu_eps)

X_2d = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0,
).fit_transform(X_latent)

# scatter and color by Oq
plt.scatter(*X_2d.T, c=Oqs, cmap="viridis")
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
import numpy as np
from tqdm import tqdm

from prism.svi import get_waveforms

N_tau = WIDTH
tau_grid = np.linspace(0.0, 1.0, N_tau)

du_tau = []
for s_tau, s_du in tqdm(list(get_waveforms(lf_samples))):
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

# %%
# dump mu_eps as .npz file
from surrogate import source

lf_samples = source.get_lf_samples()[:2000]

oq = np.array([s["p"]["Oq"] for s in lf_samples])

np.savez("mu_eps_gplvm.npz", mu_eps=mu_eps, oq=oq)