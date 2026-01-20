# %%
# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891/c/696a3d30-510c-8332-8075-7046937ecb61
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
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
from prism.svi import batch_collapsed_elbo_masked, get_data
from utils.jax import vk

# %%
num_train_samples = 5000

X, y = get_data(n=num_train_samples)
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
plt.title("ELBO runs during training")
plt.show()

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
plt.title("ELBO during training (best run)")
plt.show()

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

test_index = 321

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

lf_samples = source.get_lf_samples(10_000)[:num_train_samples]

oq = np.array([s["p"]["Oq"] for s in lf_samples])

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
# Posterior covariance matters?

C = Sigma_eps[:100]  # (100,16,16)

diag_energy = np.sum(np.square(np.diagonal(C, axis1=1, axis2=2)))
total_energy = np.sum(np.square(C))
offdiag_ratio = 1.0 - diag_energy / total_energy

print("offdiag energy fraction:", offdiag_ratio)


def cov_to_corr(S):
    d = np.sqrt(np.diag(S))
    return S / (d[:, None] * d[None, :])


corrs = np.array([cov_to_corr(S) for S in C])  # (100,16,16)
mean_corr = np.mean(np.abs(corrs), axis=0)

# ignore diagonal
mask = ~np.eye(16, dtype=bool)
print("mean |corr| offdiag:", mean_corr[mask].mean())
print("max  |corr| offdiag:", mean_corr[mask].max())

ranks = []
entropies = []

for S in C:
    w = np.linalg.eigvalsh(S)
    w = np.clip(w, 1e-12, None)
    p = w / w.sum()
    H = -np.sum(p * np.log(p))
    eff_rank = np.exp(H)
    ranks.append(eff_rank)

ranks = np.array(ranks)

print("mean effective rank:", ranks.mean())
print("min / max rank:", ranks.min(), ranks.max())

import matplotlib.pyplot as plt

i = 50
mu = mu_eps[i]
S = C[i]
Sdiag = np.diag(np.diag(S))

x_full = np.random.multivariate_normal(mu, S, size=500)
x_diag = np.random.multivariate_normal(mu, Sdiag, size=500)

# project onto first two principal directions of full covariance
w, V = np.linalg.eigh(S)
U = V[:, -2:]  # top 2 eigendirs

pf = x_full @ U
pd = x_diag @ U

plt.figure(figsize=(4, 4))
plt.scatter(pf[:, 0], pf[:, 1], s=5, alpha=0.4, label="full")
plt.scatter(pd[:, 0], pd[:, 1], s=5, alpha=0.4, label="diag")
plt.legend()
plt.title("full vs diag posterior samples")
plt.show()


def kl_full_to_diag(S):
    D = np.diag(np.diag(S))
    invD = np.linalg.inv(D)
    k = S.shape[0]
    return 0.5 * (
        np.trace(invD @ S) - k + np.log(np.linalg.det(D) / np.linalg.det(S))
    )


kls = np.array([kl_full_to_diag(S) for S in C])

print("mean KL(full || diag):", kls.mean())
print("max  KL:", kls.max())

# %%
# Apply a global whitening transform to the average posterior covariance
# This establishes a global basis in which covariances are nearly diagonal
# and there IS such a very good basis
C = Sigma_eps  # [:100]    # (100,16,16)
mu = mu_eps  # [:100]     # (100,16)

import numpy as np

# mean covariance
Cbar = C.mean(axis=0)

# eigendecomposition
w, V = np.linalg.eigh(Cbar)
w = np.clip(w, 1e-12, None)

# whitening and unwhitening
W = V @ np.diag(1.0 / np.sqrt(w)) @ V.T  # whitening
Wi = V @ np.diag(np.sqrt(w)) @ V.T  # inverse (for decoding later)


Cw = np.einsum("ij,njk,kl->nil", W, C, W.T)  # (100,16,16)

diag_energy = np.sum(np.square(np.diagonal(Cw, axis1=1, axis2=2)))
total_energy = np.sum(np.square(Cw))
offdiag_ratio = 1.0 - diag_energy / total_energy

print("WHITENED offdiag energy fraction:", offdiag_ratio)


def cov_to_corr(S):
    d = np.sqrt(np.diag(S))
    return S / (d[:, None] * d[None, :])


corrs_w = np.array([cov_to_corr(S) for S in Cw])
mean_corr_w = np.mean(np.abs(corrs_w), axis=0)

mask = ~np.eye(16, dtype=bool)
print("WHITENED mean |corr| offdiag:", mean_corr_w[mask].mean())
print("WHITENED max  |corr| offdiag:", mean_corr_w[mask].max())

ranks_w = []

for S in Cw:
    w = np.linalg.eigvalsh(S)
    w = np.clip(w, 1e-12, None)
    p = w / w.sum()
    H = -np.sum(p * np.log(p))
    eff_rank = np.exp(H)
    ranks_w.append(eff_rank)

ranks_w = np.array(ranks_w)

print("WHITENED mean effective rank:", ranks_w.mean())
print("WHITENED min / max rank:", ranks_w.min(), ranks_w.max())


def kl_full_to_diag(S):
    D = np.diag(np.diag(S))
    invD = np.linalg.inv(D)
    k = S.shape[0]
    return 0.5 * (
        np.trace(invD @ S) - k + np.log(np.linalg.det(D) / np.linalg.det(S))
    )


kls_w = np.array([kl_full_to_diag(S) for S in Cw])

print("WHITENED mean KL(full || diag):", kls_w.mean())
print("WHITENED max  KL:", kls_w.max())

i = 0
mu0 = mu[i]
S0 = C[i]
S0w = Cw[i]

# whitened mean too
mu0w = W @ mu0

Sdiag = np.diag(np.diag(S0w))

x_full = np.random.multivariate_normal(mu0w, S0w, size=500)
x_diag = np.random.multivariate_normal(mu0w, Sdiag, size=500)

w, V = np.linalg.eigh(S0w)
U = V[:, -2:]

pf = x_full @ U
pd = x_diag @ U

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
plt.scatter(pf[:, 0], pf[:, 1], s=5, alpha=0.4, label="full (whitened)")
plt.scatter(pd[:, 0], pd[:, 1], s=5, alpha=0.4, label="diag (whitened)")
plt.legend()
plt.title("whitened full vs diag")
plt.show()

# %%
# Whiten and center mu_eps and Sigma_eps
mu0 = mu.mean(axis=0)  # (16,)
mu_eps_std = (mu - mu0) @ W.T  # (N,16)
Sigma_eps_std = np.einsum("ij,njk,kl->nil", W, C, W.T)

diag_eps_std = np.diagonal(Sigma_eps_std, axis1=1, axis2=2)

# from eps_std to original:
# w = w_std @ Wi.T + mu0
# Sigma = Wi @ Sigma_std @ Wi.T

# %%
# dump mu_eps as .npz file
np.savez(
    "data/mu_eps_gplvm.npz",
    mu_eps_std=mu_eps_std,
    diag_eps_std=diag_eps_std,
    oq=oq,
)
# %%
