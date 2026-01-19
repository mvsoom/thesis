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
NUM_INDUCING = 32

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


def gp_posterior_sample_from_eps(q, t_star, eps_sample, jitter=1e-6):
    """
    GP posterior sample at t_star for ONE waveform,
    given a sample eps ~ q(eps | y).
    """
    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs

    # Kzz and its Cholesky
    Kzz = kernel.gram(Z).to_dense()
    Kzz = Kzz + jitter * jnp.eye(Z.shape[0])
    Lzz = jnp.linalg.cholesky(Kzz)

    # sample u | y  via u = Lzz @ eps
    u = Lzz @ eps_sample  # [M]

    # K_{t*Z}
    t_star = t_star[:, None]
    KtZ = kernel.cross_covariance(t_star, Z)  # [T, M]

    # f = KtZ Kzz^{-1} u
    f = KtZ @ jsp.linalg.solve(Kzz, u)

    return f.squeeze()


f_mean = gp_posterior_mean_from_eps(
    opt_posterior,
    t,
    mu_eps,
)

key = vk()
n_samples = 1

# Cholesky of posterior eps covariance
L_eps = jnp.linalg.cholesky(Sigma_eps + 1e-9 * jnp.eye(Sigma_eps.shape[0]))

eps_samples = (
    mu_eps[None, :]
    + jax.random.normal(key, shape=(n_samples, mu_eps.shape[0])) @ L_eps.T
)  # [n_samples, M]

# plot posterior samples
for i in range(n_samples):
    f_s = gp_posterior_sample_from_eps(opt_posterior, t, eps_samples[i])
    plt.plot(t, f_s, color="C0", alpha=0.3)

# plot posterior mean
# plt.plot(dataset.X[test_index], dataset.y[test_index], label="Data")
# plt.plot(t, f_mean, "--", label="Posterior mean", color="black", linewidth=2)
plt.legend()
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
# Compare this to PCA: no separability/manifold structure

pca_eps = PCA(n_components=2).fit(mu_eps)
mu_eps_2d = pca_eps.transform(mu_eps)
plt.scatter(*mu_eps_2d.T, c=Oqs, cmap="viridis")
plt.colorbar(label="OQ (open quotient)")
plt.title("PCA (2D) of inferred latent amplitudes colored by OQ")
plt.xlabel("PCA dim 1")
plt.ylabel("PCA dim 2")
plt.show()

# %%
# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891-prism/c/696ae39d-13d0-8331-a49d-48c36210192b

# can first do PCA and project, which inflates the posterior covariances
# required number ~ 16

pca = PCA().fit(mu_eps)

# now do scree plot
explained_variance_ratio = pca.explained_variance_ratio_
plt.plot(
    jnp.arange(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
)
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance ratio")
plt.title("Scree plot of embedded latent amplitudes")
plt.legend()
plt.show()

# %%
# now do cumulative scree plot
explained_variance_ratio_cumsum = jnp.cumsum(pca.explained_variance_ratio_)
plt.plot(
    jnp.arange(1, len(explained_variance_ratio_cumsum) + 1),
    explained_variance_ratio_cumsum,
)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Cumulative scree plot of embedded latent amplitudes")
plt.axhline(0.99, color="red", linestyle="--", label="99% variance")
plt.legend()
plt.show()

# %%
# Step 3: pre-PCA with uncertainty propagation
# Nomenclature:
#   r_99: PCA dimension (99% heuristic)
#   r:   PPCA rank inside each mixture component

import numpy as np
from sklearn.decomposition import PCA

MU = np.asarray(mu_eps)  # [N, M]
S = np.asarray(Sigma_eps)  # [N, M, M]

pca_99 = PCA().fit(MU)
cumsum = np.cumsum(pca_99.explained_variance_ratio_)
r_99 = int(np.searchsorted(cumsum, 0.99) + 1)

print("r_99 (99% PCA dim):", r_99)

pca = PCA(n_components=r_99).fit(MU)

# PCA projection: m_n' = R (m_n - mean)
mu0 = pca.mean_  # [M]
R = pca.components_.astype(np.float64)  # [r_99, M]

MUc = MU - mu0[None, :]
MU_p = MUc @ R.T  # [N, r_99]

# Propagate covariance: S_n' = R S_n R^T
S_p = np.einsum("am,nmb,bk->nak", R, S, R.T)  # [N, r_99, r_99]

# quick sanity
print("MU_p shape:", MU_p.shape)
print("S_p shape:", S_p.shape)
print("mean trace S_p:", float(np.mean(np.trace(S_p, axis1=1, axis2=2))))

# %%
# Step 4: MoPPCA with per-point covariance (heteroscedastic full S_p)
#
# Model in r_99-space:
#   latent true x | k ~ N(mu_k, C_k)
#   observed MU_p = m | x ~ N(x, S_p[n])
#
# EM uses:
#   p(m | k) = N(m; mu_k, C_k + S_n)
#   posterior over x: N(xhat_nk, Sigma_nk) with
#     Sigma_nk = (C_k^{-1} + S_n^{-1})^{-1}
#     xhat_nk  = Sigma_nk (C_k^{-1} mu_k + S_n^{-1} m_n)
#
# After updating full C_k, we project it to PPCA form:
#   C_k approx W_k W_k^T + sig2_k I, with rank r.
#
# This keeps "same rank per cluster" by construction.

import numpy as np


def _log_gauss(m, mean, cov):
    # log N(m; mean, cov), cov SPD
    d = m.shape[0]
    L = np.linalg.cholesky(cov)
    x = np.linalg.solve(L, (m - mean))
    quad = float(x.T @ x)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def _ppca_project(C, r, floor=1e-8):
    # project SPD C to PPCA form rank r with isotropic noise
    # eigvals descending
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    d = C.shape[0]
    r = int(min(max(r, 0), d))

    if r == d:
        sig2 = max(float(np.min(evals)), floor)
        W = evecs @ np.diag(np.sqrt(np.maximum(evals - sig2, 0.0)))
        Cppca = W @ W.T + sig2 * np.eye(d)
        return W, sig2, Cppca

    if r == 0:
        sig2 = max(float(np.mean(evals)), floor)
        W = np.zeros((d, 0), dtype=C.dtype)
        Cppca = sig2 * np.eye(d)
        return W, sig2, Cppca

    tail = evals[r:]
    sig2 = float(np.mean(tail)) if tail.size else float(floor)
    sig2 = max(sig2, floor)

    lead = evals[:r]
    load = np.maximum(lead - sig2, 0.0)
    W = evecs[:, :r] @ np.diag(np.sqrt(load))
    Cppca = W @ W.T + sig2 * np.eye(d)
    return W, sig2, Cppca


def fit_moppca_obsnoise(
    MU_p, S_p, K=3, r=6, max_iters=50, tol=1e-4, seed=0, floor=1e-6
):
    N, d = MU_p.shape
    rng = np.random.default_rng(seed)

    # init: random responsibilities
    resp = rng.random((N, K))
    resp /= resp.sum(axis=1, keepdims=True)

    # initialize mu_k, C_k from resp
    mu_k = np.zeros((K, d))
    C_k = np.zeros((K, d, d))
    pi_k = resp.mean(axis=0)

    I = np.eye(d)

    for k in range(K):
        w = resp[:, k]
        Nk = float(np.sum(w) + 1e-12)
        mu_k[k] = (w[:, None] * MU_p).sum(axis=0) / Nk
        Xc = MU_p - mu_k[k][None, :]
        C = (w[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(
            axis=0
        ) / Nk
        C += floor * I
        _, _, C_k[k] = _ppca_project(C, r, floor=floor)

    last_ll = -np.inf

    for it in range(max_iters):
        # E-step: responsibilities using p(m | k) = N(mu_k, C_k + S_n)
        logp = np.zeros((N, K), dtype=np.float64)

        for n in range(N):
            Sn = S_p[n] + floor * I
            for k in range(K):
                cov = C_k[k] + Sn
                logp[n, k] = np.log(max(pi_k[k], 1e-12)) + _log_gauss(
                    MU_p[n], mu_k[k], cov
                )

        # normalize
        logp_max = np.max(logp, axis=1, keepdims=True)
        p = np.exp(logp - logp_max)
        resp = p / np.sum(p, axis=1, keepdims=True)

        # compute observed-data loglik
        ll = float(np.sum(logp_max[:, 0] + np.log(np.sum(p, axis=1) + 1e-300)))

        # M-step: update pi_k, mu_k, C_k via expected latent x
        pi_k = resp.mean(axis=0)

        for k in range(K):
            w = resp[:, k]
            Nk = float(np.sum(w) + 1e-12)

            # accumulate for mu and C
            xhat_sum = np.zeros(d)
            for n in range(N):
                Sn = S_p[n] + floor * I

                # Solve Sigma_nk and xhat_nk stably via choleskies
                # Sigma = inv(inv(C) + inv(S))
                # xhat = Sigma (inv(C) mu + inv(S) m)
                C = C_k[k] + floor * I

                Lc = np.linalg.cholesky(C)
                Ls = np.linalg.cholesky(Sn)

                # inv(C) mu
                ic_mu = np.linalg.solve(Lc.T, np.linalg.solve(Lc, mu_k[k]))
                # inv(S) m
                is_m = np.linalg.solve(Ls.T, np.linalg.solve(Ls, MU_p[n]))

                # A = inv(C) + inv(S)
                # apply A^{-1} to vectors by solving with its Cholesky
                # build A explicitly (d is small, <= 32)
                iC = np.linalg.solve(Lc.T, np.linalg.solve(Lc, I))
                iS = np.linalg.solve(Ls.T, np.linalg.solve(Ls, I))
                A = iC + iS
                LA = np.linalg.cholesky(A)

                xhat = np.linalg.solve(LA.T, np.linalg.solve(LA, ic_mu + is_m))
                xhat_sum += w[n] * xhat

            mu_new = xhat_sum / Nk
            mu_k[k] = mu_new

            # update C_k from expected second moment
            Cacc = np.zeros((d, d))
            for n in range(N):
                Sn = S_p[n] + floor * I
                C = C_k[k] + floor * I

                Lc = np.linalg.cholesky(C)
                Ls = np.linalg.cholesky(Sn)

                iC = np.linalg.solve(Lc.T, np.linalg.solve(Lc, I))
                iS = np.linalg.solve(Ls.T, np.linalg.solve(Ls, I))
                A = iC + iS
                LA = np.linalg.cholesky(A)

                # Sigma_nk
                Sigma = np.linalg.solve(LA.T, np.linalg.solve(LA, I))

                # xhat_nk
                ic_mu = iC @ mu_k[k]
                is_m = iS @ MU_p[n]
                xhat = Sigma @ (ic_mu + is_m)

                dx = (xhat - mu_k[k])[:, None]
                Cacc += resp[n, k] * (Sigma + dx @ dx.T)

            C_full = Cacc / Nk
            C_full += floor * I

            # project to PPCA rank r
            _, _, Cppca = _ppca_project(C_full, r, floor=floor)
            C_k[k] = Cppca

        # stopping
        if it > 0:
            rel = abs(ll - last_ll) / (1.0 + abs(last_ll))
            print(f"it {it:03d}  ll {ll:.3f}  rel {rel:.3e}")
            if rel < tol:
                break
        else:
            print(f"it {it:03d}  ll {ll:.3f}")

        last_ll = ll

    # recover W_k, sig2_k for convenience (from final C_k)
    W_k = []
    sig2_k = []
    for k in range(K):
        W, sig2, _ = _ppca_project(C_k[k], r, floor=floor)
        W_k.append(W)
        sig2_k.append(sig2)
    W_k = np.stack(W_k, axis=0)  # [K, d, r]
    sig2_k = np.asarray(sig2_k)  # [K]

    return {
        "pi": pi_k,
        "mu": mu_k,
        "C": C_k,
        "W": W_k,
        "sig2": sig2_k,
        "resp": resp,
        "r_99": d,
        "r": int(r),
        "mu0": mu0,
        "R": R,
    }

# %%
# For downstream inference only I = K*r matters
K = 6
r = r_99

moppca = fit_moppca_obsnoise(
    MU_p,
    S_p * 0,  ########################## FIXME
    K=K,
    r=r,
    max_iters=100,
    tol=1e-4,
    seed=0,
    floor=1e-6,
)

print("pi:", moppca["pi"])
print("sig2:", moppca["sig2"])
print("resp shape:", moppca["resp"].shape)

# quick look: hard assignments histogram
hard = np.argmax(moppca["resp"], axis=1)
print("cluster counts:", np.bincount(hard, minlength=K))

# Lift MoPPCA clusters back to full eps-space (dimension M)
# Produces (mu_eps_k, cov_root_eps_k) for each cluster k

import numpy as np

M = mu_eps.shape[1]  # original eps dimension
R = moppca["R"]  # [r_99, M]
mu0 = moppca["mu0"]  # [M]

K = moppca["pi"].shape[0]
r_ppca = moppca["r"]

mu_eps_k = []
cov_root_eps_k = []

# residual variance outside PCA subspace
# conservative choice: mean discarded eigenvalue
pca_full = PCA().fit(mu_eps)
residual_var = np.mean(pca_full.explained_variance_[r_99:]) if r_99 < M else 0.0
residual_var = max(residual_var, 1e-8)

for k in range(K):
    # mean in eps space
    mu_k = mu0 + R.T @ moppca["mu"][k]  # [M]
    mu_eps_k.append(mu_k)

    # covariance in eps space
    C_r = moppca["C"][k]  # [r_99, r_99]
    C_eps = R.T @ C_r @ R  # [M, M]

    # add isotropic residual outside PCA subspace
    P = R.T @ R  # projector
    C_eps += residual_var * (np.eye(M) - P)

    # covariance root via cholesky
    L = np.linalg.cholesky(C_eps + 1e-8 * np.eye(M))
    cov_root_eps_k.append(L)

mu_eps_k = np.stack(mu_eps_k)  # [K, M]
cov_root_eps_k = np.stack(cov_root_eps_k)  # [K, M, M]

print("mu_eps_k:", mu_eps_k.shape)
print("cov_root_eps_k:", cov_root_eps_k.shape)

# Build BLR surrogate per cluster

import jax.numpy as jnp

from gp import blr

noise_variance = float(opt_posterior.posterior.likelihood.obs_stddev.value**2)


# reuse psi(t) from earlier code
def phi_fn(t):
    return psi(t)


blr_clusters = []

t_plot = jnp.linspace(0.0, 1.0, 400)

for k in range(K):
    blr_k = blr.BayesianLinearRegressor(
        phi=phi_fn,
        X=t_plot,  # dummy grid; actual sampling grid passed later
        mu=jnp.asarray(mu_eps_k[k]),
        cov_root=jnp.asarray(cov_root_eps_k[k]),
        noise_variance=noise_variance,
    )
    blr_clusters.append(blr_k)

print(f"Built {len(blr_clusters)} BLR surrogate models")

# Sample waveforms per cluster and plot

import matplotlib.pyplot as plt

from utils.jax import vk

n_samples = 1

fig, axes = plt.subplots(K, 1, figsize=(8, 2.2 * K), sharex=True)

if K == 1:
    axes = [axes]

for k, ax in enumerate(axes):
    blr_k = blr_clusters[k]

    for i in range(n_samples):
        y = blr_k.sample(vk(), X_test=t_plot)
        ax.plot(t_plot, y, alpha=0.5)

    ax.set_ylabel(f"cluster {k}")
    ax.plot(t_plot, blr_k.mean, color="black", linewidth=2, label="mean")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("tau")
plt.suptitle(f"PRISM surrogate samples per MoPPCA cluster (K={K}, r={r})")
plt.tight_layout()
plt.show()

# %%
# HDBSCAN
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------- preprocess ----------
X = mu_eps
X = StandardScaler().fit_transform(X)

# optional: knock down noise, keep geometry
pca = PCA(n_components=25, whiten=True, random_state=0)
Xp = pca.fit_transform(X)

print("PCA explained var (first 5):", pca.explained_variance_ratio_[:5])
print("Cumulative (10):", pca.explained_variance_ratio_.sum())

# ---------- HDBSCAN ----------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,  # large -> conservative
    min_samples=50,
    metric="euclidean",
    cluster_selection_method="eom",
)

labels = clusterer.fit_predict(Xp)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print("clusters:", n_clusters)
print("noise frac:", n_noise / len(labels))

# ---------- 2D viz (PCA, not t-SNE) ----------
Xp2 = PCA(n_components=2, random_state=0).fit_transform(Xp)

plt.figure(figsize=(6, 5))
plt.scatter(Xp2[:, 0], Xp2[:, 1], c=labels, s=10, cmap="tab10")
plt.title("HDBSCAN on mu_eps (PCA space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="cluster label")
plt.tight_layout()
plt.show()

# ---------- soft info ----------
plt.figure(figsize=(6, 4))
plt.hist(clusterer.probabilities_, bins=40)
plt.title("HDBSCAN membership probabilities")
plt.xlabel("probability")
plt.ylabel("count")
plt.tight_layout()
plt.show()

# %%

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(mu_eps)


def intrinsic_dim_lb(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    dists = dists[:, 1:]  # drop self-distance

    logs = np.log(dists[:, -1][:, None] / dists[:, :-1])
    m = (k - 1) / np.sum(logs, axis=1)
    return m.mean(), m.std()


for k in [5, 10, 20, 30]:
    mean, std = intrinsic_dim_lb(X, k=k)
    print(f"k={k:2d}  ID ≈ {mean:.2f} ± {std:.2f}")


def intrinsic_dim_2nn(X):
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mu = r2 / r1
    return 1.0 / np.mean(np.log(mu))


print("2NN intrinsic dimension:", intrinsic_dim_2nn(X))

# %%
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(mu_eps)
n = X.shape[0]
k = 30

nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
dists, idx = nbrs.kneighbors(X)
dists = dists[:, 1:]
idx = idx[:, 1:]

eps = np.median(dists)
W = np.exp(-(dists**2) / (eps**2))

rows = np.repeat(np.arange(n), k)
cols = idx.reshape(-1)
data = W.reshape(-1)

K = csr_matrix((data, (rows, cols)), shape=(n, n))
K = 0.5 * (K + K.T)

d = np.array(K.sum(axis=1)).ravel()
Dinv_sqrt = diags(1.0 / np.sqrt(d + 1e-12))

S = Dinv_sqrt @ K @ Dinv_sqrt  # symmetric

# quick sanity: spectral radius should be <= 1 + tiny numerical slack
nev = 8
vals, vecs = eigsh(S, k=nev, which="LA")
vals = np.sort(vals)[::-1]
print("top eigs:", vals)

# diffusion coords (3D)
phi = vecs[:, np.argsort(vals)[::-1]]  # align with sorted vals if needed
# easiest: recompute ordering cleanly:
order = np.argsort(-vals)
vals = vals[order]
vecs = vecs[:, order]

psi = vecs[:, 1:4] * vals[1:4]  # DM1..DM3

# %%
import matplotlib.pyplot as plt

# DM coords
psi = vecs[:, 1:4] * vals[1:4]  # (n, 3)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(psi[:, 0], psi[:, 1], psi[:, 2], c=Oqs, s=8, cmap="viridis")
ax.set_xlabel("DM1")
ax.set_ylabel("DM2")
ax.set_zlabel("DM3")
fig.colorbar(sc, ax=ax, label="OQ")
plt.tight_layout()
plt.show()

# %%
# choose coordinate: try DM3 first
oq = np.array(Oqs)
s = psi[:, 2].copy()

# make increasing with OQ (optional)
if np.corrcoef(s, oq)[0, 1] < 0:
    s = -s

# normalize
s = (s - s.min()) / (s.max() - s.min() + 1e-12)

plt.figure()
plt.scatter(s, oq, s=8)
plt.xlabel("s")
plt.ylabel("OQ")
plt.title("OQ vs manifold coord")
plt.tight_layout()
plt.show()

import numpy as np


def rbf_features(s, m=16, length=0.15):
    s = s.reshape(-1, 1)
    centers = np.linspace(0, 1, m).reshape(1, -1)
    Phi = np.exp(-0.5 * ((s - centers) / length) ** 2)
    Phi = np.concatenate([np.ones((len(s), 1)), Phi], axis=1)  # bias
    return Phi


Phi = rbf_features(s, m=16, length=0.12)  # tune length a bit
W = mu_eps  # (N,32)

lam = 1e-3  # ridge strength
A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
B = Phi.T @ W
beta = np.linalg.solve(A, B)  # (P,32)

W_hat = Phi @ beta
res = W - W_hat
print("RMSE per-dim (median):", np.median(np.sqrt(np.mean(res**2, axis=0))))