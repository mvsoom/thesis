"""
Unlike alignment based functional PCA, state space models, or neural sequence encoders, the proposed method uses a reduced rank GP as a canonical projection that maps variable length and time warped simulator outputs into a common latent coordinate system with fixed dimension, enabling downstream probabilistic compression and mixture modeling.
"""

# %%
# parameters, export
kernel = "pack:0"  # higher d means longer runtime due to zonal Mercer expansion
normalized = True
J = 64
iteration = 1
seed = 4283955834

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gfm.ack import DiagonalTACK
from gp.blr import blr_from_mercer, log_probability
from pack import PACK
from surrogate import source

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu") # GPU = 20x speedup


from utils import constants, time_this
from utils.jax import vk

# %%
NUM_TRAIN = 1000

lf_samples = source.get_lf_samples()

train_lf_samples = lf_samples[:NUM_TRAIN]
test_lf_samples = lf_samples[NUM_TRAIN:]


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


samples = []
for lf_sample in train_lf_samples:
    du = lf_sample["u"]
    t_ms = lf_sample["t"]
    period_ms = lf_sample["p"]["T0"]
    log_prob_u = lf_sample["log_prob_u"]
    if np.isfinite(log_prob_u):
        samples.append(
            {
                "period_ms": period_ms,
                "t_ms": t_ms,
                "tau": warp_time(t_ms, period_ms),
                "du": du,
                "log_prob_u": lf_sample["log_prob_u"],
            }
        )

test_samples = []
for lf_sample in test_lf_samples:
    du = lf_sample["u"]
    t_ms = lf_sample["t"]
    period_ms = lf_sample["p"]["T0"]
    log_prob_u = lf_sample["log_prob_u"]
    if np.isfinite(log_prob_u):
        test_samples.append(
            {
                "period_ms": period_ms,
                "t_ms": t_ms,
                "tau": warp_time(t_ms, period_ms),
                "du": du,
                "log_prob_u": lf_sample["log_prob_u"],
            }
        )

# %%
# Plot distribution of log_prob_u
log_probs = np.array([sample["log_prob_u"] for sample in samples])
plt.figure(figsize=(6, 4))
plt.hist(log_probs, bins=30, density=True)
plt.title("Distribution of log probability of LF samples")
plt.xlabel("log probability")
plt.ylabel("Density")
plt.show()

# Plot samples ordered by quantile of log_prob_u on a sigle plot, including min and max values
sorted_indices = np.argsort(log_probs)
num_samples_to_plot = 5
selected_indices = np.linspace(
    0, len(sorted_indices) - 1, num_samples_to_plot, dtype=int
)
plt.figure(figsize=(10, 6))
for idx in selected_indices:
    sample = samples[sorted_indices[idx]]
    tau = sample["tau"]
    du = sample["du"]
    plt.plot(
        tau, du, alpha=0.5, label=f"log_prob(u)={sample['log_prob_u']:.2f}"
    )
plt.title("LF derivative samples ordered by log probability")
plt.xlabel("Normalized time (tau)")
plt.ylabel("LF derivative (du)")
plt.legend()
plt.show()

# %%
# resample all samples to a common tau grid
N_tau = int(max(sample["du"].shape[0] for sample in samples))
tau_grid = np.linspace(0.0, 1.0, N_tau)
tau = jnp.asarray(tau_grid)
dtau = tau_grid[1] - tau_grid[0]

du_tau = []
for sample in samples:
    du_tau.append(np.interp(tau_grid, sample["tau"], sample["du"]))

du_tau = np.stack(du_tau)

sample0 = samples[0]

fig, ax = plt.subplots(figsize=(10, 5))
plt.title("Data to fit (normalized time)")
plt.plot(tau_grid, du_tau[0], label="LF derivative")
plt.xlabel("Tau (unit period)")
plt.ylabel("Amplitude")
plt.legend()

# %%
d = int(kernel[-1])
t1 = 0.0
t2 = 1.0
period_norm = 1.0
print(f"Using J={J} harmonics (max frequency={J} cycles/period)")

import jax.numpy as jnp
from jax.nn import sigmoid
from jax.scipy.stats import norm


def ptform(z):
    z1, z2 = jnp.split(z, [3])
    x = jnp.concatenate([10.0**z1, sigmoid(z2)])
    return x


def logprior(z):
    return jnp.sum(norm.logpdf(z))


def build_theta(x, J=J):
    return {
        "sigma_noise": x[0],
        "sigma_a": x[1],
        "sigma_b": x[2],
        "tc": x[3],
        "center": x[4],
    }


def build_kernel(theta):
    tack = DiagonalTACK(
        d=d,
        normalized=normalized,
        center=theta["center"],
        sigma_b=theta["sigma_b"],
        sigma_c=1.0,
    )

    pack = PACK(
        tack,
        period=period_norm,
        t1=t1,
        t2=theta["tc"],
        J=J,
    )

    return theta["sigma_a"] ** 2 * pack


def build_gp(theta):
    pack = build_kernel(theta)
    return blr_from_mercer(pack, tau, noise_variance=theta["sigma_noise"] ** 2)


z = np.random.normal(size=100)
x = ptform(z)
theta = build_theta(x)
ndim = len(theta)
pack = build_kernel(theta)

# %%
# build generative model
# Compute constants (basis does not depend on parameters)
Phi = jax.vmap(pack.compute_phi)(tau)

PhiT_Phi = jnp.matmul(Phi.T, Phi)
du_stack = jnp.asarray(du_tau)
PhiT_y_all = jnp.matmul(du_stack, Phi)


def loglikelihood_slow(theta):
    pack = build_kernel(theta)

    # Bypass build_gp() to cache building Phi, PhiT_Phi, PhiT_y_all
    cov_root = pack.compute_weights_root()
    logl = jax.vmap(
        lambda y, PhiT_y: log_probability(
            y=y,
            Phi=Phi,
            cov_root=cov_root,
            noise_variance=theta["sigma_noise"] ** 2,
            PhiT_Phi=PhiT_Phi,
            PhiT_y=PhiT_y,
            jitter=constants.NOISE_FLOOR_POWER,
        )
    )(du_stack, PhiT_y_all)

    return jnp.sum(logl)


loglikelihood_slow(theta)

# %%
# Faster, caching version

import jax
import jax.numpy as jnp
from jax.scipy import linalg


def latent_cache(cov_root, PhiT_Phi, noise_variance, dtype, jitter=None):
    R = cov_root.shape[1]

    A = cov_root.T @ PhiT_Phi @ cov_root

    diag_scale = jnp.mean(jnp.diag(A))
    eps = jnp.sqrt(jnp.finfo(dtype).eps) if jitter is None else jitter
    sigma2 = noise_variance + eps * diag_scale

    Z = A + sigma2 * jnp.eye(R, dtype=dtype)
    Lc, lower = linalg.cho_factor(Z, lower=True, check_finite=False)

    logdet_Z = 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))
    return Lc, lower, sigma2, logdet_Z


def loglikelihood(theta):
    pack = build_kernel(theta)
    cov_root = pack.compute_weights_root()

    # factorization that is shared across all y for this x
    Lc, lower, sigma2, logdet_Z = latent_cache(
        cov_root=cov_root,
        PhiT_Phi=PhiT_Phi,
        noise_variance=theta["sigma_noise"] ** 2,
        dtype=Phi.dtype,
        jitter=constants.NOISE_FLOOR_POWER,
    )

    # constants for all y
    Nobs = Phi.shape[0]
    R = cov_root.shape[1]
    logdet_K_const = (Nobs - R) * jnp.log(sigma2) + logdet_Z
    norm_const = Nobs * jnp.log(2.0 * jnp.pi)

    y_norm = jnp.sum(du_stack * du_stack, axis=1)
    cov_root_T = cov_root.T

    def one(y_norm_i, PhiT_y):
        b = cov_root_T @ PhiT_y
        m_z = linalg.cho_solve((Lc, lower), b, check_finite=False)
        quad = (1.0 / sigma2) * (y_norm_i - b @ m_z)
        return -0.5 * (logdet_K_const + quad + norm_const)

    logl = jax.vmap(one)(y_norm, PhiT_y_all)
    return jnp.sum(logl)


loglikelihood(theta), loglikelihood_slow(theta)  # equal?


# %%
@jax.jit
def neg_logpost(z):
    theta = build_theta(ptform(z))
    return -(loglikelihood(theta) + logprior(z))


neg_logpost(z)

# %%
from jaxopt import BFGS

solver = BFGS(
    fun=neg_logpost,
    maxiter=100,  # for quick illustration purposes; is good enough
    tol=1e-4,
)

z0 = jnp.zeros(ndim)

with time_this() as elapsed:  # O(10) min for 1000 training samples
    res = solver.run(z0)
    z_map = res.params

print("iters:", res.state.iter_num)
print("final neg logpost:", res.state.value)
print("error:", res.state.error)

# %%
theta_map = build_theta(ptform(z_map))

theta_map

# %%
cpu = jax.devices("cpu")[0]

with jax.default_device(cpu):  # avoid GPU OOM
    H = jax.hessian(neg_logpost)(z_map)
    H = 0.5 * (H + H.T)
    cov_z = jnp.linalg.inv(H)

# q(z) = N(z_map, cov_z)

# %%


def sample_theta(key, n_samples):
    L = jnp.linalg.cholesky(cov_z)
    eps = jax.random.normal(key, shape=(n_samples, z_map.shape[0]))
    z_samples = z_map + eps @ L.T
    theta_samples = jax.vmap(lambda z: build_theta(ptform(z)))(z_samples)
    return theta_samples, z_samples


theta_samples, z_samples = sample_theta(jax.random.PRNGKey(0), 10)
print(theta_samples)
print(jax.vmap(loglikelihood)(theta_samples))


def laplace_log_evidence(z_map, H):
    d = z_map.shape[0]
    sign, logdetH = jnp.linalg.slogdet(H)
    assert sign > 0
    logpost_map = -neg_logpost(z_map)
    return logpost_map + 0.5 * d * jnp.log(2.0 * jnp.pi) - 0.5 * logdetH


logZ = laplace_log_evidence(z_map, H)
print(logZ)

# %%
quad = jnp.einsum("ni,ij,nj->n", z_samples - z_map, H, z_samples - z_map)
print(jnp.mean(quad), "expected", ndim)


# %%
def posterior_diag_report(theta_map, du_tau, eig_subset=24, seed=0):
    map_gp = build_gp(theta_map)

    @jax.jit
    def one(dui):
        gp = map_gp.condition(dui).gp
        mu = gp.mu
        L = gp.cov_root  # shape (M, r)

        tr = jnp.sum(L * L)  # trace

        # singular values of L -> eigenvalues of Sigma = s^2
        s = jnp.linalg.svd(L, compute_uv=False)
        logpdet = jnp.sum(jnp.log(s * s + 1e-30))
        rank = jnp.sum(s > 1e-8)

        return mu, L, tr, logpdet, rank

    mu_all, L_all, tr_all, logpdet_all, rank_all = jax.vmap(one)(du_tau)

    mu_all_np = np.array(mu_all)
    tr_all_np = np.array(tr_all)
    logpdet_all_np = np.array(logpdet_all)
    rank_all_np = np.array(rank_all)

    mu_mean = mu_all_np.mean(axis=0)
    tr_cov_mu = np.mean(np.sum((mu_all_np - mu_mean) ** 2, axis=1))
    tr_sig = tr_all_np.mean()
    rho = tr_sig / max(tr_cov_mu, 1e-30)

    def qstats(x):
        qs = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
        return dict(
            q0=qs[0],
            q10=qs[1],
            q50=qs[2],
            q90=qs[3],
            q100=qs[4],
            mean=float(np.mean(x)),
        )

    print("posterior trace(Sigma):", qstats(tr_all_np))
    print("posterior log pseudo-det:", qstats(logpdet_all_np))
    print("posterior rank:", qstats(rank_all_np))
    print("trace(Cov(mu)):", float(tr_cov_mu))
    print("rho = E[tr(Sigma_i)] / tr(Cov(mu_i)):", float(rho))

    # eigen spectra on subset
    rng = np.random.default_rng(seed)
    order = np.argsort(tr_all_np)
    pick = np.linspace(0, len(order) - 1, min(eig_subset, len(order))).astype(
        int
    )
    idxs = order[pick]

    eigs = []
    for idx in idxs:
        L = np.array(L_all[idx])
        s = np.linalg.svd(L, compute_uv=False)
        eigs.append(s * s)

    eigs = np.stack(eigs, axis=0)
    print("subset eigval median (top 10):", np.median(eigs[:, :10], axis=0))
    print("subset eigval q10 (top 10):", np.quantile(eigs[:, :10], 0.1, axis=0))
    print("subset eigval q90 (top 10):", np.quantile(eigs[:, :10], 0.9, axis=0))

    rho


# %%
# export
rho = posterior_diag_report(theta_map, du_tau, eig_subset=24)


# %%

# crude calibration
w = np.random.randn(Phi.shape[1])
u = Phi @ w
scale = np.mean(u**2)
sigma_w2 = constants.NOISE_FLOOR_POWER / scale

print("Calibrated prior weight variance:", sigma_w2)

# %%
### PPCA mixture inference
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


@jax.jit
def _woodbury_logpdf_isotropic(x, m, sigma2, W):
    """
    log N(x | m, sigma2 I + W W^T)
    x,m: [M]
    W: [M, r]
    """
    M = x.shape[0]
    xm = x - m
    r = W.shape[1]

    WtW = W.T @ W
    S = jnp.eye(r, dtype=x.dtype) + WtW / sigma2
    rhs = (W.T @ xm) / sigma2
    y = jax.scipy.linalg.solve(S, rhs, assume_a="pos")

    quad = (xm @ xm) / sigma2 - (rhs @ y)
    sign, logdetS = jnp.linalg.slogdet(S)
    logdetA = M * jnp.log(sigma2) + logdetS

    return -0.5 * (M * jnp.log(2.0 * jnp.pi) + logdetA + quad)


def choose_global_rank(mu, sigma_z2, rmax=128):
    mu = np.array(mu)
    mu0 = mu - mu.mean(axis=0, keepdims=True)
    C = (mu0.T @ mu0) / max(mu.shape[0] - 1, 1)
    lam = np.linalg.eigvalsh(C)[::-1]
    r = int(np.sum(lam > sigma_z2))
    r = max(1, min(r, rmax))
    return r, lam


def fit_mix_ppca(mu, K=4, r=64, sigma_z2=1e-8, n_iter=50, seed=0, verbose=True):
    """
    Standard mixture PPCA on x_i = mu_i.
    Cov_k = W_k W_k^T + sigma_z2 I, with fixed rank r for all k.
    """
    mu = jnp.asarray(mu)
    N, M = mu.shape

    key = jax.random.PRNGKey(seed)
    idx = jax.random.choice(key, N, shape=(K,), replace=False)
    m = mu[idx, :]  # [K, M]
    W = jnp.zeros((K, M, r), dtype=mu.dtype)
    logpi = jnp.zeros((K,), dtype=mu.dtype) - jnp.log(K)

    @jax.jit
    def e_step(m, W, logpi):
        def one_k(k):
            mk = m[k, :]
            Wk = W[k, :, :]
            lpi = logpi[k]

            def one_i(i):
                return lpi + _woodbury_logpdf_isotropic(
                    mu[i, :], mk, sigma_z2, Wk
                )

            return jax.vmap(one_i)(jnp.arange(N))

        logp = jax.vmap(one_k)(jnp.arange(K)).T  # [N, K]
        logZ = logsumexp(logp, axis=1, keepdims=True)
        log_r = logp - logZ
        rnk = jnp.exp(log_r)
        ll = jnp.sum(logZ)
        return rnk, ll

    def m_step(rnk):
        Nk = jnp.sum(rnk, axis=0) + 1e-30
        logpi_new = jnp.log(Nk) - jnp.log(jnp.sum(Nk))
        m_new = (rnk.T @ mu) / Nk[:, None]

        # W_k from weighted covariance eigen-decomp, classic PPCA with fixed sigma_z2
        def one_k(k):
            rk = rnk[:, k]
            mk = m_new[k, :]
            xmc = mu - mk[None, :]
            C = (xmc.T * rk) @ xmc / Nk[k]
            C = 0.5 * (C + C.T)

            lam, Q = jnp.linalg.eigh(C)  # ascending
            lam = lam[::-1]
            Q = Q[:, ::-1]

            lam_r = lam[:r]
            Q_r = Q[:, :r]
            # PPCA: W = Q_r diag(sqrt(max(lam_r - sigma2, 0)))
            gain = jnp.sqrt(jnp.maximum(lam_r - sigma_z2, 0.0))
            Wk = Q_r * gain[None, :]
            return Wk

        W_new = jax.vmap(one_k)(jnp.arange(K))
        return m_new, W_new, logpi_new, Nk

    ll_hist = []
    for it in range(n_iter):
        rnk, ll = e_step(m, W, logpi)
        m, W, logpi, Nk = m_step(rnk)
        ll_hist.append(float(ll))
        if verbose:
            pis = np.array(jnp.exp(logpi))
            # "effective rank" per component: number of nontrivial columns
            eff = [
                int(jnp.sum(jnp.linalg.norm(W[k], axis=0) > 1e-12))
                for k in range(K)
            ]
            print(
                f"iter {it:03d}  ll {float(ll):.3f}  pi {pis}  eff_rank {eff}"
            )

    return {
        "m": m,
        "W": W,
        "logpi": logpi,
        "sigma_z2": float(sigma_z2),
        "ll_hist": ll_hist,
    }


# %%
# %%
map_gp = build_gp(theta_map)


@jax.jit
def one_mu(dui):
    return map_gp.condition(dui).gp.mu


mu_all = jax.vmap(one_mu)(du_tau)  # [N, M]


# %%


def calibrate_sigma_z2(Phi, sigma_u, nprobe=4096, seed=0):
    """
    Map an isotropic noise sigma_u^2 in u-space to an approximate isotropic
    sigma_z^2 in z-space by matching average energy of Phi w.

    We use: E||Phi w||^2 = s * E||w||^2, so sigma_z^2 ~ sigma_u^2 / s.
    """
    rng = np.random.default_rng(seed)
    M = Phi.shape[1]
    W = rng.standard_normal((nprobe, M))
    U = W @ np.array(Phi).T  # [nprobe, dim_u]
    s = float(np.mean(U * U) / np.mean(W * W))
    return float((sigma_u * sigma_u) / max(s, 1e-30))


sigma_u_like = float(theta_map["sigma_noise"])
sigma_u_floor = 1e-3  # -60 dB amplitude

sigma_z2_like = calibrate_sigma_z2(Phi, sigma_u_like)
sigma_z2_floor = calibrate_sigma_z2(Phi, sigma_u_floor)
sigma_z2 = sigma_z2_like + sigma_z2_floor

r, lam = choose_global_rank(mu_all, sigma_z2, rmax=128)
print("sigma_z2", sigma_z2, "chosen r", r)
# %%
fit = fit_mix_ppca(mu_all, K=4, r=r, sigma_z2=sigma_z2, n_iter=10, seed=0)


# %%
def sample_mix_amp(fit, nsamples=1, key=jax.random.PRNGKey(0)):
    """
    Sample from the fitted mixture amplitude prior.

    Returns:
      w:      [nsamples, M] samples in z / weight space
      comp:   [nsamples] component indices
    """
    m = fit["m"]  # [K, M]
    B = fit["W"]  # [K, M, Rk]
    logpi = fit["logpi"]  # [K]
    sigma_z2 = fit["sigma_z2"]

    K, M = m.shape
    R = B.shape[2]

    key, k1, k2 = jax.random.split(key, 3)

    # sample component indices
    comp = jax.random.categorical(k1, logpi, shape=(nsamples,))

    # standard normals for low-rank + isotropic parts
    eps_low = jax.random.normal(k2, shape=(nsamples, R))
    key, k3 = jax.random.split(key)
    eps_iso = jax.random.normal(k3, shape=(nsamples, M))

    def one(i):
        k = comp[i]
        mk = m[k]
        Bk = B[k]  # [M, R]
        return mk + Bk @ eps_low[i] + jnp.sqrt(sigma_z2) * eps_iso[i]

    w = jax.vmap(one)(jnp.arange(nsamples))
    return w, comp


def sample_u_from_w(w, Phi, sigma_u_floor=1e-3, key=jax.random.PRNGKey(1)):
    """
    Map z-space samples to waveform space.

    w:   [N, M]
    Phi: [dim_u, M]
    """
    u = w @ Phi.T  # [N, dim_u]

    if sigma_u_floor is not None and sigma_u_floor > 0.0:
        key, k = jax.random.split(key)
        eps = sigma_u_floor * jax.random.normal(k, u.shape)
        u = u + eps

    return u


# %%


# sample waveforms
K = fit["m"].shape[0]

nsamples = 1 * K
w_samp, comp = sample_mix_amp(fit, nsamples=nsamples, key=vk())
u_samp = sample_u_from_w(w_samp, Phi, sigma_u_floor=None, key=None)

comp_np = np.array(comp)
u_np = np.array(u_samp)

fig, axes = plt.subplots(K, 1, figsize=(8, 2.0 * K), sharex=True)

if K == 1:
    axes = [axes]

for k in range(K):
    ax = axes[k]
    idx = np.where(comp_np == k)[0]

    if len(idx) == 0:
        ax.set_title(f"component {k} (no samples)")
        continue

    uu = jnp.cumsum(u_np[idx], axis=1).T * dtau

    # ax.plot(tau_grid, u_np[idx].T, alpha=0.7)
    ax.plot(tau_grid, uu, alpha=0.7)
    ax.set_title(f"component {k}   (n={len(idx)})")
    ax.set_ylabel("amplitude")

axes[-1].set_xlabel("tau")
fig.suptitle(
    "Samples from learned surrogate prior (grouped by mixture component)",
    y=0.98,
)
plt.tight_layout()
plt.show()

# %%
# %%
import jax
import jax.numpy as jnp
from tqdm import tqdm

LOG10 = jnp.log(10.0)


@jax.jit
def _logpdf_iso_lowrank_batch(y, mu, sigma2, U):
    """
    y, mu: [B, T]
    U:      [B, T, r]
    sigma2: scalar
    returns logpdf: [B]
    """
    B, T = y.shape
    r = U.shape[-1]

    ym = y - mu  # [B, T]
    UtU = jnp.einsum("btr,bts->brs", U, U)  # [B, r, r]
    S = jnp.eye(r)[None, :, :] + UtU / sigma2

    rhs = jnp.einsum("btr,bt->br", U, ym) / sigma2
    sol = jax.vmap(lambda A, b: jnp.linalg.solve(A, b))(S, rhs)

    quad = jnp.sum(ym * ym, axis=1) / sigma2 - jnp.sum(rhs * sol, axis=1)

    logdetS = jnp.linalg.slogdet(S)[1]
    logdetA = T * jnp.log(sigma2) + logdetS

    return -0.5 * (T * jnp.log(2.0 * jnp.pi) + logdetA + quad)


def make_logp_u_fn(pack, fit, sigma_u2_floor):
    m = fit["m"]  # [K, M]
    W = fit["W"]  # [K, M, r]
    logpi = fit["logpi"]
    sigma_z2 = fit["sigma_z2"]
    K = m.shape[0]

    def logp_u_batch(tau, du):
        """
        tau: [B, T]
        du:  [B, T]
        returns log p(u): [B]
        """
        Phi = jax.vmap(lambda t: jax.vmap(pack.compute_phi)(t))(
            tau
        )  # [B, T, M]

        # isotropic approx for sigma_z2 * Phi Phi^T
        c = jnp.mean(jnp.sum(Phi * Phi, axis=2))
        sigma2_eff = (
            sigma_u2_floor + float(theta_map["sigma_noise"]) ** 2 + sigma_z2 * c
        )

        def one_k(k):
            mu_u = Phi @ m[k]  # [B, T]
            U = Phi @ W[k]  # [B, T, r]
            return logpi[k] + _logpdf_iso_lowrank_batch(du, mu_u, sigma2_eff, U)

        log_terms = jax.vmap(one_k)(jnp.arange(K))  # [K, B]
        return logsumexp(log_terms, axis=0)  # [B]

    return jax.jit(logp_u_batch)


def kl_bans_fast(test_samples, fit, theta_map):
    pack = build_kernel(theta_map)
    sigma_u2_floor = float(constants.NOISE_FLOOR_POWER)

    # bucket by length
    buckets = {}
    for s in test_samples:
        T = len(s["tau"])
        buckets.setdefault(T, []).append(s)

    total_kl = 0.0
    total_n = 0

    for T, samples in tqdm(buckets.items()):
        B = len(samples)

        tau = jnp.stack([s["tau"] for s in samples])  # [B, T]
        du = jnp.stack([s["du"] for s in samples])  # [B, T]
        logq = jnp.array([s["log_prob_u"] for s in samples])  # [B]

        logp_fn = make_logp_u_fn(pack, fit, sigma_u2_floor)
        logp = logp_fn(tau, du)

        total_kl += jnp.sum((logq - logp) / LOG10)
        total_n += B

    return float(total_kl / total_n)


# usage:
kl_bans = kl_bans_fast(test_samples[:100], fit, theta_map)
print("KL divergence to BANs (fast):", kl_bans, "bans")
