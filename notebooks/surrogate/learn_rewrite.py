# %%

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from gfm.ack import DiagonalTACK
from pack import PACK
from utils import constants

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)

# ============================================================
# user hooks: warp / dewarp
# ============================================================


def warp_time(t_ms, sample):
    # sample can be pytree; default uses period
    return t_ms / sample["period_ms"]


def dewarp_time(tau, sample):
    return tau * sample["period_ms"]


# ============================================================
# data ingest (no resampling)
# ============================================================


def make_samples(lf_samples):
    out = []
    for s in lf_samples:
        logq = s["log_prob_u"]
        if not np.isfinite(logq):
            continue
        period_ms = float(s["p"]["T0"])
        t_ms = np.array(s["t"], dtype=np.float64)
        du = np.array(s["u"], dtype=np.float64)

        sample = {
            "period_ms": period_ms,
            "t_ms": t_ms,
            "du": du,
            "log_prob_u": float(logq),
        }
        sample["tau"] = np.array(warp_time(t_ms, sample), dtype=np.float64)
        out.append(sample)
    return out


def bucket_by_len(samples):
    buckets = {}
    for s in samples:
        T = int(s["du"].shape[0])
        buckets.setdefault(T, []).append(s)
    return buckets


# ============================================================
# kernel / BLR
# ============================================================


def build_theta(x, J):
    return {
        "sigma_noise": x[0],
        "sigma_a": x[1],
        "sigma_b": x[2],
        "tc": x[3],
        "center": x[4],
    }


def build_kernel(theta, d=0, normalized=True):
    tack = DiagonalTACK(
        d=d,
        normalized=normalized,
        center=theta["center"],
        sigma_b=theta["sigma_b"],
        sigma_c=1.0,
    )
    pack = PACK(
        tack,
        period=1.0,
        t1=0.0,
        t2=theta["tc"],
        J=J,
    )
    return theta["sigma_a"] ** 2 * pack


# ============================================================
# posterior mean extraction: mu_i = E[w | du_i]
# (per-sample Phi(tau(t)) so variable warp OK)
# ============================================================


@jax.jit
def _phi_from_tau(pack, tau_1d):
    # tau_1d: [T]
    return jax.vmap(pack.compute_phi)(tau_1d)  # [T, M]


@jax.jit
def _posterior_mu_one(pack, cov_root, sigma2, tau, y):
    """
    BLR weights:
      w ~ N(0, cov_root cov_root^T)
      y = Phi w + eps, eps ~ N(0, sigma2 I)
    Return posterior mean of w given (tau,y).
    """
    Phi = _phi_from_tau(pack, tau)  # [T, M]
    # work in latent z space with cov_root (M,R)
    # A = cov_root^T Phi^T Phi cov_root + sigma2 I
    # b = cov_root^T Phi^T y
    PTy = Phi.T @ y  # [M]
    b = cov_root.T @ PTy  # [R]
    A = cov_root.T @ (Phi.T @ (Phi @ cov_root))  # [R,R]
    A = 0.5 * (A + A.T)

    # stabilize noise with NOISE_FLOOR_POWER in variance
    sigma2_eff = sigma2 + constants.NOISE_FLOOR_POWER
    A = A + sigma2_eff * jnp.eye(A.shape[0], dtype=A.dtype)

    z = jnp.linalg.solve(A, b)  # [R]
    mu = cov_root @ z  # [M]
    return mu


def mus_from_samples(theta, samples, d=0, normalized=True):
    pack = build_kernel(theta, d=d, normalized=normalized)
    cov_root = pack.compute_weights_root()  # [M,R]
    sigma2 = float(theta["sigma_noise"]) ** 2

    buckets = bucket_by_len(samples)
    mu_list = []

    for T, ss in buckets.items():
        tau = jnp.stack([jnp.asarray(s["tau"]) for s in ss])  # [B,T]
        y = jnp.stack([jnp.asarray(s["du"]) for s in ss])  # [B,T]

        # vmap over batch
        muB = jax.vmap(
            lambda tt, yy: _posterior_mu_one(pack, cov_root, sigma2, tt, yy)
        )(tau, y)
        mu_list.append(np.array(muB))

    mu_all = np.concatenate(mu_list, axis=0)  # [N,M]
    return jnp.asarray(mu_all)


# ============================================================
# mixture PPCA on mu (fixed rank across components)
# ============================================================


@jax.jit
def _logpdf_ppca_x(x, m, sigma2, W):
    # log N(x | m, sigma2 I + W W^T) with Woodbury
    M = x.shape[0]
    xm = x - m
    r = W.shape[1]
    WtW = W.T @ W
    S = jnp.eye(r, dtype=x.dtype) + WtW / sigma2
    rhs = (W.T @ xm) / sigma2
    sol = jnp.linalg.solve(S, rhs)
    quad = (xm @ xm) / sigma2 - (rhs @ sol)
    logdetS = jnp.linalg.slogdet(S)[1]
    logdetA = M * jnp.log(sigma2) + logdetS
    return -0.5 * (M * jnp.log(2.0 * jnp.pi) + logdetA + quad)


def choose_global_rank(mu, sigma_z2, rmax=128):
    mu = np.array(mu)
    mu0 = mu - mu.mean(axis=0, keepdims=True)
    C = (mu0.T @ mu0) / max(mu.shape[0] - 1, 1)
    lam = np.linalg.eigvalsh(C)[::-1]
    r = int(np.sum(lam > sigma_z2))
    return max(1, min(r, rmax))


def fit_mix_ppca(mu, K, r, sigma_z2, n_iter=30, seed=0):
    mu = jnp.asarray(mu)
    N, M = mu.shape
    key = jax.random.PRNGKey(seed)
    idx = jax.random.choice(key, N, shape=(K,), replace=False)
    m = mu[idx, :]
    W = jnp.zeros((K, M, r), dtype=mu.dtype)
    logpi = jnp.zeros((K,), dtype=mu.dtype) - jnp.log(K)

    @jax.jit
    def e_step(m, W, logpi):
        def one_k(k):
            mk = m[k]
            Wk = W[k]
            lpi = logpi[k]
            return jax.vmap(
                lambda x: lpi + _logpdf_ppca_x(x, mk, sigma_z2, Wk)
            )(mu)

        logp = jax.vmap(one_k)(jnp.arange(K)).T
        logZ = logsumexp(logp, axis=1, keepdims=True)
        rnk = jnp.exp(logp - logZ)
        ll = jnp.sum(logZ)
        return rnk, ll

    def m_step(rnk):
        Nk = jnp.sum(rnk, axis=0) + 1e-30
        logpi_new = jnp.log(Nk) - jnp.log(jnp.sum(Nk))
        m_new = (rnk.T @ mu) / Nk[:, None]

        def one_k(k):
            rk = rnk[:, k]
            mk = m_new[k]
            xmc = mu - mk[None, :]
            C = (xmc.T * rk) @ xmc / Nk[k]
            C = 0.5 * (C + C.T)
            lam, Q = jnp.linalg.eigh(C)
            lam = lam[::-1]
            Q = Q[:, ::-1]
            lam_r = lam[:r]
            Q_r = Q[:, :r]
            gain = jnp.sqrt(jnp.maximum(lam_r - sigma_z2, 0.0))
            return Q_r * gain[None, :]

        W_new = jax.vmap(one_k)(jnp.arange(K))
        return m_new, W_new, logpi_new

    for _ in range(n_iter):
        rnk, _ = e_step(m, W, logpi)
        m, W, logpi = m_step(rnk)

    return {"m": m, "W": W, "logpi": logpi, "sigma_z2": float(sigma_z2)}


# ============================================================
# exact log p(du | t) for mixture, no tau-vs-t mismatch
# covariance computed in t-space with Cholesky (exact, no hacks)
# ============================================================


@jax.jit
def _logpdf_full(y, mean, K):
    # y, mean: [T], K: [T,T] SPD
    T = y.shape[0]
    L = jnp.linalg.cholesky(K)
    z = jax.scipy.linalg.solve_triangular(L, y - mean, lower=True)
    quad = jnp.dot(z, z)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (T * jnp.log(2.0 * jnp.pi) + logdet + quad)


def make_logp_u_bucket_fn(theta, fit, T, d=0, normalized=True):
    pack = build_kernel(theta, d=d, normalized=normalized)
    sigma_noise2 = float(theta["sigma_noise"]) ** 2

    # total observation variance floor in u-space
    sigma_u2 = constants.NOISE_FLOOR_POWER + sigma_noise2

    m = fit["m"]  # [K,M]
    W = fit["W"]  # [K,M,r]
    logpi = fit["logpi"]
    sigma_z2 = fit["sigma_z2"]

    @jax.jit
    def logp_batch(tau, y):
        # tau,y: [B,T]
        Phi = jax.vmap(lambda t: jax.vmap(pack.compute_phi)(t))(tau)  # [B,T,M]

        def one_sample(Phi_i, y_i):
            # Build exact component covariance in t-space:
            # K = sigma_u2 I + Phi (W W^T + sigma_z2 I) Phi^T
            # = sigma_u2 I + (Phi W)(Phi W)^T + sigma_z2 (Phi)(Phi)^T
            def one_k(k):
                mk = m[k]
                Wk = W[k]
                mean = Phi_i @ mk  # [T]
                UW = Phi_i @ Wk  # [T,r]
                # low-rank terms to full covariance (exact, but cheap enough at T~512)
                Kmat = sigma_u2 * jnp.eye(T, dtype=y.dtype)
                Kmat = Kmat + UW @ UW.T
                Kmat = Kmat + sigma_z2 * (Phi_i @ Phi_i.T)
                return logpi[k] + _logpdf_full(y_i, mean, Kmat)

            terms = jax.vmap(one_k)(jnp.arange(m.shape[0]))
            return logsumexp(terms)

        return jax.vmap(one_sample)(Phi, y)  # [B]

    return logp_batch


from tqdm import tqdm


def kl_bans_test(theta, fit, test_samples, d=0, normalized=True):
    LOG10 = np.log(10.0)
    buckets = bucket_by_len(test_samples)

    total = 0.0
    n = 0
    for T, ss in tqdm(buckets.items()):
        tau = jnp.stack([jnp.asarray(s["tau"]) for s in ss])
        y = jnp.stack([jnp.asarray(s["du"]) for s in ss])
        logq = jnp.array([s["log_prob_u"] for s in ss])

        logp_fn = make_logp_u_bucket_fn(
            theta, fit, T, d=d, normalized=normalized
        )
        logp = logp_fn(tau, y)

        total += float(jnp.sum((logq - logp) / LOG10))
        n += len(ss)

    return total / max(n, 1)


# %%

# ============================================================
# main
# ============================================================


NUM_TRAIN = 500
NUM_TEST = 100
K = 4
J = 512
d = 0
normalized = False
seed = 0

# %%

lf = get_lf_samples()
train = make_samples(lf[:NUM_TRAIN])
test = make_samples(lf[NUM_TRAIN : NUM_TRAIN + NUM_TEST])

# your existing MAP fit code can stay mostly as-is, but must not rely on common tau grid.
# Here we assume you already have theta_map from your optimizer.
# %%
Array = jnp.array
float64 = jnp.float64

theta_map = {
    "sigma_noise": Array(0.03677499, dtype=float64),
    "sigma_a": Array(2.52370113, dtype=float64),
    "sigma_b": Array(0.66658194, dtype=float64),
    "tc": Array(0.99992432, dtype=float64),
    "center": Array(0.99978166, dtype=float64),
}


# compute posterior means (no storing L_i)
mu_train = mus_from_samples(theta_map, train, d=d, normalized=normalized)

# %%

# choose sigma_z2 using theta_map sigma_noise and floor; no magic 1e-3
# map u-space amplitude floor to z-space variance crudely using Phi energy on a probe grid
# (kept minimal; you can reuse your calibrate function if you like)
sigma_u_like = float(theta_map["sigma_noise"])
sigma_u_floor = float(np.sqrt(constants.NOISE_FLOOR_POWER))
# simple conservative z-variance: just use u-variance directly as z-variance scale
sigma_z2 = sigma_u_like**2 + sigma_u_floor**2

r = choose_global_rank(mu_train, sigma_z2, rmax=128)
fit = fit_mix_ppca(mu_train, K=4, r=r, sigma_z2=sigma_z2, n_iter=50, seed=seed)

# %%

kl_bans = kl_bans_test(theta_map, fit, test[:200], d=d, normalized=normalized)
print("D_KL(q||p) on test (bans, log10):", kl_bans)

# %%
