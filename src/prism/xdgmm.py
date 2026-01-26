"""Background Extreme Deconvolution style EM for y = x + eps, eps ~ N(0, S_i) known per-point
Supports a fixed background component at k=0 (mu0, Sigma0 fixed), only pi learned, intended to soak up outliers.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.stats
from flax import struct
from gpjax.scan import vscan
from tqdm import tqdm

from utils.jax import safe_cholesky, symmetrize


def logpdf_mvn_chol(x, mu, L):
    """
    log N(x | mu, LL^T) for single x, single mu, single chol.
    """
    d = x.shape[-1]
    y = jsp.linalg.solve_triangular(L, x - mu, lower=True)
    quad = jnp.sum(y * y)
    logdet = jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2 * jnp.pi) + 2 * logdet + quad)


def fit_background_blob(m, S, alpha_quantile=0.99):
    """
    Compute the KL-optimal Gaussian N(mu0, Sigma0) approximating
    q(x) = 1/N sum_n N(m_n, S_n), then inflate to a quantile envelope.

    Returns mu0, Sigma0_enveloped.
    """
    N, d = m.shape

    mu0 = jnp.mean(m, axis=0)

    # Covariance of mixture of Gaussians
    centered = m - mu0
    Sigma0 = jnp.mean(
        S + jnp.einsum("ni,nj->nij", centered, centered),
        axis=0,
    )

    # Quantile envelope inflation
    # Mahalanobis distances under base Sigma0
    L0 = jnp.linalg.cholesky(Sigma0)
    y = jsp.linalg.solve_triangular(L0, centered.T, lower=True).T
    d2 = jnp.sum(y * y, axis=1)

    chi2_q = scipy.stats.chi2.ppf(alpha_quantile, d)
    alpha = jnp.quantile(d2, alpha_quantile) / chi2_q

    Sigma0 = alpha * Sigma0
    return mu0, Sigma0


@struct.dataclass
class GMMParams:
    mu: jnp.ndarray  # (K, d)  free components only
    cov: jnp.ndarray  # (K, d, d)
    logits: jnp.ndarray  # (K+1,) includes background weight


def e_step(m, S, params, mu0, cov0, jitter):
    """
    Extreme deconvolution E-step with fixed background component.

    Returns:
      r    : (N, K+1) responsibilities
      m_nk : (N, K, d) posterior means for free components
      V_nk : (N, K, d, d) posterior covariances for free components
      ll   : scalar log-likelihood
    """
    N, d = m.shape
    K = params.mu.shape[0]

    pi = jax.nn.softmax(params.logits)

    # Allocate log-terms
    log_terms = []

    # Background component k=0
    C0 = cov0[None, :, :] + S
    L0 = jax.vmap(safe_cholesky, in_axes=(0, None))(C0, jitter)
    lp0 = jax.vmap(logpdf_mvn_chol, in_axes=(0, None, 0))(m, mu0, L0)
    log_terms.append(lp0 + jnp.log(pi[0] + 1e-32))

    # Free components
    m_post = []
    V_post = []

    for k in range(K):
        muk = params.mu[k]
        Sigk = params.cov[k]

        C = Sigk[None, :, :] + S
        C = symmetrize(C)
        L = jax.vmap(safe_cholesky, in_axes=(0, None))(C, jitter)

        lp = jax.vmap(logpdf_mvn_chol, in_axes=(0, None, 0))(m, muk, L)
        log_terms.append(lp + jnp.log(pi[k + 1] + 1e-32))

        # XD posterior moments
        # W = Sigk @ inv(C)
        Ci = jax.vmap(jnp.linalg.inv)(C)
        W = jnp.einsum("ij,njk->nik", Sigk, Ci)

        mnk = muk + jnp.einsum("nij,nj->ni", W, m - muk)
        Vnk = Sigk - jnp.einsum("nij,jk->nik", W, Sigk)
        Vnk = symmetrize(Vnk)

        m_post.append(mnk)
        V_post.append(symmetrize(Vnk))

    log_terms = jnp.stack(log_terms, axis=1)  # (N, K+1)

    log_norm = jsp.special.logsumexp(log_terms, axis=1)
    r = jnp.exp(log_terms - log_norm[:, None])
    ll = jnp.sum(log_norm)

    m_post = jnp.stack(m_post, axis=1)  # (N, K, d)
    V_post = jnp.stack(V_post, axis=1)  # (N, K, d, d)

    return r, m_post, V_post, ll


def m_step(m_post, V_post, r, cov_floor):
    """
    Update only free components (k=1..K).
    """
    N, K, d = m_post.shape

    r_free = r[:, 1:]  # (N, K)
    Nk = jnp.sum(r_free, axis=0) + 1e-16  # (K,)

    mu = jnp.einsum("nk,nkd->kd", r_free, m_post) / Nk[:, None]

    # Covariance update (XD)
    centered = m_post - mu[None, :, :]
    cov = (
        jnp.einsum("nk,nkij->kij", r_free, V_post)
        + jnp.einsum("nk,nki,nkj->kij", r_free, centered, centered)
    ) / Nk[:, None, None]

    # Floor + symmetrizemetrize
    dI = cov_floor * jnp.eye(d)
    cov = symmetrize(cov) + dI

    # Weights (including background)
    pi = jnp.mean(r, axis=0)
    logits = jnp.log(pi + 1e-32)

    return GMMParams(mu=mu, cov=cov, logits=logits)


def em_step(carry, _):
    params, m, S, mu0, cov0, jitter, cov_floor = carry

    r, m_post, V_post, ll = e_step(m, S, params, mu0, cov0, jitter)
    new_params = m_step(m_post, V_post, r, cov_floor)

    carry = (new_params, m, S, mu0, cov0, jitter, cov_floor)
    return carry, ll


def fit_xdgmm(
    m,
    S,
    K,
    n_iter=100,
    jitter=1e-6,
    cov_floor=1e-6,
    alpha_quantile=0.99,
    init_mu=None,
    init_cov=None,
    verbose=False,
):
    """
    Extreme Deconvolution GMM with a fixed "background radiation" component.

    This implements:
      q(x) = 1/N sum_n N(m_n, S_n)
      p(x) = pi0 N(mu0, alpha Sigma0) + sum_{k=1}^K pi_k N(mu_k, Sigma_k)

    where (mu0, Sigma0) is the KL projection of q onto Gaussians,
    inflated to an alpha-quantile envelope.

    Notes:
    - This is standard XD (Bovy et al.) for components k>=1.
    - Component 0 is fixed and absorbs outliers / off-manifold mass.
    - Responsibilities r_{n0} act as an automatic outlier detector.
    """

    N, d = m.shape

    # Background component
    mu0, cov0 = fit_background_blob(m, S, alpha_quantile)

    # Init free components
    if init_mu is None:
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, N, (K,), replace=False)
        mu = m[idx]
    else:
        mu = init_mu

    if init_cov is None:
        cov = jnp.tile(jnp.eye(d)[None, :, :], (K, 1, 1))
    else:
        cov = init_cov

    # Init weights: small background, uniform rest
    logits = jnp.zeros(K + 1)
    logits = logits.at[0].set(-2.0)

    params = GMMParams(mu=mu, cov=cov, logits=logits)

    carry = (params, m, S, mu0, cov0, jitter, cov_floor)

    scan = vscan if verbose else jax.lax.scan

    (params, *_), history = scan(em_step, carry, None, length=n_iter)

    return params, history, (mu0, cov0)


def logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def chol_logdet(L):
    return 2.0 * np.sum(np.log(np.diag(L)))


def gmm_data_loglikelihoods(
    f_list,  # list of (T,) arrays, length Ntest
    Psi_list,  # list of (T,D) arrays, same length
    pi,  # (K,)
    mu_k,  # (K,D)
    Sigma_k,  # (K,D,D)
    obs_std,  # scalar
    noise_floor=1e-3,
):
    sigma = max(float(obs_std), float(noise_floor))
    sigma2 = sigma * sigma
    inv_sigma2 = 1.0 / sigma2
    inv_sigma4 = inv_sigma2 * inv_sigma2

    K, D = mu_k.shape
    Ntest = len(f_list)

    # precompute per k
    Sig_inv = np.empty_like(Sigma_k)
    logdet_Sig = np.empty((K,), dtype=float)
    for k in range(K):
        L = np.linalg.cholesky(Sigma_k[k])
        logdet_Sig[k] = chol_logdet(L)
        Sig_inv[k] = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D)))

    # precompute per i
    G = []
    t = []
    s = []
    T_list = []
    for f, Psi in zip(f_list, Psi_list):
        G.append(Psi.T @ Psi)
        t.append(Psi.T @ f)
        s.append(float(f @ f))
        T_list.append(f.shape[0])

    lls = []
    for i in tqdm(range(Ntest)):
        Gi = G[i]
        ti = t[i]
        si = s[i]
        Ti = T_list[i]

        lps = np.empty((K,), dtype=float)
        for k in range(K):
            muk = mu_k[k]
            bik = ti - Gi @ muk
            r2 = si - 2.0 * (muk @ ti) + (muk @ (Gi @ muk))

            Mik = Sig_inv[k] + inv_sigma2 * Gi
            LM = np.linalg.cholesky(Mik)
            logdet_M = chol_logdet(LM)

            x = np.linalg.solve(LM, bik)
            quad = inv_sigma2 * r2 - inv_sigma4 * (x @ x)

            logdet_C = Ti * np.log(sigma2) + logdet_Sig[k] + logdet_M
            lps[k] = np.log(pi[k]) - 0.5 * (
                Ti * np.log(2.0 * np.pi) + logdet_C + quad
            )

        ll = logsumexp(lps)
        lls.append(ll)

    return np.array(lls)
