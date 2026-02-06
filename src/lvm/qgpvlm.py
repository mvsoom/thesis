import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from joblib import Parallel, delayed
from scipy.special import gammaln

from lvm.bgplvm import BGPLVMPosterior
from lvm.xdgmm import GMMFit


@struct.dataclass
class QGPVLM:
    K: int
    M: int

    gmm: GMMFit
    lvm: BGPLVMPosterior

    psi: callable  # PRISM basis
    whiten: callable  # Global whitening ...
    unwhiten: callable  # ... and unwhitening transforms

    pi: jnp.ndarray  # (K,) mixture weights
    mu: jnp.ndarray  # (K, M) component means
    cov: jnp.ndarray  # (K, M, M) component covariances


def make_qgpvlm(
    latent_gmm: GMMFit,
    latent_bgplvm: BGPLVMPosterior,
    psi: callable,
    whiten: callable,
    unwhiten: callable,
) -> QGPVLM:
    y_pi, y_mu, y_cov = latent_bgplvm.forward_x_gmm(
        latent_gmm.pi, latent_gmm.params.mu, latent_gmm.params.cov
    )

    # print("Any nans?", jnp.isnan(y_mu).any(), jnp.isnan(y_cov).any())
    # print("> and where?", jnp.where(jnp.isnan(y_mu)), jnp.where(jnp.isnan(y_cov)))

    eps_mu, eps_cov = unwhiten(y_mu, y_cov)
    eps_cov = 0.5 * (eps_cov + jnp.swapaxes(eps_cov, -1, -2))

    K = latent_gmm.K
    M = eps_mu.shape[1]

    return QGPVLM(
        K=K,
        M=M,
        gmm=latent_gmm,
        lvm=latent_bgplvm,
        psi=psi,
        whiten=whiten,
        unwhiten=unwhiten,
        pi=y_pi,
        mu=eps_mu,
        cov=eps_cov,
    )


def sample_qgpvlm(key, qgp: QGPVLM, tau, nsamples):
    Psi = jax.vmap(qgp.psi)(tau)

    key, k_key, z_key = jax.random.split(key, 3)

    ks = jax.random.categorical(k_key, jnp.log(qgp.pi), shape=(nsamples,))
    z_keys = jax.random.split(z_key, nsamples)

    def one_sample(k, subkey):
        mu_k = qgp.mu[k]
        cov_k = qgp.cov[k]
        sample_k = jax.random.multivariate_normal(
            subkey, mu_k, cov_k, method="svd"
        )
        return Psi @ sample_k

    return jax.vmap(one_sample)(ks, z_keys)


def np_safe_cholesky(A, jitter=1e-6):
    nuggets = float(np.mean(np.diag(A))) * float(jitter)
    return np.linalg.cholesky(A + nuggets * np.eye(A.shape[-1]))


def chol_psd(A, jitter=1e-6, eps=1e-12):
    A = 0.5 * (A + A.T)
    try:
        return np_safe_cholesky(A, jitter=jitter)
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(A)
        scale = max(float(np.max(w)), 1.0)
        w = np.maximum(w, eps * scale)
        A = (Q * w) @ Q.T
        A = 0.5 * (A + A.T)
        return np.linalg.cholesky(A)


def loglikelihood_on_test(
    qgp,
    f_list,
    Psi_list,
    obs_std,
    nu=np.inf,
    noise_floor=1e-3,
    jitter=1e-6,
    n_jobs=-1,
    backend="loky",  # "loky" (processes) or "threading"
    verbose=0,
):
    """Calculate the log likelihood of test data under the qGPVLM model

    Depending on the value of nu, this uses either a Gaussian likelihood (nu=inf) or a Student-t likelihood (finite nu).

    Note: this is just a GMM in data space, but the reduced rank Psi matrices depend on each data point.
    This defeats jax and plain numpy was too slow, so we use joblib for parallelism, which gives 50x speedup
    """
    sigma = max(float(obs_std), float(noise_floor))
    sigma2 = sigma * sigma
    inv_sigma2 = 1.0 / sigma2
    log2pi = np.log(2.0 * np.pi)

    mu = np.asarray(qgp.mu)
    pi = np.asarray(qgp.pi)
    cov = np.asarray(qgp.cov)

    K, D = mu.shape
    Ntest = len(f_list)

    # Precompute per-component stuff once.
    Sig_inv = np.empty((K, D, D), dtype=np.float64)
    logdet_Sig = np.empty(K, dtype=np.float64)
    I_D = np.eye(D, dtype=np.float64)

    for k in range(K):
        L = chol_psd(cov[k], jitter=jitter)
        logdet_Sig[k] = 2.0 * np.sum(np.log(np.diag(L)))
        Sig_inv[k] = np.linalg.solve(L.T, np.linalg.solve(L, I_D))

    log_pi = np.log(pi + 1e-300)

    def one_item(f, Psi):
        f = np.asarray(f)
        Psi = np.asarray(Psi)

        Gi = Psi.T @ Psi  # (D,D)
        ti = Psi.T @ f  # (D,)
        si = float(f @ f)  # scalar
        T = f.shape[0]

        lps = np.empty(K, dtype=np.float64)

        for k in range(K):
            muk = mu[k]
            bik = ti - Gi @ muk
            r2 = si - 2.0 * (muk @ ti) + muk @ (Gi @ muk)

            Mik = Sig_inv[k] + inv_sigma2 * Gi
            LM = chol_psd(Mik, jitter=jitter)

            x = np.linalg.solve(LM, bik)
            quad = inv_sigma2 * (r2 - inv_sigma2 * (x @ x))

            logdet_M = 2.0 * np.sum(np.log(np.diag(LM)))
            logdet_C = T * np.log(sigma2) + logdet_Sig[k] + logdet_M

            if np.isinf(nu):
                # Gaussian likelihood
                lps[k] = log_pi[k] - 0.5 * (T * log2pi + logdet_C + quad)
            else:
                # Student-t likelihood
                term1 = gammaln((nu + T) / 2) - gammaln(nu / 2)
                term2 = -0.5 * logdet_C
                term3 = -(T / 2) * np.log(nu * np.pi)
                term4 = -((nu + T) / 2) * np.log1p(quad / nu)

                lps[k] = log_pi[k] + term1 + term2 + term3 + term4

        m = np.max(lps)
        return m + np.log(np.sum(np.exp(lps - m)))

    # Parallel over items (works with varying shapes).
    if n_jobs == 1:
        out = np.empty(Ntest, dtype=np.float64)
        for i, (f, Psi) in enumerate(zip(f_list, Psi_list)):
            out[i] = one_item(f, Psi)
        return out

    vals = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(one_item)(f, Psi) for f, Psi in zip(f_list, Psi_list)
    )
    return np.asarray(vals, dtype=np.float64)