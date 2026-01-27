import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

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

    eps_mu, eps_cov = unwhiten(y_mu, y_cov)

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
        sample_k = jax.random.multivariate_normal(subkey, mu_k, cov_k)
        return Psi @ sample_k

    return jax.vmap(one_sample)(ks, z_keys)


def logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def chol_logdet(L):
    return 2.0 * np.sum(np.log(np.diag(L)))


def loglikelihood_on_test(
    qgp: QGPVLM,
    f_list,
    Psi_list,
    obs_std,
    noise_floor=1e-3,
):
    sigma = max(float(obs_std), float(noise_floor))
    sigma2 = sigma * sigma
    inv_sigma2 = 1.0 / sigma2
    inv_sigma4 = inv_sigma2 * inv_sigma2

    K, D = qgp.mu.shape
    Ntest = len(f_list)

    Sig_inv = np.empty_like(qgp.cov)
    logdet_Sig = np.empty(K)
    for k in range(K):
        L = np.linalg.cholesky(qgp.cov[k])
        logdet_Sig[k] = 2.0 * np.sum(np.log(np.diag(L)))
        Sig_inv[k] = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D)))

    lls = np.empty(Ntest)

    for i, (f, Psi) in enumerate(zip(f_list, Psi_list)):
        Gi = Psi.T @ Psi
        ti = Psi.T @ f
        si = float(f @ f)
        Ti = f.shape[0]

        lps = np.empty(K)

        for k in range(K):
            muk = qgp.mu[k]
            bik = ti - Gi @ muk
            r2 = si - 2.0 * (muk @ ti) + muk @ (Gi @ muk)

            Mik = Sig_inv[k] + inv_sigma2 * Gi
            LM = np.linalg.cholesky(Mik)

            x = np.linalg.solve(LM, bik)
            quad = inv_sigma2 * r2 - inv_sigma4 * (x @ x)

            logdet_M = 2.0 * np.sum(np.log(np.diag(LM)))
            logdet_C = Ti * np.log(sigma2) + logdet_Sig[k] + logdet_M

            lps[k] = np.log(qgp.pi[k]) - 0.5 * (
                Ti * np.log(2.0 * np.pi) + logdet_C + quad
            )

        m = np.max(lps)
        lls[i] = m + np.log(np.sum(np.exp(lps - m)))

    return lls
