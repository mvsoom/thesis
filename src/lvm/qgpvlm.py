import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from lvm.bgplvm import BGPLVMPosterior
from lvm.xdgmm import GMMFit
from prism.svi import blr_log_evidence, svi_basis
from utils.constants import NOISE_FLOOR_POWER


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


def surrogate_mixture_log_evidence_on_test(
    qgp,
    qsvi,
    dataset,
    batch_size=None,
    device=None,
    noise_floor=np.sqrt(NOISE_FLOOR_POWER),
    jitter=1e-6,
):
    """
    Exact reduced-rank Gaussian marginal likelihood under a mixture of BLRs.

    Returns:
        logp : (N,) array, one scalar per waveform
    """
    sigma = jnp.maximum(
        qsvi.posterior.likelihood.obs_stddev,
        noise_floor,
    )
    sigma2 = sigma**2

    log_pi = jnp.log(qgp.pi + 1e-30)

    def one_waveform(inputs):
        t, y = inputs
        Psi = jax.vmap(lambda t: svi_basis(qsvi, t))(t)

        def per_component(k):
            return log_pi[k] + blr_log_evidence(
                y,
                Psi,
                qgp.mu[k],
                qgp.cov[k],
                sigma2,
                jitter=jitter,
            )

        lps = jax.vmap(per_component)(jnp.arange(qgp.K))
        return jax.scipy.special.logsumexp(lps)

    with jax.default_device(device):
        return jax.lax.map(
            one_waveform,
            (dataset.X, dataset.y),
            batch_size=batch_size,
        )