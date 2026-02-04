import jax
import jax.numpy as jnp
import jax.scipy as jsp

from utils.jax import safe_cholesky

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def gamma_kl(alpha_q, beta_q, alpha_p, beta_p):
    """
    KL( Gamma(alpha_q, beta_q) || Gamma(alpha_p, beta_p) )
    shape-rate parameterization
    """
    return (
        (alpha_q - alpha_p) * jsp.special.digamma(alpha_q)
        - jsp.special.gammaln(alpha_q)
        + jsp.special.gammaln(alpha_p)
        + alpha_p * (jnp.log(beta_q) - jnp.log(beta_p))
        + alpha_q * (beta_p / beta_q - 1.0)
    )


# ------------------------------------------------------------
# Core: collapsed Gaussian machinery with diagonal weights
# ------------------------------------------------------------


def _collapsed_gaussian_stats(q, t, y, w):
    """
    Compute collapsed GP quantities for ONE waveform,
    given per-point weights w = E[lambda].

    Returns:
        m : [W,1] posterior mean of f
        v : [W,1] posterior variance of f
        aux : dict with reusable terms for ELBO
    """
    jitter = q.jitter

    mask_w = ~jnp.isnan(y)
    mask = mask_w[:, None]

    t = t[:, None]
    y = y[:, None]

    t = jnp.where(mask, t, 0.0)
    y = jnp.where(mask, y, 0.0)
    w = jnp.where(mask, w, 0.0)

    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs
    sigma2 = q.posterior.likelihood.obs_stddev**2

    M = Z.shape[0]

    # Kernel blocks
    Kzz = kernel.gram(Z).to_dense()
    Kzx = kernel.cross_covariance(Z, t)
    Kxx_diag = jax.vmap(kernel, in_axes=(0, 0))(t, t)

    Kzx = Kzx * mask_w[None, :]
    Kxx_diag = Kxx_diag * mask_w

    # Whitening
    Lz = safe_cholesky(Kzz, jitter=jitter)
    Psi = jsp.linalg.solve_triangular(Lz, Kzx, lower=True)  # [M,W]

    # Weighted design
    sw = jnp.sqrt(w)
    A = (Psi * sw.T) / jnp.sqrt(sigma2)  # [M,W]

    AAT = A @ A.T
    B = jnp.eye(M) + AAT
    L = safe_cholesky(B, jitter=jitter)

    # Posterior mean of eps
    diff = y * sw
    tmp = jsp.linalg.solve_triangular(L, A @ diff, lower=True)
    mu_eps = jsp.linalg.solve_triangular(L.T, tmp, lower=False) / jnp.sqrt(
        sigma2
    )

    # Posterior moments of f
    m = Psi.T @ mu_eps  # [W,1]

    Qxx_diag = jnp.sum(Psi * Psi, axis=0)
    SigmaPsi = jsp.linalg.cho_solve((L, True), Psi)
    diag_PsiT_Sigma_Psi = jnp.sum(Psi * SigmaPsi, axis=0)

    v = (Kxx_diag - Qxx_diag + diag_PsiT_Sigma_Psi)[:, None]
    v = jnp.maximum(v, 0.0)

    aux = dict(
        Psi=Psi,
        L=L,
        AAT=AAT,
        Kxx_diag=Kxx_diag,
        n_eff=jnp.sum(mask_w),
    )

    return m, v, aux


def _collapsed_gaussian_elbo(y, w, aux, sigma2):
    """
    Collapsed Gaussian ELBO term given fixed weights.
    This is exactly the Titsias bound with diagonal noise.
    """
    Psi = aux["Psi"]
    L = aux["L"]
    AAT = aux["AAT"]
    Kxx_diag = aux["Kxx_diag"]
    n_eff = aux["n_eff"]

    sw = jnp.sqrt(w)
    diff = y * sw

    tmp = jsp.linalg.solve_triangular(
        L, (Psi * sw.T) @ diff / jnp.sqrt(sigma2), lower=True
    )

    quad = (jnp.sum(diff * diff) - jnp.sum(tmp * tmp)) / sigma2
    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

    two_log_prob = -n_eff * jnp.log(2.0 * jnp.pi * sigma2) - log_det_B - quad

    two_trace = jnp.sum(w[:, 0] * Kxx_diag) / sigma2 - jnp.trace(AAT)

    return 0.5 * (two_log_prob - two_trace)


# ------------------------------------------------------------
# t-PRISM ELBO for one waveform
# ------------------------------------------------------------


def collapsed_elbo_masked_t(q, t, y, nu, num_inner=3):
    """
    Collapsed t-PRISM ELBO for ONE waveform with NaN masking.

    Local variables: q(lambda_n) via CAVI
    Global variables: Z, theta, sigma2, nu
    """
    sigma2 = q.posterior.likelihood.obs_stddev**2

    # initialization: E[lambda] = 1
    w = jnp.ones_like(y)[:, None]

    for _ in range(num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

        r2 = (y[:, None] - m) ** 2 + v
        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)

        beta = jnp.maximum(beta, 1e-12)
        w = alpha / beta

    # final moments
    m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

    # Gaussian collapsed term
    elbo_gauss = _collapsed_gaussian_elbo(y[:, None], w, aux, sigma2)

    # Gamma terms
    alpha = 0.5 * (nu + 1.0)
    beta = 0.5 * (nu + ((y[:, None] - m) ** 2 + v) / sigma2)

    E_loglam = jsp.special.digamma(alpha) - jnp.log(beta)

    kl_lam = gamma_kl(
        alpha_q=alpha,
        beta_q=beta,
        alpha_p=0.5 * nu,
        beta_p=0.5 * nu,
    )

    # Mask padded points
    mask = ~jnp.isnan(y)
    E_loglam = E_loglam * mask[:, None]
    kl_lam = kl_lam * mask[:, None]

    return elbo_gauss + 0.5 * jnp.sum(E_loglam) - jnp.sum(kl_lam)


# ------------------------------------------------------------
# Batch ELBO (SVI)
# ------------------------------------------------------------


def batch_collapsed_elbo_masked_t(q, data, nu, I_total, num_inner=3):
    X = data.X
    y = data.y
    B = X.shape[0]

    elbos = jax.vmap(
        lambda t_i, y_i: collapsed_elbo_masked_t(q, t_i, y_i, nu, num_inner),
        in_axes=(0, 0),
    )(X, y)

    return (I_total / B) * jnp.sum(elbos)


# ------------------------------------------------------------
# Robust projection: posterior over eps
# ------------------------------------------------------------


def infer_eps_posterior_single_t(q, t, y, nu, num_inner=3):
    """
    Robust PRISM projection:
    returns N(mu_eps, Sigma_eps) for one waveform
    """
    sigma2 = q.posterior.likelihood.obs_stddev**2

    w = jnp.ones_like(y)[:, None]

    for _ in range(num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)
        r2 = (y[:, None] - m) ** 2 + v
        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)
        w = alpha / jnp.maximum(beta, 1e-12)

    # build weighted BLR posterior
    Psi = aux["Psi"].T  # [W,M]
    sw = jnp.sqrt(w[:, 0])

    A = (Psi * sw[:, None]) / jnp.sqrt(sigma2)
    precision = jnp.eye(Psi.shape[1]) + A.T @ A
    Lp = safe_cholesky(precision)

    Sigma_eps = jsp.linalg.cho_solve((Lp, True), jnp.eye(Psi.shape[1]))
    mu_eps = (Sigma_eps @ (Psi.T @ (w[:, 0] * y)) / sigma2).squeeze()

    return mu_eps, Sigma_eps
