import jax
import jax.numpy as jnp
import jax.scipy as jsp

from utils.jax import safe_cholesky


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

    # Posterior covariance of eps
    Sigma_eps = jsp.linalg.cho_solve((L, True), jnp.eye(M))

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
        mu_eps=mu_eps,
        Sigma_eps=Sigma_eps,
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


def collapsed_elbo_masked_t(q, t, y, nu, num_inner=3):
    sigma2 = q.posterior.likelihood.obs_stddev**2

    mask = ~jnp.isnan(y)
    y0 = jnp.where(mask, y, 0.0)[:, None]
    mask_col = mask[:, None]

    # initialise E[lambda] = 1 on valid points, 0 on padded
    w = mask_col.astype(y.dtype)

    for _ in range(num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

        r2 = (y0 - m) ** 2 + v
        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)

        beta = jnp.maximum(beta, 1e-12)
        w = (alpha / beta) * mask_col

    # final Gaussian collapsed term (expects safe y)
    elbo_gauss = _collapsed_gaussian_elbo(y0, w, aux, sigma2)

    # lambda terms
    r2 = (y0 - m) ** 2 + v
    alpha = 0.5 * (nu + 1.0)
    beta = 0.5 * (nu + r2 / sigma2)
    beta = jnp.maximum(beta, 1e-12)

    E_loglam = jsp.special.digamma(alpha) - jnp.log(beta)
    E_loglam = E_loglam * mask_col

    # KL(q(lambda) || p(lambda)) where both are Gamma(shape, rate)
    a = alpha
    b = beta
    a0 = 0.5 * nu
    b0 = 0.5 * nu

    kl_lam = gamma_kl(a, b, a0, b0)
    kl_lam = jnp.sum(kl_lam * mask_col)

    return elbo_gauss + 0.5 * jnp.sum(E_loglam) - kl_lam


def batch_collapsed_elbo_masked_t(q, data, nu, I_total, num_inner=3):
    X = data.X
    y = data.y
    B = X.shape[0]

    elbos = jax.vmap(
        lambda t_i, y_i: collapsed_elbo_masked_t(q, t_i, y_i, nu, num_inner),
        in_axes=(0, 0),
    )(X, y)

    return (I_total / B) * jnp.sum(elbos)


def infer_eps_posterior_single_t(q, t, y, nu, num_inner=3):
    sigma2 = q.posterior.likelihood.obs_stddev**2

    mask = ~jnp.isnan(y)
    y0 = jnp.where(mask, y, 0.0)[:, None]
    mask_col = mask[:, None]

    w = mask_col.astype(y.dtype)

    for _ in range(num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

        r2 = (y0 - m) ** 2 + v
        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)

        beta = jnp.maximum(beta, 1e-12)
        w = (alpha / beta) * mask_col

    # one final consistent Gaussian posterior with final weights
    m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

    mu_eps = aux["mu_eps"].squeeze() # (M,)
    Sigma_eps = aux["Sigma_eps"] # (M,M)

    return mu_eps, Sigma_eps