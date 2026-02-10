import jax
import jax.numpy as jnp
import jax.scipy as jsp
from gpjax.variational_families import CollapsedVariationalGaussian

from utils.jax import safe_cholesky


def gamma_logpdf(x, alpha, beta, eps=1e-30):
    x = jnp.maximum(x, eps)
    return (
        alpha * jnp.log(beta)
        - jsp.special.gammaln(alpha)
        + (alpha - 1.0) * jnp.log(x)
        - beta * x
    )


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


class t_CollapsedVariationalGaussian(CollapsedVariationalGaussian):
    def __init__(
        self,
        posterior,
        inducing_inputs,
        nu: float = 1.0,
        num_inner: int = 3,
        jitter=1e-6,
    ):
        super().__init__(
            posterior=posterior, inducing_inputs=inducing_inputs, jitter=jitter
        )

        self.num_inner = num_inner  # static
        self.nu = nu  # static (for now)


def t_collapsed_elbo_masked(q: t_CollapsedVariationalGaussian, t, y):
    nu = jnp.array(q.nu)
    sigma2 = q.posterior.likelihood.obs_stddev**2

    mask = ~jnp.isnan(y)
    y0 = jnp.where(mask, y, 0.0)[:, None]
    mask_col = mask[:, None]

    # initialise E[lambda] = 1 on valid points, 0 on padded
    w = mask_col.astype(y.dtype)

    for _ in range(q.num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

        r2 = (y0 - m) ** 2 + v
        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)

        beta = jnp.maximum(beta, 1e-12)
        w = (
            jax.lax.stop_gradient(alpha / beta) * mask_col
        )  # Don't differentiate through CAVI: Hoffman local/global decoupling

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


def t_batch_collapsed_elbo_masked(
    q: t_CollapsedVariationalGaussian, data, I_total
):
    X = data.X
    y = data.y
    B = X.shape[0]

    elbos = jax.vmap(
        lambda t_i, y_i: t_collapsed_elbo_masked(q, t_i, y_i),
        in_axes=(0, 0),
    )(X, y)

    return (I_total / B) * jnp.sum(elbos)


def t_infer_eps_posterior_single(q, t, y):
    nu = jnp.array(q.nu)
    sigma2 = q.posterior.likelihood.obs_stddev**2

    mask = ~jnp.isnan(y)
    y0 = jnp.where(mask, y, 0.0)[:, None]
    mask_col = mask[:, None]

    w = mask_col.astype(y.dtype)

    for _ in range(q.num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

        r2 = (y0 - m) ** 2 + v
        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)

        beta = jnp.maximum(beta, 1e-12)
        w = (
            jax.lax.stop_gradient(alpha / beta) * mask_col
        )  # Don't differentiate through CAVI

    # one final consistent Gaussian posterior with final weights
    m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

    mu_eps = aux["mu_eps"].squeeze()  # (M,)
    Sigma_eps = aux["Sigma_eps"]  # (M,M)

    return mu_eps, Sigma_eps, w.squeeze()


def do_t_prism(q, dataset, device=jax.devices("cpu")[0]):
    """Calculate the amplitude posterior for all waveforms in dataset, thereby refracting the dataset like a prism into latent space"""
    with jax.default_device(device):
        mu_eps, Sigma_eps, w = jax.vmap(
            t_infer_eps_posterior_single,
            in_axes=(None, 0, 0),
        )(q, dataset.X, dataset.y)
    return mu_eps, Sigma_eps, w


def t_infer_lambda_posterior_single(q, t, y):
    nu = jnp.array(q.nu)
    sigma2 = q.posterior.likelihood.obs_stddev**2

    mask = ~jnp.isnan(y)
    y0 = jnp.where(mask, y, 0.0)[:, None]
    mask_col = mask[:, None]

    w = mask_col.astype(y.dtype)

    for _ in range(q.num_inner):
        m, v, aux = _collapsed_gaussian_stats(q, t, y, w)

        r2 = (y0 - m) ** 2 + v

        alpha = 0.5 * (nu + 1.0)
        beta = 0.5 * (nu + r2 / sigma2)

        beta = jnp.maximum(beta, 1e-12)

        w = jax.lax.stop_gradient(alpha / beta) * mask_col

    return alpha.squeeze(), beta.squeeze(), aux


def _collapsed_gaussian_loglik(q, y, lam, aux):
    sigma2 = q.posterior.likelihood.obs_stddev**2
    Psi = aux["Psi"]
    M = Psi.shape[0]

    sw = jnp.sqrt(lam)  # (W,)

    A = (Psi * sw[None, :]) / jnp.sqrt(sigma2)  # (M,W)
    B = jnp.eye(M) + A @ A.T
    L = safe_cholesky(B, jitter=q.jitter)

    diff = y * sw[:, None]  # (W,1)

    tmp = jsp.linalg.solve_triangular(L, A @ diff, lower=True)
    quad = (jnp.sum(diff * diff) - jnp.sum(tmp * tmp)) / sigma2

    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
    n_eff = aux["n_eff"]

    return -0.5 * (n_eff * jnp.log(2.0 * jnp.pi * sigma2) + log_det_B + quad)


def t_log_evidence_is_single(q, t, y, key, S=32):
    """Estimate log p(y) = log int p(y|f, lam) p(f) p(lam) df dlam using Monte Carlo samples from q(lambda) (importance-weighted)"""
    nu = jnp.array(q.nu)

    alpha, beta, aux = t_infer_lambda_posterior_single(q, t, y)

    mask = ~jnp.isnan(y)
    y0 = jnp.where(mask, y, 0.0)[:, None]

    # sample lambda from q(lambda)
    keys = jax.random.split(key, S)

    def one_sample(k):
        lam = jax.random.gamma(k, alpha) / beta
        lam = lam * mask

        log_py_lam = _collapsed_gaussian_loglik(q, y0, lam, aux)

        log_p_lam = jnp.sum(gamma_logpdf(lam, 0.5 * nu, 0.5 * nu) * mask)
        log_q_lam = jnp.sum(gamma_logpdf(lam, alpha, beta) * mask)

        # jprint("log_py_lam = {x}", x=log_py_lam)
        # jprint("log_p_lam = {x}", x=log_p_lam)  # all nan
        # jprint("log_q_lam = {x}", x=log_q_lam)  # all nan

        return log_py_lam + log_p_lam - log_q_lam

    logw = jax.vmap(one_sample)(keys)

    m = jnp.max(logw)
    return m + jnp.log(jnp.mean(jnp.exp(logw - m)))


def t_log_evidence_is_on_test(
    q, dataset, key, S=32, device=jax.devices("cpu")[0]
):
    keys = jax.random.split(key, dataset.X.shape[0])

    with jax.default_device(device):
        return jax.vmap(
            lambda k, t_i, y_i: t_log_evidence_is_single(q, t_i, y_i, k, S),
            in_axes=(0, 0, 0),
        )(keys, dataset.X, dataset.y)