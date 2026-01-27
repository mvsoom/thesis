"""Collapsed Bayesian GPLVM (Titsias & Lawrence 2010) with ARD RBF kernel

Plus analytic psi-statistics for full rank Gaussians, which can be used to lift GMMs in latent space to data space
"""
# %%

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax as ox
from flax import nnx
from gpjax.linalg.utils import add_jitter
from gpjax.parameters import PositiveReal, Real

from utils.jax import kl_diag_gauss, safe_cholesky


def psi_stats_rbf_ard_diagonal(mu, var, Z, kernel):
    """
    Pointwise analytic psi stats for ARD RBF with q(x)=N(mu, diag(var)).
    Special case of psi_stats_rbf_ard_full() but kept here for speedy elbo() computation.

    mu:  [Q]
    var: [Q]
    Z:   [M,Q]

    Returns:
      psi0: scalar
      psi1: [M]
      psi2: [M,M]
    """
    mu = jnp.asarray(mu)
    var = jnp.asarray(var)
    Z = jnp.asarray(Z)

    M, Q = Z.shape

    sigma2_f = kernel.variance
    ell = kernel.lengthscale
    inv_l2 = 1.0 / (ell**2)

    # psi0 = sigma_f^2 for RBF
    psi0 = jnp.asarray(sigma2_f).reshape(())

    # psi1[m]
    diff_mq = mu[None, :] - Z
    den1 = 1.0 + var[None, :] * inv_l2[None, :]
    exp1 = jnp.exp(-0.5 * inv_l2[None, :] * (diff_mq**2) / den1) / jnp.sqrt(
        den1
    )
    psi1 = sigma2_f * jnp.prod(exp1, axis=-1)  # [M]

    # psi2[m,mp]
    Zm = Z[:, None, :]
    Zmp = Z[None, :, :]
    Zdiff_mmq = Zm - Zmp
    zbar_mmq = 0.5 * (Zm + Zmp)

    term_a = jnp.exp(-0.25 * inv_l2[None, None, :] * (Zdiff_mmq**2))

    diff_mm_q = mu[None, None, :] - zbar_mmq
    den2 = 1.0 + 2.0 * var[None, None, :] * inv_l2[None, None, :]
    term_b = jnp.exp(
        -(inv_l2[None, None, :] * (diff_mm_q**2)) / den2
    ) / jnp.sqrt(den2)

    prod_q = jnp.prod(term_a * term_b, axis=-1)
    psi2 = (sigma2_f**2) * prod_q  # [M,M]

    return psi0, psi1, psi2


def psi_stats_rbf_ard_full(mu, Sigma, Z, kernel, jitter=1e-6):
    mu = jnp.asarray(mu)
    Sigma = jnp.asarray(Sigma)
    Z = jnp.asarray(Z)

    M, Q = Z.shape

    sigma2_f = kernel.variance
    ell = kernel.lengthscale

    Lam = jnp.diag(ell * ell)
    inv_l2 = 1.0 / (ell * ell)

    psi0 = jnp.asarray(sigma2_f).reshape(())

    B1 = jnp.eye(Q, dtype=Sigma.dtype) + Sigma * inv_l2[None, :]
    LB1 = safe_cholesky(B1, jitter)
    logdetB1 = 2.0 * jnp.sum(jnp.log(jnp.diag(LB1)))

    A1 = Sigma + Lam
    LA1 = safe_cholesky(A1, jitter)

    Dm = Z - mu[None, :]
    Y = jsp.linalg.solve_triangular(LA1, Dm.T, lower=True)
    quad1 = jnp.sum(Y * Y, axis=0)

    psi1 = sigma2_f * jnp.exp(-0.5 * logdetB1 - 0.5 * quad1)

    B2 = jnp.eye(Q, dtype=Sigma.dtype) + 2.0 * (Sigma * inv_l2[None, :])
    LB2 = safe_cholesky(B2, jitter)
    logdetB2 = 2.0 * jnp.sum(jnp.log(jnp.diag(LB2)))

    C = Sigma + 0.5 * Lam
    LC = safe_cholesky(C, jitter)

    Zm = Z[:, None, :]
    Zmp = Z[None, :, :]
    d = Zm - Zmp
    quad_d = jnp.sum((d * d) * inv_l2[None, None, :], axis=-1)
    term_a = jnp.exp(-0.25 * quad_d)

    zbar = 0.5 * (Zm + Zmp)
    U = (zbar - mu[None, None, :]).reshape(M * M, Q)

    Y = jsp.linalg.solve_triangular(LC, U.T, lower=True)
    quad_u = jnp.sum(Y * Y, axis=0).reshape(M, M)
    term_b = jnp.exp(-0.5 * quad_u)

    psi2 = (sigma2_f * sigma2_f) * jnp.exp(-0.5 * logdetB2) * (term_a * term_b)

    return psi0, psi1, psi2


if __name__ == "__main__":
    import gpjax as gpx

    from utils.jax import vk

    Q = 3

    mu = jax.random.normal(vk(), (Q,))
    var = jnp.exp(jax.random.normal(vk(), (Q,)))
    Sigma = jnp.diag(var)

    Z = jax.random.normal(vk(), (50, Q))

    lengthscale = jnp.exp(jax.random.normal(vk(), (Q,)))
    kernel = gpx.kernels.RBF(lengthscale=lengthscale)

    psi0_d, psi1_d, psi2_d = psi_stats_rbf_ard_diagonal(mu, var, Z, kernel)
    psi0_f, psi1_f, psi2_f = psi_stats_rbf_ard_full(mu, Sigma, Z, kernel)

    print("psi0 difference:", jnp.abs(psi0_d - psi0_f))
    print("psi1 difference:", jnp.linalg.norm(psi1_d - psi1_f))
    print("psi2 difference:", jnp.linalg.norm(psi2_d - psi2_f))

# %%


def psi_stats_rbf_ard_diagonal_batch(mu, var, Z, kernel):
    """
    Analytic psi-stats for ARD RBF under q(x_n)=N(mu_n, diag(var_n)).

    Implemented as vmap of pointwise psi stats and summation.

    Returns:
      psi0: scalar
      psi1: [N,M]
      psi2: [M,M]   (summed over n)
    """
    mu = jnp.asarray(mu)
    var = jnp.asarray(var)

    psi0_n, psi1_n, psi2_n = jax.vmap(
        lambda m, v: psi_stats_rbf_ard_diagonal(m, v, Z, kernel)
    )(mu, var)

    psi0 = jnp.sum(psi0_n)
    psi1 = psi1_n  # [N,M]
    psi2 = jnp.sum(psi2_n, axis=0)  # [M,M]

    return psi0, psi1, psi2


def psi_stats_rbf_ard_full_batch(mu, Sigma, Z, kernel, jitter=1e-6):
    psi0_n, psi1_n, psi2_n = jax.vmap(
        lambda m, S: psi_stats_rbf_ard_full(m, S, Z, kernel, jitter=jitter)
    )(mu, Sigma)

    psi0 = jnp.sum(psi0_n)
    psi1 = psi1_n
    psi2 = jnp.sum(psi2_n, axis=0)
    return psi0, psi1, psi2


class BayesianGPLVM(nnx.Module):
    def __init__(
        self,
        kernel,
        X_mu,
        X_var,
        Z,
        sigma2=1.0,
        X_prior_mu=None,
        X_prior_var=None,
        jitter=1e-6,
    ):
        self.kernel = kernel
        self.jitter = jitter

        # trainables
        self.X_mu = Real(X_mu)
        self.X_var = PositiveReal(X_var)
        self.Z = Real(Z)
        self.sigma2 = PositiveReal(sigma2)

        # fixed priors
        if X_prior_mu is None:
            X_prior_mu = jnp.zeros_like(X_mu, shape=X_mu.shape)

        if X_prior_var is None:
            X_prior_var = jnp.ones_like(X_var, shape=X_var.shape)

        self.X_prior_mu = X_prior_mu
        self.X_prior_var = X_prior_var

    def elbo(self, Y, obs_var_diag=None):
        """ELBO of Titsias & Lawrence (2010) collapsed Bayesian GPLVM with extension that approximates diagonal data observation noise"""
        Y = jnp.asarray(Y)

        X_mu = self.X_mu.value
        X_var = self.X_var.value
        Z = self.Z.value
        sigma2 = self.sigma2.value

        N, D = Y.shape
        M = Z.shape[0]

        psi0, psi1, psi2 = psi_stats_rbf_ard_diagonal_batch(
            X_mu, X_var, Z, self.kernel
        )

        Kuu = self.kernel.gram(Z).to_dense()
        Kuu = add_jitter(Kuu, self.jitter)

        L = safe_cholesky(Kuu, self.jitter)

        A = jsp.linalg.solve_triangular(L, psi1.T, lower=True)
        tmp = jsp.linalg.solve_triangular(L, psi2, lower=True)
        AAT = jsp.linalg.solve_triangular(L, tmp.T, lower=True) / sigma2
        B = AAT + jnp.eye(M, dtype=Y.dtype)
        LB = safe_cholesky(B, self.jitter)

        log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diag(LB)))
        c = jsp.linalg.solve_triangular(LB, A @ Y, lower=True) / sigma2

        KLx = kl_diag_gauss(X_mu, X_var, self.X_prior_mu, self.X_prior_var)

        ND = jnp.asarray(N * D, dtype=Y.dtype)

        bound = -0.5 * ND * jnp.log(2.0 * jnp.pi * sigma2)
        bound = bound - 0.5 * jnp.asarray(D, dtype=Y.dtype) * log_det_B
        bound = bound - 0.5 * jnp.sum(Y * Y) / sigma2

        if obs_var_diag is not None:
            R = jsp.linalg.solve_triangular(LB, A, lower=True) / sigma2
            W_diag = (1.0 / sigma2) - jnp.sum(R * R, axis=0)
            bound = bound - 0.5 * jnp.sum(W_diag[:, None] * obs_var_diag)

        bound = bound + 0.5 * jnp.sum(c * c)
        bound = bound - 0.5 * jnp.asarray(D, dtype=Y.dtype) * (
            psi0 / sigma2 - jnp.trace(AAT)
        )
        bound = bound - KLx

        return bound

    def build_posterior(self, Y):
        Y = jnp.asarray(Y)

        X_mu = self.X_mu.value
        X_var = self.X_var.value
        Z = self.Z.value
        sigma2 = self.sigma2.value

        psi0, psi1, psi2 = psi_stats_rbf_ard_diagonal_batch(
            X_mu, X_var, Z, self.kernel
        )

        Kuu = self.kernel.gram(Z).to_dense()
        Kuu = add_jitter(Kuu, self.jitter)

        L = safe_cholesky(Kuu, self.jitter)

        A = jsp.linalg.solve_triangular(L, psi1.T, lower=True)
        tmp = jsp.linalg.solve_triangular(L, psi2, lower=True)
        AAT = jsp.linalg.solve_triangular(L, tmp.T, lower=True) / sigma2
        B = AAT + jnp.eye(Z.shape[0], dtype=Y.dtype)
        LB = safe_cholesky(B, self.jitter)

        c = jsp.linalg.solve_triangular(LB, A @ Y, lower=True) / sigma2

        return BGPLVMPosterior(
            kernel=self.kernel,
            Z=Z,
            sigma2=sigma2,
            L=L,
            LB=LB,
            c=c,
            jitter=self.jitter,
        )


class BGPLVMPosterior:
    """Cached posterior with predictive equations"""

    def __init__(self, kernel, Z, sigma2, L, LB, c, jitter):
        self.kernel = kernel
        self.Z = Z
        self.sigma2 = sigma2

        self.L = L
        self.LB = LB
        self.c = c

        self.jitter = jitter

    def predict_f_meanvar_batch(self, x_star_mu, x_star_var):
        """Compute y* = f(x*) ~ N(y_mean, diag(y_var)) for a batch of inputs x* ~ N(x_star_mu, diag(x_star_var))"""
        x_star_mu = jnp.atleast_2d(x_star_mu)
        x_star_var = jnp.atleast_2d(x_star_var)

        psi0, psi1, psi2 = jax.vmap(
            lambda m, v: psi_stats_rbf_ard_diagonal(m, v, self.Z, self.kernel)
        )(x_star_mu, x_star_var)

        def one_point(psi1_b, psi2_b):
            a = jsp.linalg.solve_triangular(
                self.L, psi1_b[:, None], lower=True
            ).squeeze(-1)
            r = jsp.linalg.solve_triangular(
                self.LB, a[:, None], lower=True
            ).squeeze(-1)

            mean = r @ self.c

            tmp = jsp.linalg.solve_triangular(self.L, psi2_b, lower=True)
            EaaT = jsp.linalg.solve_triangular(self.L, tmp.T, lower=True).T

            t1 = jnp.trace(EaaT)

            C = jsp.linalg.solve_triangular(self.LB, EaaT, lower=True)
            Binv_EaaT = jsp.linalg.solve_triangular(self.LB.T, C, lower=False)

            t2 = jnp.trace(Binv_EaaT)

            var_f = psi0 - t1 + t2
            var_f = jnp.clip(var_f, a_min=0.0)

            return mean, var_f

        y_means, y_vars = jax.vmap(one_point)(psi1, psi2)

        D = self.c.shape[1]
        y_vars = y_vars[:, None] * jnp.ones((1, D), dtype=y_means.dtype)

        return y_means, y_vars

    def _Wmat(self):
        tmp = jsp.linalg.solve_triangular(self.LB.T, self.c, lower=False)
        W = jsp.linalg.solve_triangular(self.L.T, tmp, lower=False)
        return W

    def forward_x(self, x_mu, x_Sigma, jitter=1e-6):
        """Propagate a point X ~ N(x_mu, x_Sigma) through a moment-matched BGPLVM nonlinearity to get Y ~ N(y_mu, y_Sigma) in data space

        NOTE: this already takes into learned BGPLVM observation noise! (self.sigma2 below)
        """
        psi0, ek, Ekk = psi_stats_rbf_ard_full(
            x_mu, x_Sigma, self.Z, self.kernel, jitter=jitter
        )

        W = self._Wmat()

        y_mu = ek @ W

        tmp = jsp.linalg.solve_triangular(self.L, Ekk, lower=True)
        EaaT = jsp.linalg.solve_triangular(self.L, tmp.T, lower=True).T
        t1 = jnp.trace(EaaT)

        C = jsp.linalg.solve_triangular(self.LB, EaaT, lower=True)
        Binv_EaaT = jsp.linalg.solve_triangular(self.LB.T, C, lower=False)
        t2 = jnp.trace(Binv_EaaT)

        v_base = psi0 - t1 + t2
        v_base = jnp.clip(v_base, a_min=0.0)

        Covk = Ekk - ek[:, None] * ek[None, :]
        S_mean = W.T @ Covk @ W

        D = self.c.shape[1]
        y_Sigma = S_mean + (v_base + self.sigma2) * jnp.eye(
            D, dtype=S_mean.dtype
        )

        return y_mu, y_Sigma

    def forward_x_gmm(self, pis, x_mus, x_Sigmas, jitter=1e-6):
        """Propagate a GMM in X through the BGPLVM nonlinearity to get a GMM in Y"""

        def one(m, S):
            return self.forward_x(m, S, jitter=jitter)

        y_mus, y_Sigmas = jax.vmap(one)(x_mus, x_Sigmas)
        return pis, y_mus, y_Sigmas


# %%
def optimize(key, model, dataset, lr, num_iters, **fit_kwargs):
    model = model(key)
    optim = ox.adam(learning_rate=lr)

    def cost(q, d):
        return -q.elbo(d.y, obs_var_diag=d.X)

    fitted, cost_history = gpx.fit(
        model=model,
        objective=cost,
        train_data=dataset,
        optim=optim,
        num_iters=num_iters,
        **fit_kwargs,
    )

    elbo_history = -cost_history
    return fitted, elbo_history
