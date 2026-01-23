# bgplvm_collapsed.py
#
# Collapsed Bayesian GPLVM (Titsias & Lawrence 2010) with ARD RBF kernel
# GPJax / nnx compatible, Option B design with cached posterior + predictive equations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx
from gpjax.linalg.utils import add_jitter
from gpjax.parameters import PositiveReal, Real

from utils.jax import kl_diag_gauss, safe_cholesky

# -----------------------------------------------------------------------------
# Pointwise psi stats (single uncertain input)
# -----------------------------------------------------------------------------


def psi_stats_rbf_ard_pointwise(mu, var, Z, kernel):
    """
    Pointwise analytic psi stats for ARD RBF with q(x)=N(mu, diag(var)).

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


# -----------------------------------------------------------------------------
# Training psi stats = vmap(pointwise) + sum
# -----------------------------------------------------------------------------


def psi_stats_rbf_ard(mu, var, Z, kernel):
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
        lambda m, v: psi_stats_rbf_ard_pointwise(m, v, Z, kernel)
    )(mu, var)

    psi0 = jnp.sum(psi0_n)
    psi1 = psi1_n  # [N,M]
    psi2 = jnp.sum(psi2_n, axis=0)  # [M,M]

    return psi0, psi1, psi2


# -----------------------------------------------------------------------------
# Core model
# -----------------------------------------------------------------------------


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

    # -------------------------------------------------------------------------
    # ELBO (collapsed)
    # -------------------------------------------------------------------------

    def elbo(self, Y, obs_var_diag=None):
        Y = jnp.asarray(Y)

        X_mu = self.X_mu.value
        X_var = self.X_var.value
        Z = self.Z.value
        sigma2 = self.sigma2.value

        N, D = Y.shape
        M = Z.shape[0]

        psi0, psi1, psi2 = psi_stats_rbf_ard(X_mu, X_var, Z, self.kernel)

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

    # -------------------------------------------------------------------------
    # Build cached posterior
    # -------------------------------------------------------------------------

    def build_posterior(self, Y):
        Y = jnp.asarray(Y)

        X_mu = self.X_mu.value
        X_var = self.X_var.value
        Z = self.Z.value
        sigma2 = self.sigma2.value

        psi0, psi1, psi2 = psi_stats_rbf_ard(X_mu, X_var, Z, self.kernel)

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


# -----------------------------------------------------------------------------
# Cached posterior with predictive equations
# -----------------------------------------------------------------------------


class BGPLVMPosterior:
    def __init__(self, kernel, Z, sigma2, L, LB, c, jitter):
        self.kernel = kernel
        self.Z = Z
        self.sigma2 = sigma2

        self.L = L
        self.LB = LB
        self.c = c

        self.jitter = jitter

    # -------------------------------------------------------------------------
    # Predict mean and variance of latent f(x*)
    # -------------------------------------------------------------------------

    def predict_f_meanvar(self, x_star_mu, x_star_var):
        x_star_mu = jnp.atleast_2d(x_star_mu)
        x_star_var = jnp.atleast_2d(x_star_var)

        psi0, psi1, psi2 = jax.vmap(
            lambda m, v: psi_stats_rbf_ard_pointwise(m, v, self.Z, self.kernel)
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

        means, vars_f = jax.vmap(one_point)(psi1, psi2)

        D = self.c.shape[1]
        vars_f = vars_f[:, None] * jnp.ones((1, D), dtype=means.dtype)

        return means, vars_f
