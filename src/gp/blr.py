# %%
from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy import linalg
from tinygp.gp import ConditionResult, GaussianProcess
from tinygp.helpers import JAXArray

from gp.mercer import Mercer


def sample_latent(key, cov_root):
    """z ~ N(0, I_R), f = Phi @ (cov_root @ z)."""
    R = cov_root.shape[1]
    z = jax.random.normal(key, (R,))
    return z


def sample(key, Phi, cov_root):
    """f = Phi @ cov_root @ z"""
    z = sample_latent(key, cov_root)
    return Phi @ (cov_root @ z)


def posterior_latent(
    y: JAXArray,
    Phi: JAXArray,
    cov_root: JAXArray,
    noise_variance: JAXArray = 0.0,
    *,
    PhiT_Phi: JAXArray | None = None,
    PhiT_y: JAXArray | None = None,
    jitter: float | None = None,
    return_aux: bool = False,
):
    """Compute posterior for z | y where

        y ~ N(Phi cov_root z, noise_variance I)
        z ~ N(0, I)

    Here diagonal term is noise_variance (observational noise) PLUS jitter
    (numerical stability, always added unless set to zero, scale aware).
    To get zero diagonal, set both noise_variance and jitter to 0.0.

    From log likelihood formulas in Section 3.2 in [1].

    Note 1: if basisfunctions phi(X) do not depend on hyperparameters, then
    PhiT_Phi = Phi.T @ Phi should be precomputed and passed in for efficiency.
    This reduces computational cost from O(N M^2 + M^3) to O(M^3), where N is
    number of data points and M is number of basis functions. Same goes for the
    projection of the data y on the basisfunctions: PhiT_y = Phi.T @ y.

    Note 2: if Phi.T @ y can be expressed as inner products of sinusoids with
    the data, a further speedup is possible using FFTs [1] or NUFFTs like
    jax-finufft package. This reduces O(N M) to O(N log N + M).

        [1] Solin, A., & Särkkä, S. (2020). Hilbert space methods for
            reduced-rank Gaussian process regression.
            Statistics and Computing, 30(2), 419-446.
    """

    y = jnp.asarray(y)
    Phi = jnp.asarray(Phi)
    N, M = Phi.shape
    R = cov_root.shape[1]

    if PhiT_Phi is None:
        PhiT_Phi = Phi.T @ Phi
    if PhiT_y is None:
        PhiT_y = Phi.T @ y

    A = cov_root.T @ PhiT_Phi @ cov_root

    diag_scale = jnp.mean(jnp.diag(A))
    eps = jnp.sqrt(jnp.finfo(y.dtype).eps) if jitter is None else jitter
    jitter_eff = eps * diag_scale

    sigma2 = noise_variance + jitter_eff

    Z = A + sigma2 * jnp.eye(R, dtype=Phi.dtype)
    Lc, lower = linalg.cho_factor(Z, lower=True, check_finite=False)

    b = cov_root.T @ PhiT_y  # shape (R,)
    m_z = linalg.cho_solve((Lc, lower), b, check_finite=False)

    # latent posterior covariance root:
    # Sigma_z = sigma2 * Z^{-1}, with chol(Z) = Lc Lc^T
    # Sigma_z^(1/2) = sqrt(sigma2) * Lc^{-T}
    I_R = jnp.eye(R, dtype=Phi.dtype)
    Lc_invT = linalg.solve_triangular(Lc.T, I_R, lower=False)
    L_z_post = jnp.sqrt(sigma2) * Lc_invT  # (R, R)

    if return_aux:
        return m_z, L_z_post, Lc, b, PhiT_Phi, PhiT_y, sigma2

    return m_z, L_z_post


def posterior_weight(
    y: JAXArray,
    Phi: JAXArray,
    mu: JAXArray,
    cov_root: JAXArray,
    noise_variance: JAXArray = 0.0,
    *,
    PhiT_Phi: JAXArray | None = None,
    jitter: float | None = None,
):
    """Compute posterior for w | y where

        y ~ N(Phi w, noise_variance I)
        w ~ N(mu, cov_root @ cov_root.T)

    Implemented via latent z with w = mu + cov_root z and z ~ N(0, I).
    """

    Phi = jnp.asarray(Phi)
    y = jnp.asarray(y)

    # work with residuals so latent model is zero-mean
    y_resid = y - Phi @ mu

    m_z, L_z_post, _, _, PhiT_Phi, _, _ = posterior_latent(
        y=y_resid,
        Phi=Phi,
        cov_root=cov_root,
        noise_variance=noise_variance,
        PhiT_Phi=PhiT_Phi,
        jitter=jitter,
        return_aux=True,
    )

    # posterior over w = mu + L z
    m_w = mu + cov_root @ m_z
    L_w_post = cov_root @ L_z_post

    return m_w, L_w_post


def log_probability(
    y: JAXArray,
    Phi: JAXArray,
    cov_root: JAXArray,
    noise_variance: JAXArray = 0.0,
    *,
    mu: JAXArray | None = None,
    PhiT_Phi: JAXArray | None = None,
    PhiT_y: JAXArray | None = None,
    jitter: float | None = None,
):
    """Compute log p(y | Phi, mu, cov_root, noise_variance) where

        y ~ N(Phi w, noise_variance I)
        w ~ N(mu, cov_root @ cov_root.T)

    Here diagonal term is noise_variance (observational noise) PLUS jitter
    (numerical stability, always added unless set to zero).

    If basisfunctions phi(X) do not depend on hyperparameters, PhiT_Phi = Phi.T @ Phi
    and PhiT_y = Phi.T @ y can be precomputed and passed in for efficiency.
    """

    y = jnp.asarray(y)
    Phi = jnp.asarray(Phi)

    N, M = Phi.shape
    if mu is None:
        mu = jnp.zeros((M,), dtype=Phi.dtype)
    else:
        mu = jnp.asarray(mu)

    # precompute Phi^T Phi if needed
    if PhiT_Phi is None:
        PhiT_Phi = Phi.T @ Phi

    # Phi^T y (original y)
    if PhiT_y is None:
        PhiT_y = Phi.T @ y

    # residuals
    y_resid = y - Phi @ mu
    PhiT_y_resid = PhiT_y - PhiT_Phi @ mu

    # run latent posterior on residuals
    m_z, L_z_post, Lc, b, _, _, sigma2 = posterior_latent(
        y=y_resid,
        Phi=Phi,
        cov_root=cov_root,
        noise_variance=noise_variance,
        PhiT_Phi=PhiT_Phi,
        PhiT_y=PhiT_y_resid,
        jitter=jitter,
        return_aux=True,
    )

    N = y.shape[0]
    R = m_z.shape[0]

    logdet_Z = 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))
    logdet_K = (N - R) * jnp.log(sigma2) + logdet_Z

    quad = (1.0 / sigma2) * (y_resid @ y_resid - b @ m_z)
    norm = N * jnp.log(2.0 * jnp.pi)

    return -0.5 * (logdet_K + quad + norm)


def mercer_from_blr(phi, cov_root) -> Mercer:
    class _(Mercer):
        def compute_phi(self, X: JAXArray) -> JAXArray:
            return phi(X)

        def compute_weights_root(self) -> JAXArray:
            return cov_root

    return _()


class BLRState(eqx.Module):
    X: JAXArray
    Phi: JAXArray
    PhiT_Phi: JAXArray


class _DummySolver(eqx.Module):
    def __init__(self, *args, **kwargs):
        pass


class BayesianLinearRegressor(GaussianProcess):
    """A Bayesian linear regression model implemented as a Gaussian process

    Model:

        y ~ N(Phi w, noise_variance I)      [DATA space]
        w ~ N(mu, cov_root @ cov_root.T)    [WEIGHT space]

    That's it.

    For computations however we express this in latent space

        w ~ mu + cov_root z
        z ~ N(0, I)                         [LATENT space]

    Here the dimensions involved are:
        y:      (N,)
        Phi:    (N, M)
        w:      (M,)
        z:      (R,)   with R not necessarily equal to M
    """

    phi: Callable[[JAXArray], JAXArray]
    mu: JAXArray
    cov_root: JAXArray
    noise_variance: JAXArray

    state: BLRState

    def __init__(
        self,
        phi: Callable[[JAXArray], JAXArray],
        X: JAXArray,
        *,
        mu: JAXArray | None = None,
        cov_root: JAXArray | None = None,
        noise_variance: float | None = 0.0,
        solver=None,
    ):
        Phi = jax.vmap(phi)(X)
        PhiT_Phi = Phi.T @ Phi
        self.state = BLRState(X, Phi, PhiT_Phi)  # precompute

        N, M = Phi.shape
        dtype = Phi.dtype

        mu = mu if mu is not None else jnp.zeros((M,), dtype=X.dtype)
        cov_root = cov_root if cov_root is not None else jnp.eye(M, dtype=dtype)
        noise_variance = noise_variance if noise_variance is not None else 0.0

        self.phi = phi
        self.mu = mu
        self.cov_root = cov_root
        self.noise_variance = noise_variance

        def mean(X_):
            return phi(X_) @ mu

        kernel = mercer_from_blr(self.phi, self.cov_root)
        diag = self.noise_variance

        solver = solver if solver is not None else _DummySolver

        super().__init__(
            kernel=kernel, X=X, diag=diag, mean=mean, solver=solver
        )

    @property
    def variance(self) -> JAXArray:
        A = self.state.Phi @ self.cov_root
        return jnp.sum(A * A, axis=1) + self.noise.diagonal()

    @property
    def covariance(self) -> JAXArray:
        A = self.state.Phi @ self.cov_root
        return A @ A.T + self.noise

    def log_probability(self, y: JAXArray) -> JAXArray:
        return log_probability(
            y=y,
            Phi=self.state.Phi,
            cov_root=self.cov_root,
            noise_variance=self.noise_variance,
            mu=self.mu,
            PhiT_Phi=self.state.PhiT_Phi,
        )

    def condition(
        self,
        y: JAXArray,
        X_test: JAXArray | None = None,
        kernel=None,
        include_mean: bool = True,
    ) -> ConditionResult:
        """Compute the posterior GP conditioned on data y.

        Posterior is over the latent function (noise absorbed into weights).
        """
        if kernel is not None:
            raise NotImplementedError(
                "Conditioning with a different kernel is not yet implemented for BLR."
            )

        if not include_mean:
            raise NotImplementedError(
                "Conditioning without including the mean is not yet implemented for BLR."
            )

        Phi = self.state.Phi

        # posterior over weights w | y
        m_w, L_w_post = posterior_weight(
            y=y,
            Phi=Phi,
            mu=self.mu,
            cov_root=self.cov_root,
            noise_variance=self.noise_variance,
            PhiT_Phi=self.state.PhiT_Phi,
        )

        # coordinates for posterior GP
        X_new = self.state.X if X_test is None else X_test

        # Posterior GP (latent): no observation noise
        gp = BayesianLinearRegressor(
            phi=self.phi,
            X=X_new,
            mu=m_w,
            cov_root=L_w_post,
            noise_variance=0.0,
        )

        # Marginal log-likelihood under the prior BLR model
        log_prob = log_probability(
            y=y,
            Phi=Phi,
            cov_root=self.cov_root,
            noise_variance=self.noise_variance,
            mu=self.mu,
            PhiT_Phi=self.state.PhiT_Phi,
            PhiT_y=Phi.T @ y,
        )

        return ConditionResult(log_prob, gp)

    def sample(
        self,
        key,
        X_test: JAXArray | None = None,
        shape: tuple = (),
    ) -> JAXArray:
        if X_test is None:
            Phi = self.state.Phi
        else:
            Phi = jax.vmap(self.phi)(X_test)

        def _one_sample(k):
            return sample(k, Phi, self.cov_root) + Phi @ self.mu

        if shape == ():
            return _one_sample(key)

        keys = jax.random.split(key, int(jnp.prod(jnp.array(shape))))
        samples = jax.vmap(_one_sample)(keys)
        return samples.reshape(shape + samples.shape[1:])


def blr_from_mercer(
    kernel: Mercer,
    X: JAXArray,
    noise_variance: JAXArray | None = None,
    **kwargs,
) -> BayesianLinearRegressor:
    return BayesianLinearRegressor(
        phi=kernel.compute_phi,
        X=X,
        cov_root=kernel.compute_weights_root(),
        noise_variance=noise_variance,
        **kwargs,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tinygp.solvers.direct import DirectSolver

    from gp.periodic import SPACK
    from utils.jax import vk

    fs = 16000

    noise_variance = 0.5

    T = 5.0
    t1, t2 = 0.0, 3.0
    d = 1

    N = int(T * fs / 1000)
    J = int((fs / 2) / (1000.0 / T))  # number of harmonics
    t, dt = jnp.linspace(-T, T, N * 4, retstep=True)

    spack = SPACK(d=d, period=T, J=J, t1=t1, t2=t2)

    blr = blr_from_mercer(spack, t, noise_variance=noise_variance)

    directblr = blr_from_mercer(
        spack, t, noise_variance=noise_variance, solver=DirectSolver
    )

    du = blr.sample(vk())
    u = jnp.cumsum(du) * dt

    plt.title("Sample from BLR with SPACK kernel")

    cgp = blr.condition(du, t).gp
    # cgp = GaussianProcess.condition(directblr, du, t).gp

    plt.plot(t, cgp.mean, label="posterior mean")
    for _ in range(4):
        plt.plot(t, cgp.sample(vk()), label="posterior sample")

    plt.plot(t, du, ".", c="black", label="data du")
    # plt.plot(t, u)

    plt.legend()
    plt.show()

    tau = jnp.linspace(T, T * 3, N)
    ocgp = blr.condition(du, tau).gp

    plt.title("Posterior at new points")
    plt.plot(tau, ocgp.mean, label="posterior mean")
    for _ in range(4):
        plt.plot(tau, ocgp.sample(vk()), label="posterior sample")

    plt.legend()
    plt.show()

    # Test our BLR implementation against direct (cholesky) implementation
    # variance

    new = blr.variance
    truth = GaussianProcess.variance.__get__(directblr)
    print("Variance max diff:", jnp.max(jnp.abs(new - truth)))  # ok

    # covariance
    new = blr.covariance
    truth = GaussianProcess.covariance.__get__(directblr)
    print("Covariance max diff:", jnp.max(jnp.abs(new - truth)))  # ok

    # log_probability
    new = blr.log_probability(du)
    truth = GaussianProcess.log_probability.__get__(directblr)(du)
    print(
        "Log prob diff:", jnp.abs(new - truth)
    )  # ~1e-6 from jitter applied twice
