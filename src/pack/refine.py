# %%
"""Utilities to refine LF surrogate priors from exemplar waveforms.

This module factors the learning logic developed in `fit_lf_test.py` into
reusable pieces:

1. Fit a Bayesian linear regressor to each exemplar.
2. Approximate the mixture of posteriors with a single Gaussian via the
   reverse KL solution (``envelope_gaussians``).
3. Instantiate a new ``BayesianLinearRegressor`` that encodes the learned
   surrogate prior and can be sampled/conditioned elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax.numpy as jnp

from gp.blr import BayesianLinearRegressor, blr_from_mercer
from gp.mercer import Mercer


class PosteriorParams(NamedTuple):
    """Sufficient statistics for the Gaussian posterior of an exemplar."""

    mu: jnp.ndarray
    cov_root: jnp.ndarray


def _as_arrays(exemplar: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract time/grid and dU arrays from an exemplar object or mapping."""

    if isinstance(exemplar, Mapping):
        t = exemplar["t"]
        du = exemplar["du"]
    else:
        t = getattr(exemplar, "t")
        du = getattr(exemplar, "du")

    return jnp.asarray(t), jnp.asarray(du)


def infer_exemplar_posterior(
    kernel: Mercer,
    exemplar: Any,
    *,
    noise_variance: float = 1e-4,
) -> PosteriorParams:
    """Fit a BLR posterior to a single exemplar.

    Args:
        kernel: Mercer kernel object exposing ``compute_phi`` and
            ``compute_weights_root`` (e.g., ``pack.periodic.SPACK``).
        exemplar: Object or mapping with ``t`` and ``du`` entries.
        noise_variance: Observation noise variance for the BLR fit.
    """

    t, du = _as_arrays(exemplar)
    gp = blr_from_mercer(kernel, t, noise_variance=noise_variance)
    conditioned = gp.condition(du).gp
    return PosteriorParams(mu=conditioned.mu, cov_root=conditioned.cov_root)


def envelope_gaussians(
    posteriors: Sequence[PosteriorParams],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Match a single Gaussian to several Gaussian posteriors.

    The returned parameters ``(mu_star, Sigma_star)`` minimize the mean
    reverse KL divergence (``KL(p_i || q)``) over the supplied
    ``posteriors = {p_i}``.
    """

    if not posteriors:
        raise ValueError("Need at least one posterior to build an envelope.")

    mus = jnp.stack([p.mu for p in posteriors], axis=0)
    cov_roots = jnp.stack([p.cov_root for p in posteriors], axis=0)

    mu_star = jnp.mean(mus, axis=0)

    sigma_terms = cov_roots @ jnp.swapaxes(cov_roots, -2, -1)
    mean_outer = mus[..., None] * mus[:, None, :]
    second_moment = jnp.mean(sigma_terms + mean_outer, axis=0)

    Sigma_star = second_moment - mu_star[:, None] * mu_star[None, :]
    return mu_star, Sigma_star


def learn_surrogate_blr(
    kernel: Mercer,
    exemplars: Sequence[Any],
    *,
    evaluation_times,
    noise_variance: float = 1e-4,
    envelope_jitter: float = 1e-9,
    enforce_zero_mean: bool = False,
) -> BayesianLinearRegressor:
    """Train a surrogate BLR prior from exemplar dU trajectories."""
    if not exemplars:
        raise ValueError("No exemplars supplied.")

    posteriors = [
        infer_exemplar_posterior(
            kernel, exemplar, noise_variance=noise_variance
        )
        for exemplar in exemplars
    ]

    mu_star, Sigma_star = envelope_gaussians(posteriors)

    if enforce_zero_mean:
        # KL-optimal envelope under mean = 0 constraint:
        # Sigma <- E[xx^T] = Cov + mu mu^T
        Sigma_star = Sigma_star + jnp.outer(mu_star, mu_star)
        mu_star = jnp.zeros_like(mu_star)

    dim = Sigma_star.shape[0]
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)
    Sigma_star += envelope_jitter * jnp.eye(dim, dtype=Sigma_star.dtype)
    Sigma_star_root = jnp.linalg.cholesky(Sigma_star)

    evaluation_times = jnp.asarray(evaluation_times)

    return BayesianLinearRegressor(
        kernel.compute_phi,
        evaluation_times,
        mu=mu_star,
        cov_root=Sigma_star_root,
    )
