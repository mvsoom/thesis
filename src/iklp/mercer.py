"""Compute Mercer expansion of a batch of PSD matrices using SVD"""

import jax.numpy as jnp
import numpy as np


def sqrt_clip(x):
    return jnp.sqrt(jnp.clip(x, a_min=0.0, a_max=None))


def psd_svd(K, noise_floor_db=-jnp.inf):
    """Compute Mercer expansion of the K = [..., M, M] matrix using SVD

    Returns the reduced rank approximation Phi = [..., M, r], where r is the rank common to all kernels K_i, such that K_i ~= Phi_i @ Phi_i^T for each K_i = [i, M, M].

    The common rank is determined according to the noise floor in dB to be the largest rank necessary across all kernel matrices to achieve the desired noise floor.

    Since Ks are normalized to unit power, a meaningful value of noise_floor_db is -60.0 dB.

    Note: even for `noise_floor_db == -jnp.inf` this can still return `r < M` because jnp.linalg.eigh() can return negative eigenvalues for nearly singular K_i's. In this case, the rank is determined by the number of positive eigenvalues.
    """
    w, U = jnp.linalg.eigh(K)  # w: [..., M], U: [..., M, M]
    w = w[..., ::-1]  # [..., M]
    U = U[..., :, ::-1]  # [..., M, M]

    cutoff = 10 ** (noise_floor_db / 10)
    ratios = w / jnp.max(w, axis=-1, keepdims=True)  # [..., M]
    r_per = jnp.sum(ratios > cutoff, axis=-1)  # [...]
    r = jnp.max(r_per)

    U_r = U[..., :, :r]  # [..., M, r]
    w_r = w[..., :r]  # [..., r]

    # Some of the w_r values can be ~= 0^- if they already converged well with rank r, so we zero these associated eigenvectors out
    Phi = U_r * sqrt_clip(w_r)[..., None, :]  # [..., M, r]
    return Phi


def psd_svd_fixed(K, rank):
    """
    Compute Mercer expansion of PSD matrices with fixed rank.

    Args:
    K : ndarray, shape (..., M, M)
        Batch of PSD kernel matrices.
    rank : int
        Fixed rank to truncate to. Must be <= M.

    Returns:
    Phi : ndarray, shape (..., M, rank)
        Reduced-rank approximation factor such that
        K_i ≈ Phi_i @ Phi_i.T for each K_i.
    energy : ndarray, shape (...)
        Fraction of spectral energy captured by the top-`rank` eigenvalues.
    """
    w, U = np.linalg.eigh(K)  # w: [..., M], U: [..., M, M]
    w = np.flip(w, axis=-1)  # sort descending
    U = np.flip(U, axis=-1)

    w = np.clip(w, 0.0, None)  # clip negative eigenvalues

    U_r = U[..., :, :rank]  # [..., M, rank]
    w_r = w[..., :rank]  # [..., rank]

    Phi = U_r * np.sqrt(w_r)[..., None, :]  # [..., M, rank]

    # normalized energy coverage
    total = np.sum(w, axis=-1)
    covered = np.sum(w_r, axis=-1)
    energy = covered / (total + 1e-15)  # avoid div by 0

    return Phi, energy
