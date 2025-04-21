"""Infinite Kernel Linear Prediction (IKLP)

See Yoshii, K. & Goto, M. (2013).
Infinite kernel linear prediction for joint estimation
of spectral envelope and fundamental frequency.
ICASSP 2013, 463‑467. DOI: 10.1109/ICASSP.2013.6637690
"""

from jax.numpy import array, asarray, concatenate, tril, zeros
from jax.scipy.linalg import toeplitz


def build_Psi(M, a):
    """Eq. (3) in Yoshii & Goto (2013)"""
    a = asarray(a)
    col = concatenate([array([1.0]), -a, zeros(M - a.size - 1, dtype=a.dtype)])
    full = toeplitz(col)  # Hermitian Toeplitz from first column
    return tril(full)  # keep only the lower‑triangular part


def build_X(x, P):
    """Eq. (3) in Yoshii & Goto (2013)"""
    x = asarray(x)
    col = concatenate([zeros(1, dtype=x.dtype), x[:-1]])
    row = zeros(P, dtype=x.dtype)
    return toeplitz(col, row)
