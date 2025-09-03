"""
Hilbert kernels are reduced-rank approximations to Spectral kernels.
Implementation is based on https://arxiv.org/pdf/1401.5508 and Riutort-Mayol et al. (2020).

Note: we don't subclass :class:`Spectral` because
- :class:`Spectral` assume L1 distance by default and the spectral density formulas are based on L2 distance -- we impose that here
- Not all Spectral kernels have a known spectral density (e.g. periodic) and we want to define sum and product of Spectral kernels
- :class:`QuasiSep` does the same
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray

from .mercer import Mercer
from .spectral import Spectral
from .util import require_1d


def compute_S(M: tuple[int]):
    """Make set of tuples S \in N^(M*, D) where M* = prod(M) [Riutort-Mayol et al., 2020, p. 8]"""
    grids = jnp.meshgrid(*[jnp.arange(1, m + 1) for m in M], indexing="ij")
    return jnp.stack(grids, axis=-1).reshape(-1, len(M))  # (M*, D)


def compute_sqrt_lambda(M: tuple[int], L: tuple[float]):
    """Compute sqrt(lambda_j) [Riutort-Mayol et al., 2020, p. 8 (9)]"""
    assert len(M) == len(L)
    S = compute_S(M)
    L = jnp.asarray(L)
    return jnp.pi * S / (2.0 * L)  # (M*, D)


class Hilbert(Mercer):
    """Kernel using Spectral space approximation on D-dim input space

    Args:
    kernel: Spectral kernel with spectral_density method defined on R^D
        M: int or (D,): Number of basis functions per dimension
        L: float or (D,): Domain half-size per dimension (Dirichlet boundary conditions on [-L_d, L_d])
        D: int: Number of input dimensions (inferred if M and L are arrays, or if M and L are scalars, promote to D dimensions)

    """

    kernel: Spectral
    M: int | tuple[int] = eqx.field(static=True)
    L: float | tuple[float] = eqx.field(static=True)
    D: int = eqx.field(default=None, static=True)

    def __post_init__(self):
        M = require_1d(self.M)
        L = require_1d(self.L)
        D = self.D

        if D is None:
            if M.size > 1:
                D = M.size
            if L.size > 1:
                if D is not None and L.size != D:
                    raise ValueError("Inconsistent dimensions between M and L")
                D = L.size
            if D is None:
                D = 1
        else:
            if M.size > 1 and M.size != D:
                raise ValueError("M length != ndim")
            if L.size > 1 and L.size != D:
                raise ValueError("L length != ndim")

        M = jnp.broadcast_to(M, (D,))
        L = jnp.broadcast_to(L, (D,))

        def static_hashable(a):  # keep equinox happy
            return tuple(a.tolist())

        object.__setattr__(self, "M", static_hashable(M))
        object.__setattr__(self, "L", static_hashable(L))
        object.__setattr__(self, "D", D)

    def compute_phi(self, X: JAXArray) -> JAXArray:
        """Compute features Phi(X) [Riutort-Mayol et al., 2020, p. 8 (9)]

        Uses M* = prod(M) basis functions in D dimensions and Dirichlet boundary conditions on [-L_1, L_1] x ... x [-L_D, L_D]

        Args:
            X: () or (D,) a single input point
        """
        X = require_1d(X)  # (D,)
        sl = compute_sqrt_lambda(self.M, self.L)
        L = jnp.asarray(self.L)[None, :]  # (1, D)
        phi = jnp.prod(jnp.sin(sl * (X + L)) / jnp.sqrt(L), axis=1)  # (M*,)
        return phi  # (M*,)

    def compute_weights_root(self) -> JAXArray:
        # We probably didn't need to know D as a property if not for this function
        sl = compute_sqrt_lambda(self.M, self.L)

        S = jax.vmap(self.kernel.log_spectral_density)(sl)  # (M*,)
        S = jnp.exp(
            0.5 * S
        )  # calculate sqrt of spectral density as we need to return a root (Cholesky) of the weights matrix

        # TODO(mvsoom): return cola.ops.Diagonal(S); is worth it for 2D and higher as number of basis functions grows like prod(M)
        return jnp.diag(S)
