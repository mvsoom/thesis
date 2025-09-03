"""
TODO(mvsoom): documentation
TODO(mvsoom): use COLA to annotate JAX arrays
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.scipy import linalg
from tinygp.helpers import JAXArray
from tinygp.kernels import Kernel
from tinygp.kernels.base import Kernel


class Mercer(Kernel):
    def compute_phi(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError  # (M,)

    def compute_weights_root(self) -> JAXArray:
        raise NotImplementedError  # (M, M)

    def compute_weights(self) -> JAXArray:
        L = self.compute_weights_root()
        return L @ L.T

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        L = self.compute_weights_root()
        f1 = self.compute_phi(X1) @ L
        f2 = self.compute_phi(X2) @ L
        return f1 @ f2.T

    def evaluate_diag(self, X):
        L = self.compute_weights_root()
        f = self.compute_phi(X) @ L
        return jnp.sum(f**2)

    def __add__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, Mercer):
            return Sum(self, other)
        return super().__add__(other)

    def __radd__(self, other) -> Kernel:
        if isinstance(other, Mercer):
            return Sum(other, self)
        return super().__radd__(other)

    def __mul__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, Mercer):
            return Product(self, other)
        elif isinstance(other, Kernel):
            return super().__mul__(other)
        else:  # assume scalar
            return Scale(factor=other, kernel=self)

    def __rmul__(self, other) -> Kernel:
        if isinstance(other, Mercer):
            return Product(other, self)
        elif isinstance(other, Kernel):
            return super().__rmul__(other)
        else:
            return Scale(factor=other, kernel=self)


class Sum(Mercer):
    """Stack features"""

    kernel1: Kernel
    kernel2: Kernel

    def compute_phi(self, X: JAXArray) -> JAXArray:
        phi1 = self.kernel1.compute_phi(X)
        phi2 = self.kernel2.compute_phi(X)
        return jnp.hstack([phi1, phi2])  # (M1 + M2,)

    def compute_weights_root(self) -> JAXArray:
        L1 = self.kernel1.compute_weights_root()
        L2 = self.kernel2.compute_weights_root()

        M1, M2 = L1.shape[0], L2.shape[0]
        Z12 = jnp.zeros((M1, M2), dtype=L1.dtype)
        Z21 = jnp.zeros((M2, M1), dtype=L1.dtype)

        # TODO(mvsoom): return cola.ops.BlockDiagonal
        return jnp.block([[L1, Z12], [Z21, L2]])


class Scale(Mercer):
    factor: JAXArray
    kernel: Kernel

    def compute_phi(self, X: JAXArray) -> JAXArray:
        return self.kernel.compute_phi(X)

    def compute_weights_root(self) -> JAXArray:
        return jnp.sqrt(self.factor) * self.kernel.compute_weights_root()


class Product(Mercer):
    """Khatri-Rao product of features"""

    kernel1: Mercer
    kernel2: Mercer

    def compute_phi(self, X: JAXArray) -> JAXArray:
        phi1 = self.kernel1.compute_phi(X)
        phi2 = self.kernel2.compute_phi(X)

        # TODO(mvsoom): return cola.ops.Kronecker
        return jnp.kron(phi1, phi2)  # (M1*M2,)

    def compute_weights_root(self) -> JAXArray:
        L1 = self.kernel1.compute_weights_root()
        L2 = self.kernel2.compute_weights_root()
        # TODO(mvsoom): return cola.ops.Kronecker
        return jnp.kron(L1, L2)  # (M1*M2, M1*M2)


class Transform(Mercer):
    transform: Callable = eqx.field(static=True)
    kernel: Mercer

    def compute_phi(self, X):
        return self.kernel.compute_phi(self.transform(X))

    def compute_weights_root(self):
        return self.kernel.compute_weights_root()


class Linear(Mercer):
    scale: JAXArray
    kernel: Mercer

    def compute_phi(self, X):
        if jnp.ndim(self.scale) < 2:
            transform = partial(jnp.multiply, self.scale)
        elif jnp.ndim(self.scale) == 2:
            transform = partial(jnp.dot, self.scale)
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        return self.kernel.compute_phi(transform(X))

    def compute_weights_root(self):
        return self.kernel.compute_weights_root()


class Cholesky(Mercer):
    factor: JAXArray
    kernel: Mercer

    def compute_phi(self, X):
        if jnp.ndim(self.factor) < 2:
            transform = partial(jnp.multiply, 1.0 / self.factor)
        elif jnp.ndim(self.factor) == 2:
            transform = partial(
                linalg.solve_triangular, self.factor, lower=True
            )
        else:
            raise ValueError("'factor' must be 0-, 1-, or 2-dimensional")
        return self.kernel.compute_phi(transform(X))

    def compute_weights_root(self):
        return self.kernel.compute_weights_root()


class Subspace(Mercer):
    axis: int | tuple[int, ...] = eqx.field(static=True)
    kernel: Mercer

    def compute_phi(self, X):
        return self.kernel.compute_phi(X[self.axis])

    def compute_weights_root(self):
        return self.kernel.compute_weights_root()
