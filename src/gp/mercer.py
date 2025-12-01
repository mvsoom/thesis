"""
Mercer kernel with general rectangular weights W = L L^T with L (M, R), R may not equal M

TODO(mvsoom): documentation
TODO(mvsoom): use COLA to annotate JAX arrays
"""
# %%

from __future__ import annotations

from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy import linalg
from tinygp.helpers import JAXArray
from tinygp.kernels import Kernel
from tinygp.kernels.base import Kernel


class Mercer(Kernel):
    """Low-rank Mercer kernel

    The kernel is represented a degenerate rank M kernel of the form

        k(x, x') = phi(x)^T W phi(x')

    where phi(x) is an M-dimensional feature vector, and W is an (M, M) weight matrix with root L (M, R) such that W = L L^T (where R may not equal M).

    Here

        compute_phi(X) <-> phi(x)
        compute_weights() <-> W
        compute_weights_root() <-> L where W = L L^T
    """
    def compute_phi(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError  # (M,)

    def compute_weights_root(self) -> JAXArray:
        raise NotImplementedError  # (M, M)

    def compute_weights(self) -> JAXArray:
        L = self.compute_weights_root()  # (M, R)
        W = L @ L.T
        return W  # (M, M)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        phi1 = self.compute_phi(X1)
        phi2 = self.compute_phi(X2)

        L = self.compute_weights_root()

        f1 = L.T @ phi1
        f2 = L.T @ phi2

        return jnp.dot(f1, f2)  # ()

    def evaluate_diag(self, X):
        phi = self.compute_phi(X)
        L = self.compute_weights_root()
        f = L.T @ phi
        return jnp.dot(f, f)  # ()

    def matmul(
        self,
        X1: JAXArray,
        X2: JAXArray | None = None,
        y: JAXArray | None = None,
    ) -> JAXArray:
        if y is None:
            assert X2 is not None
            y = X2
            X2 = None

        if X2 is None:
            X2 = X1

        Phi1 = jax.vmap(self.compute_phi)(X1)  # (N1, R)
        Phi2 = jax.vmap(self.compute_phi)(X2)  # (N2, R)
        L = self.compute_weights_root()  # (R, R)
        return Phi1 @ (L @ (L.T @ (Phi2.T @ y)))

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


def sample(key: jax.random.KeyArray, kernel: Mercer, X: JAXArray) -> JAXArray:
    """Sample from GP(0, k: Mercer) with Mercer kernel at inputs X"""
    Phi = jax.vmap(kernel.compute_phi)(X)  # (N, M)
    L = kernel.compute_weights_root()  # (M, R) where K may != M

    R = L.shape[1]
    z = jax.random.normal(key, (R,))

    f = Phi @ (L @ z)
    return f  # (N,)


def posterior_latent(
    y: JAXArray,
    kernel: Mercer,
    X: JAXArray,
    noise_variance: JAXArray,
    *,
    jitter: float | None = None,
    PhiT_y: JAXArray | None = None,
    PhiT_Phi: JAXArray | None = None,
    return_aux: bool = False,
):
    """Calculate mean and covariance of z | y ~ N(m_z, Sigma_z) for a Mercer kernel GP

    If basisfunctions phi(X) do not depend on hyperparameters, PhiT_Phi = Phi.T @ Phi and PhiT_y = Phi.T @ y can be precomputed and passed in for efficiency.
    """

    y = jnp.asarray(y)
    Phi = jax.vmap(kernel.compute_phi)(X)  # (N, M)
    Lw = kernel.compute_weights_root()  # (M, R)
    R = Lw.shape[1]

    if jitter is None:
        jitter = jnp.sqrt(jnp.finfo(y.dtype).eps)

    sigma2 = noise_variance + jitter

    if PhiT_Phi is None:
        PhiT_Phi = Phi.T @ Phi  # (M, M)
    if PhiT_y is None:
        PhiT_y = Phi.T @ y  # (M,)

    Z = sigma2 * jnp.eye(R) + Lw.T @ PhiT_Phi @ Lw  # (R, R)

    Lc, lower = jax.scipy.linalg.cho_factor(Z, lower=True, check_finite=False)

    b = Lw.T @ PhiT_y  # (R,)
    m_z = jax.scipy.linalg.cho_solve((Lc, lower), b, check_finite=False)

    I_R = jnp.eye(R, dtype=y.dtype)
    invZ = jax.scipy.linalg.cho_solve((Lc, lower), I_R, check_finite=False)
    Sigma_z = sigma2 * invZ

    if return_aux:
        return m_z, Sigma_z, Lc, b, PhiT_Phi, PhiT_y

    return m_z, Sigma_z


def log_probability(
    y: JAXArray,
    kernel: Mercer,
    X: JAXArray,
    noise_variance: JAXArray,
    *,
    jitter: float | None = None,
    PhiT_y: JAXArray | None = None,
    PhiT_Phi: JAXArray | None = None,
) -> JAXArray:
    """Compute log p(y | GP(0, k + noise_variance*I)) for a Mercer kernel k.

    From log likelihood formulas in Section 3.2 in [1]

    Note 1: if basisfunctions phi(X) do not depend on hyperparameters, then PhiT_Phi = Phi.T @ Phi should be precomputed and passed in for efficiency.
    This reduces computational cost from O(N M² + M³) to O(M³), where N is number of data points and M is number of basis functions.
    Same goes for the projection of the data y on the basisfunctions: PhiT_y = Phi.T @ y.

    Note 2: if Phi.T @ y can be expressed as inner products of sinusoids with the data, a further speedup is possible using FFTs [1] or NUFFTs like jax-finufft package. This reduces O(N M) to O(N log N + M).

        [1] Solin, A., & Särkkä, S. (2020). Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), 419-446.
    """

    m_z, Sigma_z, Lc, b, PhiT_Phi, PhiT_y = posterior_latent(
        y,
        kernel,
        X,
        noise_variance,
        jitter=jitter,
        PhiT_y=PhiT_y,
        PhiT_Phi=PhiT_Phi,
        return_aux=True,
    )

    N = y.shape[0]
    R = m_z.shape[0]
    sigma2 = noise_variance + (
        jnp.sqrt(jnp.finfo(y.dtype).eps) if jitter is None else jitter
    )

    logdet_Z = 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))
    logdet_Kreg = (N - R) * jnp.log(sigma2) + logdet_Z

    quad = (1.0 / sigma2) * (y @ y - b @ m_z)
    norm = N * jnp.log(2.0 * jnp.pi)

    return -0.5 * (logdet_Kreg + quad + norm)


if __name__ == "__main__":
    from tinygp.gp import GaussianProcess
    from tinygp.kernels import Kernel

    key = jax.random.key(420)

    class KitchenSink(Mercer):
        """Random-test Mercer kernel with arbitrary rectangular L.

        phi(x) = Phi_dict[x] where Phi_dict is a (N, M) random feature matrix.
        Lw     is (M, R) rectangular weight root (W = Lw Lw^T).
        """

        Phi_dict: JAXArray  # (N, M)
        Lw: JAXArray  # (M, R)

        def compute_phi(self, X: JAXArray) -> JAXArray:
            # Convert X to int without breaking JAX tracing
            idx = jnp.asarray(X, jnp.int32)
            return self.Phi_dict[idx]  # (M,)

        def compute_weights_root(self) -> JAXArray:
            return self.Lw  # (M, R)

    N = 500  # number of inputs
    M = 30  # feature dimension
    R = 10  # latent rank

    k1, k2, k3 = jax.random.split(key, 3)

    Phi_dict = jax.random.normal(k1, (N, M))
    Lw = jax.random.normal(k2, (M, R)) * 0.3

    kernel = KitchenSink(Phi_dict, Lw)

    X = jnp.arange(N)
    y = sample(k3, kernel, X)

    # TinyGP GP
    diag = 1e-6

    gp = GaussianProcess(kernel, X, diag=diag)
    tiny_logp = gp.log_probability(y)

    # Our Mercer likelihood (noise variance = 1e-6)
    ours_logp = log_probability(
        y=y,
        kernel=kernel,
        X=X,
        noise_variance=0.0,
        jitter=diag,
    )

    print("TinyGP log_prob :", float(tiny_logp))
    print("Mercer  log_prob:", float(ours_logp))
    print("Difference        :", float(abs(tiny_logp - ours_logp)))


class Sum(Mercer):
    """Stack features"""

    kernel1: Kernel
    kernel2: Kernel

    def compute_phi(self, X: JAXArray) -> JAXArray:
        phi1 = self.kernel1.compute_phi(X)
        phi2 = self.kernel2.compute_phi(X)
        return jnp.hstack([phi1, phi2])  # (M1 + M2,)

    def compute_weights_root(self) -> JAXArray:
        L1 = self.kernel1.compute_weights_root()  # (M1, R1)
        L2 = self.kernel2.compute_weights_root()  # (M2, R2)

        M1, R1 = L1.shape
        M2, R2 = L2.shape

        Z12 = jnp.zeros((M1, R2), dtype=L1.dtype)
        Z21 = jnp.zeros((M2, R1), dtype=L1.dtype)

        top = jnp.concatenate([L1, Z12], axis=1)  # (M1, R1+R2)
        bot = jnp.concatenate([Z21, L2], axis=1)  # (M2, R1+R2)

        # TODO(mvsoom): return cola.ops.BlockDiagonal
        return jnp.concatenate([top, bot], axis=0)  # (M1+M2, R1+R2)


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
        return jnp.kron(L1, L2)  # (M1*M2, R1*R2)


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
