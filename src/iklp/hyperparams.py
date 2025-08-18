from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from utils.jax import maybe32, static_constant

from .mercer import psd_svd
from .util import _periodic_kernel_batch


@struct.dataclass
class Hyperparams:
    """VI hyperparameters

    Both model and simulation hyperparameters are set here.
      * `static_constant`s are specialized on when jitting
      * `maybe32` is used to allow constants to become x32 when jax_enable_x64 is False
        NOTE: Use maybe32() *again* when initializing, e.g. `h = Hyperparams(Phi, aw=maybe32(aw))`
    """

    Phi: jnp.ndarray  # (I,M,r)

    P: int = static_constant(
        30
    )  # Must be static because determines shape of xi.delta_a

    alpha: jnp.ndarray = maybe32(1.0)
    aw: jnp.ndarray = maybe32(1.0)
    bw: jnp.ndarray = maybe32(1.0)
    ae: jnp.ndarray = maybe32(1.0)
    be: jnp.ndarray = maybe32(1.0)
    lam: jnp.ndarray = maybe32(0.1)

    smoothness: float = static_constant(100.0)

    num_vi_restarts: int = static_constant(1)
    num_vi_iters: int = static_constant(30)
    num_epsilon_samples: int = static_constant(5)


def random_periodic_kernel_hyperparams(
    key, I=32, M=512, kernel_kwargs={}, hyper_kwargs={}, return_K=False
) -> Hyperparams:
    """Cannot vmap this over key because the shape of Phi depends on it"""
    noise_floor_db = kernel_kwargs.pop("noise_floor_db", -60.0)

    T = jnp.sort(jax.random.exponential(key, (I,)) * 10)
    K = _periodic_kernel_batch(T, M, **kernel_kwargs)  # (I, M, M)

    Phi = psd_svd(K, noise_floor_db)

    h = Hyperparams(Phi, **hyper_kwargs)

    return (h, K) if return_K else h
