# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from .mercer import psd_svd
from .util import _periodic_kernel_batch


@struct.dataclass
class Hyperparams:
    Phi: jnp.ndarray  # (I,M,r)

    P: int = 30
    alpha: float = 1.0
    aw: float = 1.0
    bw: float = 1.0
    ae: float = 1.0
    be: float = 1.0
    lam: float = 0.1

    smoothness: float = 100.0
    num_iters: int = 100


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


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)

    h = random_periodic_kernel_hyperparams(key)
    print("Phi shape:", h.Phi.shape)
    print(h)
