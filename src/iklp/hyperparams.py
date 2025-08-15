# %%
from __future__ import annotations

from typing import get_type_hints

import jax
import jax.numpy as jnp
from flax import struct

from .mercer import psd_svd
from .util import _periodic_kernel_batch


@struct.dataclass
class Hyperparams:
    Phi: jnp.ndarray  # (I,M,r)

    P: int = struct.field(pytree_node=False, default=30)

    alpha: jnp.ndarray = 1.0
    aw: jnp.ndarray = 1.0
    bw: jnp.ndarray = 1.0
    ae: jnp.ndarray = 1.0
    be: jnp.ndarray = 1.0
    lam: jnp.ndarray = 0.1

    smoothness: jnp.ndarray = 100.0

    num_vi_restarts: int = struct.field(pytree_node=False, default=1)
    num_vi_iters: int = struct.field(pytree_node=False, default=30)
    num_epsilon_samples: int = struct.field(pytree_node=False, default=5)

    def __post_init__(self):
        """Make sure all jnp.ndarray fields follow jax_enable_x64()"""
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        hints = get_type_hints(type(self))
        for name, ann in hints.items():
            if ann is jnp.ndarray:
                value = getattr(self, name)
                if value is not None:
                    casted_value = jnp.asarray(value, dtype=dtype)
                    object.__setattr__(
                        self,
                        name,
                        casted_value,
                    )


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

# %%
