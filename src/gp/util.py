import jax.numpy as jnp


def require_1d(a):
    a = jnp.atleast_1d(a)
    assert a.ndim == 1, f"Cannot squeeze to 1D, got shape {a.shape}"
    return a


def require_2d(a):
    a = jnp.atleast_2d(a)
    assert a.ndim == 2, f"Cannot squeeze to 2D, got shape {a.shape}"
    return a
