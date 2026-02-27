import jax.numpy as jnp
import numpyro.distributions.transforms as npt
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    ParameterTag,
    T,
    _safe_assert,
)
from jax.experimental import checkify


class Simplex(Parameter[T]):
    """Parameter that lies on the probability simplex (nonnegative, sums to 1)."""

    def __init__(self, value: T, tag: ParameterTag = "simplex", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        # Only perform validation in non-JIT contexts
        if (
            not isinstance(value, jnp.ndarray)
            or getattr(value, "aval", None) is not None
        ):
            _safe_assert(_check_is_simplex, self.value)


@checkify.checkify
def _check_is_simplex(value: T) -> None:
    checkify.check(
        jnp.all(value >= 0),
        "value needs to be non-negative (simplex), got {value}",
        value=value,
    )
    s = jnp.sum(value)
    checkify.check(
        jnp.isfinite(s) & jnp.allclose(s, 1.0, rtol=1e-5, atol=1e-6),
        "value needs to sum to 1 (simplex), got sum={s}",
        s=s,
    )


DEFAULT_BIJECTION["simplex"] = npt.StickBreakingTransform()
