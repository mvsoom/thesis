import jax.numpy as jnp
from flax import struct

from .state import Expectations, VIState, compute_expectations
from .vi import compute_elbo_bound


@struct.dataclass
class StateMetrics:
    elbo: jnp.ndarray  # ()

    E: Expectations
    a: jnp.ndarray  # (P,)

    # epsilon_samples: jnp.ndarray  # (epsilon_samples, M)


def compute_metrics(state: VIState) -> StateMetrics:
    # TODO: we need to compute the auxes, elbo and expectations need this
    elbo = compute_elbo_bound(state)
    E = compute_expectations(state)
    a = state.xi.delta_a
    # epsilon_samples = compute_epsilon_samples(state)
    return StateMetrics(elbo=elbo, E=E, a=a)
