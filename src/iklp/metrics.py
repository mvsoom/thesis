import jax.numpy as jnp
from flax import struct

from .state import Expectations, LatentVars, VIState, compute_expectations
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


def compute_power_distibution(z):
    """Return normalized power distribution for (noise, kernel_1, ..., kernel_I)"""
    power = jnp.concatenate((jnp.array([z.nu_e]), z.nu_w * z.theta))
    return power / jnp.sum(power)  # (I+1,)


def compute_state_power_distribution(state):
    E = compute_expectations(state)
    z = LatentVars(E.theta, E.nu_w, E.nu_e, None)
    return compute_power_distibution(z)
