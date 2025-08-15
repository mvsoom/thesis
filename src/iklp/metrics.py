import jax
import jax.numpy as jnp
from flax import struct

from iklp.mercer_op import build_operator
from iklp.psi import psi_matvec

from .state import Expectations, VIState, compute_expectations
from .vi import compute_elbo_bound


@struct.dataclass
class StateMetrics:
    elbo: jnp.ndarray  # ()

    E: Expectations
    a: jnp.ndarray  # (P,)

    epsilon_samples: jnp.ndarray  # (epsilon_samples, M)


def compute_epsilon_samples(state: VIState) -> jnp.ndarray:
    e = psi_matvec(state.xi.delta_a, state.data.x)

    E = compute_expectations(state)
    Omega = build_operator(E.nu_e, E.nu_w * E.theta, state.data)

    # See https://chatgpt.com/c/689e1037-44e4-8323-a074-cdbe9b06b0e5

    # TODO: test this
    num_epsilon_samples = state.data.h.num_epsilon_samples

    # Hash the state to get a random key
    hash = 1000
    key = jax.random.PRNGKey(hash)

    M = state.data.h.Phi.shape[1]
    z = jax.random.normal(key, (num_epsilon_samples, M))

    return z


def compute_metrics(state: VIState) -> StateMetrics:
    # TODO: we need to compute the auxes, elbo and expectations need this
    elbo = compute_elbo_bound(state)
    E = compute_expectations(state)
    a = state.xi.delta_a
    epsilon_samples = compute_epsilon_samples(state)
    return StateMetrics(elbo=elbo, E=E, a=a, epsilon_samples=epsilon_samples)
