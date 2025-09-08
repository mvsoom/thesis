import jax
import jax.numpy as jnp
from flax import struct

from iklp.mercer_op import sample_parts_given_observation
from iklp.psi import psi_matvec
from utils.jax import maybe32

from .state import (
    Auxiliaries,
    Expectations,
    LatentVars,
    VIState,
    compute_auxiliaries,
)
from .vi import compute_elbo_bound_aux


@struct.dataclass
class StateMetrics:
    elbo: jnp.ndarray  # ()

    E: Expectations
    a: jnp.ndarray  # (P,)

    signals: jnp.ndarray  # (h.num_metrics_samples, M)

    i: jnp.ndarray = maybe32(0)  # ()
    improvement: jnp.ndarray = jnp.nan  # ()


def compute_metrics(key, state: VIState) -> StateMetrics:
    aux = compute_auxiliaries(state)

    elbo = compute_elbo_bound_aux(state, aux)
    E = aux.E
    a = state.xi.delta_a

    signals = compute_signals_aux(
        key, state, aux, state.data.h.num_metrics_samples
    )

    return StateMetrics(elbo=elbo, E=E, a=a, signals=signals)


def compute_new_metrics(key, state: VIState, old: StateMetrics) -> StateMetrics:
    metrics = compute_metrics(key, state)
    return metrics.replace(
        i=old.i + 1,
        improvement=(metrics.elbo - old.elbo) / jnp.abs(old.elbo),
    )


def compute_signals(key, state: VIState, num_samples=5):
    aux = compute_auxiliaries(state)
    return compute_signals_aux(key, state, aux, num_samples=num_samples)


def compute_signals_aux(key, state: VIState, aux: Auxiliaries, num_samples=5):
    """Sample from p(signal | x, z = E[z|xi])

    In other words, samples come from the Gaussian process which is conditioned on the **expectation values** of the latent variables `z` (thus the latter are not sampled themselves).
    """
    e = psi_matvec(state.xi.delta_a, state.data.x)  # (M,)

    keys = jax.random.split(key, num_samples)

    signals, _ = jax.vmap(
        lambda k: sample_parts_given_observation(aux.Omega, e, k)
    )(keys)

    return signals  # (num_samples, M)


def compute_power_distibution(z):
    """Return normalized power distribution for (noise, kernel_1, ..., kernel_I)"""
    power = jnp.concatenate((jnp.array([z.nu_e]), z.nu_w * z.theta))
    return power / jnp.sum(power)  # (I+1,)


def compute_metrics_power_distribution(metrics: StateMetrics):
    E = metrics.E
    z = LatentVars(E.theta, E.nu_w, E.nu_e, None)
    return compute_power_distibution(z)
