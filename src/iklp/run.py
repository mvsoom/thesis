from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from iklp.mercer_op import build_data
from iklp.metrics import StateMetrics, compute_metrics, compute_new_metrics
from iklp.state import (
    VIState,
    init_state,
    init_variational_params,
)
from iklp.vi import vi_step


def vi_run(key, data):
    """Do a single VI optimization with random initialization for a fixed number of iterations"""
    key, k1, k2 = jax.random.split(key, 3)
    xi = init_variational_params(k1, data.h)
    state = VIState(data, xi)
    metrics = compute_metrics(k2, state)

    def body(carry, _):
        key, state, metrics = carry

        new_key, key = jax.random.split(key)
        new_state = vi_step(state)
        new_metrics = compute_new_metrics(key, new_state, old=metrics)

        return (new_key, new_state, new_metrics), new_metrics

    _, accumulated_metrics = jax.lax.scan(
        body, (key, state, metrics), length=data.h.num_vi_iters
    )

    # Preprend first metrics to get shape [n_iter + 1, ...]
    all_metrics = jax.tree_util.tree_map(
        lambda m, am: jnp.concatenate([jnp.expand_dims(m, 0), am], axis=0),
        metrics,
        accumulated_metrics,
    )

    return all_metrics  # leaves now have leading time axis T = n_iter + 1


def vi_frame(key, frame, h):
    """Run VI on a single frame with multiple restarts"""
    data = build_data(frame, h)

    subkeys = jax.random.split(key, h.num_vi_restarts)

    vi_run_vmap = jax.vmap(vi_run, in_axes=(0, None), out_axes=0)

    metrics = vi_run_vmap(subkeys, data)
    return metrics  # (h.num_vi_restarts, h.num_vi_iters, ...)


def vi_frames(key, frames, h):
    """Run VI on multiple frames -- see `vi_frames_batched()`"""
    subkeys = jax.random.split(key, len(frames))

    vi_frame_vmap = jax.vmap(vi_frame, in_axes=(0, 0, None), out_axes=0)

    metrics = vi_frame_vmap(subkeys, frames, h)
    return metrics  # (len(frames), h.num_vi_restarts, h.num_vi_iters, ...)


def vi_frames_batched(key, frames, h, batch_size=None):
    """Batch version of `vi_frames()`

    Args:
        key: jax.random.PRNGKey
        frames: jnp.ndarray shaped (num_frames, M) where M is number of samples per frame
        h: Hyperparams
          *Note*: this function jit-specializes on the shape of `h.P`, `h.num_vi_restarts`, and `h.num_vi_iters`.
        batch_size: int or None. If None, process all frames in one batch.

    Returns:
        metrics: pytree with leaves shaped (num_frames, h.num_vi_restarts, h.num_vi_iters + 1, ...)
          Number of total VI iterations done is `num_frames * h.num_vi_restarts * h.num_vi_iters`.
    """
    n = frames.shape[0]
    batch_size = batch_size or n

    n_batches = (n + batch_size - 1) // batch_size
    keys = jax.random.split(key, n_batches)

    vi_frames_jit = jax.jit(
        vi_frames
    )  # Specializes on batch size, h.P, h.num_vi_restarts, h.num_vi_iters

    chunks = []

    for i in range(n_batches):
        start = i * batch_size
        chunk_metrics = vi_frames_jit(
            keys[i], frames[start : start + batch_size], h
        )

        chunks.append(chunk_metrics)

    # Concatenate along the frames axis (axis=0) for every leaf.
    metrics = jax.tree_util.tree_map(
        lambda *parts: jnp.concatenate(parts, axis=0), *chunks
    )

    return metrics  # (len(frames), h.num_vi_restarts, h.num_vi_iters + 1, ...)


def print_progress(metrics: StateMetrics):
    print(
        f"iter {metrics.i}: elbo = {metrics.elbo:.2f} ({metrics.improvement:+.8f} improvement)"
    )

@partial(jax.jit, static_argnames=("callback",))
def vi_run_criterion(
    key,
    x,
    h,
    max_iters=jnp.inf,
    callback=None,
) -> tuple[VIState, StateMetrics]:
    """Run a single VI run with given data `x` and hyperparameters `h`, where each state is sequentially updated, until the relative improvement in ELBO is (a) below `criterion`, (b) diverges (becomes nan or 0^-)

    `callback` gets called after each state update with the current `StateMetrics`. See `print_progress()` for an example callback function. Use sparingly, because this always triggers an expensive COPY transfer from GPU to host of the metrics of the current state -- NOT the entire state including Phi, the data, etc. If `None`, this gets compiled out as expected.

    Note: this function works with vmap(); it stalls the other lanes until the last one is done, and returns the final state with a batch dimension indexing the lane. Even the callback works with vmap() -- it is called sequentially per lane for each iteration.
    """

    def maybe_callback(metrics):
        if callback:
            jax.debug.callback(callback, metrics, ordered=True)

    key, k1, k2 = jax.random.split(key, 3)
    state = init_state(k1, x, h)
    metrics = compute_metrics(k2, state)
    maybe_callback(metrics)

    def cond(carry):
        _, _, metrics = carry

        i = metrics.i
        improvement = metrics.improvement

        converged = improvement < h.vi_criterion
        diverged = improvement < 0.0
        naned = jnp.isnan(improvement) & (i > 0)

        stop = converged | diverged | naned
        keep_going = (~stop) & (i < max_iters)
        return keep_going

    def body(carry):
        key, state, metrics = carry

        new_key, key = jax.random.split(key)
        new_state = vi_step(state)
        new_metrics = compute_new_metrics(key, new_state, old=metrics)
        maybe_callback(new_metrics)

        return (new_key, new_state, new_metrics)

    _, final_state, final_metrics = lax.while_loop(
        cond, body, (key, state, metrics)
    )
    return final_state, final_metrics