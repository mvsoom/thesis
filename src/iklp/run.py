from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from iklp.mercer_op import build_data
from iklp.metrics import compute_metrics
from iklp.state import (
    VIState,
    init_state,
    init_variational_params,
)
from iklp.vi import compute_elbo_bound, vi_step


def vi_run(key, data):
    """Do a single VI optimization on the given data, where each state is sequentially updated for a fixed number of iterations"""
    xi = init_variational_params(key, data.h)
    state = VIState(data, xi)

    def body(state, _):
        metrics_item = compute_metrics(state)
        state = vi_step(state)
        return state, metrics_item

    num_vi_iters = data.h.num_vi_iters

    final_state, metrics = jax.lax.scan(body, state, length=num_vi_iters)
    final_metrics = compute_metrics(final_state)

    # Append final metrics to get shape [n_iter + 1, ...]
    metrics = jax.tree_util.tree_map(
        lambda m, fm: jnp.concatenate([m, jnp.expand_dims(fm, 0)], axis=0),
        metrics,
        final_metrics,
    )

    return metrics  # leaves now have leading time axis T = n_iter + 1


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
          *Note*: this function jit-specializes on the shape of `h.Phi`, `h.P`, `h.num_vi_restarts`, and `h.num_vi_iters`.
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
    )  # Specializes on batch size and h.Phi, h.P, h.num_vi_restarts, h.num_vi_iters

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


@partial(jax.jit, static_argnames=("print_progress",))
def vi_run_criterion(key, x, h, max_iters=jnp.inf, print_progress=False):
    """Run a single VI run with given data `x` and hyperparameters `h`, where each state is sequentially updated, until the relative improvement in ELBO is (a) below `criterion`, (b) diverges (becomes nan or 0^-)

    *Note*: JAX can't compile and carry state in a while loop, so we can only return the last state. The number of iterations and final ELBO value are also returned.
    """
    criterion = h.vi_criterion
    state0 = init_state(key, x, h)

    carry0 = (
        jnp.array(0, jnp.int32),  # i
        state0,  # state
        -jnp.inf,  # last elbo
        jnp.array(True),  # keep_going
    )

    def cond(carry):
        i, _, _, keep_going = carry
        return (i < max_iters) & keep_going

    def body(carry):
        i, state, last_elbo, _ = carry
        new_state = vi_step(state)
        elbo = compute_elbo_bound(new_state)
        improvement = jnp.where(
            i == 0, jnp.inf, (elbo - last_elbo) / jnp.abs(last_elbo)
        )

        if print_progress:
            jax.debug.print(
                "iter {iter}/{max_iters}: elbo = {ELBO}, improvement = {improvement}",
                iter=i + 1,
                max_iters=max_iters,
                ELBO=elbo,
                improvement=improvement,
            )

        converged = improvement < criterion
        diverged = improvement < 0.0
        naned = jnp.isnan(improvement)
        keep_going = ~(converged | diverged | naned)

        return (i + 1, new_state, elbo, keep_going)

    final_num_iters, final_state, final_elbo, _ = lax.while_loop(
        cond, body, carry0
    )
    return final_state, final_num_iters, final_elbo