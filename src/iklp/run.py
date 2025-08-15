import jax
import jax.numpy as jnp

from iklp.mercer_op import build_data
from iklp.metrics import compute_metrics
from iklp.state import (
    VIState,
    init_variational_params,
)
from iklp.vi import vi_step


def vi_run(key, data):
    """Run a single VI run on the given data, where each state is sequentially updated"""
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
    """Run VI on a single frame"""
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
