from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from tqdm import tqdm

from iklp.metrics import StateMetrics, compute_metrics, compute_new_metrics
from iklp.state import (
    VIState,
    init_state,
)
from iklp.vi import vi_step


def print_progress(metrics: StateMetrics):
    print(
        f"iter {metrics.i}: elbo = {metrics.elbo:.2f} ({metrics.improvement:+.8f} improvement)"
    )

@partial(jax.jit, static_argnames=("callback",))
def vi_run_criterion(
    key,
    x,
    h,
    max_iters=10_000,
    callback=None,
) -> tuple[VIState, StateMetrics]:
    """Run a single VI run with given data `x` and hyperparameters `h`, where each state is sequentially updated, until the relative improvement in ELBO is (a) below `criterion`, (b) diverges (becomes nan or 0^-) or (c) `max_iters` is reached.

    `callback` gets called after each state update with the current `StateMetrics`. See `print_progress()` for an example callback function. Use sparingly, because this always triggers an expensive COPY transfer from GPU to host of the metrics of the current state -- NOT the entire state including Phi, the data, etc. If `None`, this gets compiled out as expected.

    Note: this function works with vmap(); it stalls the other lanes until the last one is done, and returns the final state with a batch dimension indexing the lane. Even the callback works with vmap() -- it is called sequentially per lane for each iteration.
    """

    def maybe_callback(metrics):
        if callback:
            jax.debug.callback(callback, metrics, ordered=True)

    key, k0, k1, k2 = jax.random.split(key, 4)
    h = h.replace(krylov=h.krylov.replace(key=k0))
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


def vi_run_criterion_batched(
    key, x, h, batch_size=None, callback=None, verbose=False
):
    """Batch version of `vi_run_criterion()` without restarts

    **Note**: maximum number of iterations is given by `h.num_vi_iters` (not `max_iters`) and criterion is `h.vi_criterion`.

    Args:
        key: jax.random.PRNGKey
        x: jnp.ndarray shaped (num_frames, M) where M is number of samples per frame
        h: Hyperparams
          *Note*: this function jit-specializes on `h.arprior`, `h.num_vi_restarts`, and `h.num_vi_iters`.
        batch_size: int or None. If None, process all frames in one batch.

    Returns:
        metrics: pytree with leaves shaped (num_frames, ...)
    """
    n = x.shape[0]
    batch_size = batch_size or n

    n_batches = (n + batch_size - 1) // batch_size
    keys = jax.random.split(key, n_batches)

    def run_one(key, x, h):
        final_state, final_metrics = vi_run_criterion(
            key, x, h, max_iters=h.num_vi_iters, callback=callback
        )
        del final_state
        return final_metrics

    def run_batch(key, x, h):
        keys = jax.random.split(key, x.shape[0])
        return jax.vmap(run_one, in_axes=(0, 0, None))(keys, x, h)

    run_batch = jax.jit(run_batch)

    host_chunks = []
    for i in tqdm(range(n_batches), disable=not verbose):
        s = i * batch_size
        e = min(s + batch_size, n)
        dev_chunk = run_batch(keys[i], x[s:e], h)
        host_chunk = jax.device_get(dev_chunk)  # sync + copy to host
        del dev_chunk
        host_chunks.append(host_chunk)

    # Concatenate along the frames axis (axis=0) for every leaf.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        metrics = jax.tree_util.tree_map(
            lambda *parts: jnp.concatenate(parts, axis=0), *host_chunks
        )

    def unpack(tree):
        with jax.default_device(cpu):
            for i in range(x.shape[0]):
                yield jax.tree_util.tree_map(lambda a: a[i], tree)

    return metrics, unpack  # (len(frames), ...)
