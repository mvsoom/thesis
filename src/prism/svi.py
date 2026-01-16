# %%
import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
from flax import nnx, struct
from gpjax.dataset import Dataset
from gpjax.fit import Model
from gpjax.objectives import Objective
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)
from gpjax.scan import vscan
from gpjax.typing import (
    KeyArray,
    ScalarFloat,
)
from jax import lax
from numpyro.distributions.transforms import Transform
from tqdm import tqdm


@struct.dataclass
class Data:
    tau: jnp.ndarray
    du: jnp.ndarray
    mask: jnp.ndarray


def pad_waveforms(lf_samples, npad=None):
    if npad is None:
        npad = max(len(s["t"]) for s in lf_samples)

    I = len(lf_samples)
    tau_all = np.zeros((I, npad), dtype=np.float32)
    du_all = np.zeros((I, npad), dtype=np.float32)
    mask_all = np.zeros((I, npad), dtype=np.float32)

    for i, s in enumerate(tqdm(lf_samples, desc=f"Padding to width={npad}")):
        n = s["tau"].shape[0]
        tau_all[i, :n] = s["tau"]
        du_all[i, :n] = s["du"]
        mask_all[i, :n] = 1.0

    tau_all = jnp.array(tau_all)
    du_all = jnp.array(du_all)
    mask_all = jnp.array(mask_all)

    return Data(tau_all, du_all, mask_all)


def window_or_crop(key, datapoint: Data, width) -> Data:
    tau = datapoint.tau
    du = datapoint.du
    mask = datapoint.mask

    n = jnp.sum(mask).astype(jnp.int32)

    def window():
        max_start = n - width
        s = jax.random.randint(key, (), 0, max_start + 1)

        tau_w = lax.dynamic_slice(tau, (s,), (width,))
        du_w = lax.dynamic_slice(du, (s,), (width,))
        mask_w = jnp.ones((width,), jnp.float32)

        return Data(
            tau_w,
            du_w,
            mask_w,
        )

    def crop():
        return Data(
            tau[:width],
            du[:width],
            mask[:width],
        )

    return jax.lax.cond(n >= width, window, crop)


def sample_batch(key, dataset: Data, B, width):
    I = len(dataset.tau)
    key, subkey = jax.random.split(key)

    indices = jax.random.choice(key, I, shape=(B,))

    batch = Data(
        tau=dataset.tau[indices],
        du=dataset.du[indices],
        mask=dataset.mask[indices],
    )

    keys = jax.random.split(subkey, B)

    def f(key, datapoint):
        return window_or_crop(key, datapoint, width)

    return jax.vmap(f)(keys, batch)


def fit_batched_elbo(
    *,
    model: Model,
    objective: Objective,
    get_batch: callable,
    optim: ox.GradientTransformation,
    params_bijection: tp.Union[
        dict[Parameter, Transform], None
    ] = DEFAULT_BIJECTION,
    trainable: nnx.filterlib.Filter = Parameter,
    key: KeyArray = jr.PRNGKey(42),
    num_iters: int = 100,
    verbose: bool = True,
    unroll: int = 1,
) -> tuple[Model, jax.Array]:
    """Patch of gpjax.fit to support minibatches over waveforms rather than datapoints"""
    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    # Parameters bijection to unconstrained space
    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    # Loss definition
    def loss(params: nnx.State, batch: Dataset) -> ScalarFloat:
        params = transform(params, params_bijection)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, batch)

    # Initialise optimiser state.
    opt_state = optim.init(params)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        params, opt_state, index = carry

        batch = get_batch(key, index)

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)

        carry = (params, opt_state, index + 1)
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (params, _), history = scan(
        step, (params, opt_state, 0), (iter_keys), unroll=unroll
    )

    # Parameters bijection to constrained space
    if params_bijection is not None:
        params = transform(params, params_bijection)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, history
