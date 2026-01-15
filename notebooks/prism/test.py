# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax as ox
from gpjax.parameters import Parameter

from prism.blockindependent import gpxBlockIndependent
from prism.pack import NormalizedPACK
from surrogate import source
from utils import constants

# %%
lf_samples = source.get_lf_samples()[:5]


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


for lf_sample in lf_samples:
    lf_sample["tau"] = warp_time(lf_sample["t"], lf_sample["p"]["T0"])

# %%
# Make training data
du = jnp.concatenate([lf_sample["u"] for lf_sample in lf_samples])
tau = jnp.concatenate([lf_sample["tau"] for lf_sample in lf_samples])
idx = jnp.concatenate(
    [
        jnp.full(lf_sample["u"].shape, i)
        for i, lf_sample in enumerate(lf_samples)
    ]
)
X = jnp.vstack([idx, tau]).T

# %%

# jax debug nans
jax.config.update("jax_debug_nans", True)

# k = gpxPeriodicMatern(nu=1.5)  # WORKS
k = NormalizedPACK(d=1)
kb = gpxBlockIndependent(k)

# plt.imshow(kb.gram(X[:1000, :]).to_dense())

dataset = gpx.Dataset(X=X, y=jnp.reshape(du, (-1, 1)))

NUM_INDUCING = 16

z = jnp.linspace(tau.min(), tau.max(), NUM_INDUCING).reshape(-1, 1)

"""
fig, ax = plt.subplots()
ax.vlines(
    z,
    ymin=du.min(),
    ymax=du.max(),
    alpha=0.3,
    linewidth=1,
    label="Inducing point",
)
ax.scatter(tau, du, alpha=0.2, label="Observations")
ax.legend()
ax.set(xlabel=r"$x$", ylabel=r"$f(x)$")
"""

# plt.imshow(kb.gram(z).to_dense())


meanf = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
kernel = kb
prior = gpx.gps.Prior(
    mean_function=meanf, kernel=kernel, jitter=constants.NOISE_FLOOR_POWER
)
p = prior * likelihood
q = gpx.variational_families.VariationalGaussian(posterior=p, inducing_inputs=z)

# gpx.objectives.elbo(q, dataset)

schedule = ox.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=0.02,
    warmup_steps=75,
    decay_steps=2000,
    end_value=0.001,
)

key = jax.random.PRNGKey(42)

batch_size = 256
num_iters = 1000

opt_posterior, history = gpx.fit(
    model=q,
    objective=lambda p, d: -gpx.objectives.elbo(p, d),
    train_data=dataset,
    optim=ox.adam(learning_rate=schedule),
    num_iters=num_iters,
    key=key,
    batch_size=batch_size,
    trainable=Parameter,
)

print(history)

# %%
# Convert to the BLR implicit
m = opt_posterior.variational_mean.value.squeeze()
S = opt_posterior.variational_root_covariance.value

kernel = opt_posterior.posterior.prior.kernel
Z = opt_posterior.inducing_inputs.value

Kzz = kernel.gram(Z).to_dense()
Lzz = jnp.linalg.cholesky(Kzz)
Kzz_inv = jax.scipy.linalg.cho_solve((Lzz, True), jnp.eye(Z.shape[0]))


def phi(t):
    x = jnp.asarray(t).reshape(1, 1)
    Kxz = kernel.cross_covariance(x, Z)  # (1, M), Dense operator
    # Kxz = Kxz.to_dense()
    return jax.scipy.linalg.cho_solve((Lzz, True), Kxz.T).squeeze()


t = jnp.linspace(0, 3, 500)

Phi = jax.vmap(phi)(t)  # should be (N, M)

mu = Phi @ m  # (N,)

plt.plot(t, mu, ".", label="Posterior mean via feature map")
plt.scatter(tau, du, alpha=0.15, label="Training Data")
plt.legend()

# %%
# Draw posterior samples
from utils.jax import vk

num_samples = 1
eps = jax.random.normal(vk(), shape=(num_samples, m.shape[0]))
m_samples = m + eps @ S

plt.plot(t, Phi @ m_samples.T)

# %%
# %%
# STILL GET NANS AFTER A WHILE
# CAN FIND OUT WHERE HERE =>

# jax debug nans
jax.config.update("jax_debug_nans", True)

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.statelib import State


def is_bad_array(x):
    return isinstance(x, jnp.ndarray) and (
        jnp.any(jnp.isnan(x)) | jnp.any(jnp.isinf(x))
    )


def report_bad(tree, prefix=""):
    # NNX State: recurse into .items()
    if isinstance(tree, State):
        for k, v in tree.items():
            report_bad(v, f"{prefix}.{k}" if prefix else k)
        return

    # dict-like
    if isinstance(tree, dict):
        for k, v in tree.items():
            report_bad(v, f"{prefix}.{k}" if prefix else str(k))
        return

    # tuple / list
    if isinstance(tree, (tuple, list)):
        for i, v in enumerate(tree):
            report_bad(v, f"{prefix}[{i}]")
        return

    # array leaf
    if isinstance(tree, jnp.ndarray):
        if bool(jnp.any(jnp.isnan(tree)) | jnp.any(jnp.isinf(tree))):
            print("BAD GRAD:", prefix, "shape", tree.shape)
        return

    # everything else (scalars, None, objects): ignore


import jax
import jax.numpy as jnp
import optax as ox
from gpjax.fit import get_batch
from gpjax.parameters import Parameter

# --- setup ---
optim = ox.adam(learning_rate=schedule)
key = jax.random.PRNGKey(42)

graphdef, params, rest = nnx.split(q, Parameter, ...)
opt_state = optim.init(params)


def loss_from_params(params, batch):
    mdl = nnx.merge(graphdef, params, rest)
    return -gpx.objectives.elbo(mdl, batch)


@jax.jit
def step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_from_params)(params, batch)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = ox.apply_updates(params, updates)
    return params, opt_state, loss, grads


iter_keys = jax.random.split(key, num_iters)

for i, subkey in zip(range(num_iters), iter_keys):
    # minibatch exactly like gpx.fit
    batch = get_batch(dataset, batch_size, subkey)

    params, opt_state, loss, grads = step(params, opt_state, batch)

    if not jnp.isfinite(loss):
        print(f"\nLOSS NAN at step {i}")
        report_bad(grads)
        break

    if i % 50 == 0:
        print(f"step {i:4d} | loss {loss:.3f}")
        sigmas = params["posterior"]["prior"]["kernel"]["base_kernel"]
        print(
            f"sigma_a = {sigmas['sigma_a'].value.item():.3f}, sigma_b = {sigmas['sigma_b'].value.item():.3f}, sigma_c = {sigmas['sigma_c'].value.item():.3f}"
        )
