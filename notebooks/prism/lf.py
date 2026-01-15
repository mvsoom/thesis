# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax as ox
from gpjax.parameters import Parameter
from tqdm import tqdm

from prism.blockindependent import gpxBlockIndependent
from prism.pack import NormalizedPACK
from surrogate import source
from utils import constants
from utils.jax import safe_cholesky, vk

# FIXME
jax.config.update("jax_debug_nans", True)

# %%
lf_samples = source.get_lf_samples()[:3]


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


for lf_sample in tqdm(lf_samples):
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
base = NormalizedPACK(d=1)
kernel = gpxBlockIndependent(base)

plt.imshow(kernel.gram(X[:500, :]).to_dense())

# %%
dataset = gpx.Dataset(X=X, y=jnp.reshape(du, (-1, 1)))

NUM_INDUCING = 32

z = jnp.linspace(tau.min(), tau.max(), NUM_INDUCING).reshape(-1, 1)

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

# %%

plt.imshow(kernel.gram(z).to_dense())

# %%
meanf = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
prior = gpx.gps.Prior(
    mean_function=meanf, kernel=kernel, jitter=constants.NOISE_FLOOR_POWER
)
p = prior * likelihood
q = gpx.variational_families.VariationalGaussian(posterior=p, inducing_inputs=z)

key = jax.random.PRNGKey(42)

batch_size = 128
num_iters = 3000

lr = 0.001

opt_posterior, history = gpx.fit(
    model=q,
    objective=lambda p, d: -gpx.objectives.elbo(p, d),
    train_data=dataset,
    optim=ox.adam(learning_rate=lr),
    num_iters=num_iters,
    key=key,
    batch_size=batch_size,
    trainable=Parameter,
)

# %%
plt.plot(history[0:])
# plt.xscale("log")

# %%
# Convert to the BLR implicit


m = opt_posterior.variational_mean.value.squeeze()
S = opt_posterior.variational_root_covariance.value

kernel = opt_posterior.posterior.prior.kernel
Z = opt_posterior.inducing_inputs.value

Kzz = kernel.gram(Z).to_dense()
Lzz = safe_cholesky(Kzz)
Kzz_inv = jax.scipy.linalg.cho_solve((Lzz, True), jnp.eye(Z.shape[0]))


def phi(t):
    x = jnp.asarray(t).reshape(1, 1)
    Kxz = kernel.cross_covariance(x, Z)  # (1, M), Dense operator
    # Kxz = Kxz.to_dense()
    return jax.scipy.linalg.cho_solve((Lzz, True), Kxz.T).squeeze()


t = jnp.linspace(0, 1, 500)

Phi = jax.vmap(phi)(t)  # should be (N, M)

mu = Phi @ m  # (N,)


fig, ax = plt.subplots()
ax.vlines(
    opt_posterior.inducing_inputs.value,
    ymin=du.min(),
    ymax=du.max(),
    alpha=0.3,
    linewidth=1,
    label="Inducing point",
)
ax.scatter(tau, du, alpha=0.2, label="Observations")
ax.plot(t, mu, ".", c="red", label="Posterior mean via feature map")
ax.legend()
ax.set(xlabel=r"$x$", ylabel=r"$f(x)$")

# %%
# Draw posterior samples
num_samples = 3
eps = jax.random.normal(vk(), shape=(num_samples, m.shape[0]))
m_samples = m + eps @ S

plt.plot(t, Phi @ m_samples.T)
plt.show()

# %%
# Examine basis functions
# Note: these model deviation from the mean, not the function values directly
basisfunctions = Phi @ S

plt.plot(t, basisfunctions[:, :50])
plt.show()
