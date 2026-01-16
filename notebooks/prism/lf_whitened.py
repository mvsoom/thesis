# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax as ox
from flax import nnx
from gpjax.parameters import Parameter
from tqdm import tqdm

from prism.blockindependent import gpxBlockIndependent
from prism.pack import NormalizedPACK
from surrogate import source
from utils import constants
from utils.jax import safe_cholesky, vk

# %%
lf_samples = source.get_lf_samples()[:500]


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
# from prism.pmatern import gpxPeriodicMatern
# base = gpxPeriodicMatern(nu=3/2)
kernel = gpxBlockIndependent(base)

plt.imshow(kernel.gram(X[:500, :]).to_dense())

# %%
dataset = gpx.Dataset(X=X, y=jnp.reshape(du, (-1, 1)))

NUM_INDUCING = 16

# %%
meanf = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
prior = gpx.gps.Prior(
    mean_function=meanf, kernel=kernel, jitter=constants.NOISE_FLOOR_POWER
)
p = prior * likelihood

batch_size = 128
num_iters = 3000
num_restarts = 100

lr = 0.01

key = vk()
keys = jax.random.split(key, num_restarts)


def optimize(key):
    key, subkey = jax.random.split(key)

    z = jax.random.uniform(
        subkey, shape=(NUM_INDUCING, 1), minval=0.0, maxval=1.0
    )

    q = gpx.variational_families.WhitenedVariationalGaussian(
        posterior=p, inducing_inputs=z
    )

    opt_posterior, history = gpx.fit(
        model=q,
        objective=lambda p, d: -gpx.objectives.elbo(p, d),
        train_data=dataset,
        optim=ox.adam(learning_rate=lr),
        num_iters=num_iters,
        key=key,
        batch_size=batch_size,
        trainable=Parameter,
        log_rate=1000,
    )

    return nnx.state(opt_posterior), history


states, histories = jax.vmap(optimize)(keys)

# %%
# pick one posterior (e.g. best idx) and rebuild

i = int(jnp.nanargmin(jnp.array([h[-1] for h in histories])))  # best elbo

graphdef, _ = nnx.split(
    gpx.variational_families.VariationalGaussian(
        posterior=p,
        inducing_inputs=jnp.zeros((NUM_INDUCING, 1)),
    )
)

opt_posterior = nnx.merge(graphdef, jax.tree.map(lambda x: x[i], states))
history = histories[i]

# %%
plt.plot(history[0:])
# plt.xscale("log")

print(
    "Observation sigma_noise:",
    opt_posterior.posterior.likelihood.obs_stddev.value,
)

# %%
# Convert to the BLR implicit

mu_w = opt_posterior.variational_mean.value.squeeze()  # (M,)
Sroot_w = opt_posterior.variational_root_covariance.value  # (M,M) lower-tri
kernel = opt_posterior.posterior.prior.kernel
mean_fn = opt_posterior.posterior.prior.mean_function
Z = opt_posterior.inducing_inputs.value  # (M,D)

Kzz = kernel.gram(Z).to_dense()
Lzz = safe_cholesky(Kzz)  # (M,M) lower

mZ = mean_fn(Z).squeeze()  # (M,)

# antiwhiten: q(u) in inducing space
m_u = mZ + Lzz @ mu_w  # (M,)
S_u_root = Lzz @ Sroot_w  # (M,M)
# so S_u = S_u_root @ S_u_root.T

Kzz_inv = jsp.linalg.cho_solve((Lzz, True), jnp.eye(Z.shape[0]))


def phi(t):
    x = jnp.asarray(t).reshape(1, -1)  # (1,D)
    Kxz = kernel.cross_covariance(
        x, Z
    )  # (1,M) in GPJax ordering? you used x,Z; keep consistent
    return jsp.linalg.cho_solve((Lzz, True), Kxz.T).squeeze()  # (M,)


t = jnp.linspace(0, 1, 500)
Phi = jax.vmap(phi)(t)  # (N,M)

mu = Phi @ m_u
std = jnp.sqrt(jnp.sum((Phi @ S_u_root) ** 2, axis=1))

fig, ax = plt.subplots()
ax.vlines(
    opt_posterior.inducing_inputs.value,
    ymin=du.min(),
    ymax=du.max(),
    alpha=0.3,
    linewidth=1,
    label="Inducing point",
)
ax.scatter(tau, du, alpha=0.01, label="Observations")
ax.plot(t, mu, ".", c="red", label="Posterior mean")
ax.fill_between(
    t,
    mu - 3 * std,
    mu + 3 * std,
    color="red",
    alpha=0.2,
    label="Latent 3 sigma",
)
ax.legend()
ax.set(xlabel=r"$x$", ylabel=r"$f(x)$")

# %%
# Draw posterior samples
num_samples = 4
eps = jax.random.normal(vk(), shape=(m_u.shape[0], num_samples))
u_samples = m_u[:, None] + S_u_root @ eps

fig, ax = plt.subplots()
ax.plot(t, mu, ".", c="red", label="Posterior mean")
ax.fill_between(
    t,
    mu - 3 * std,
    mu + 3 * std,
    color="red",
    alpha=0.2,
    label="Latent 3 sigma",
)

ax.plot(t, Phi @ u_samples, alpha=0.7)
fig.show()

# %%
# Examine basis functions
# Note: these model deviation from the mean, not the function values directly
basisfunctions = Phi @ S_u_root

plt.plot(t, basisfunctions[:, :50])
plt.show()

# %%
