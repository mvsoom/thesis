# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
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
lf_samples = source.get_lf_samples()[:100]


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
print("Number of datapoints:", dataset.n)

NUM_INDUCING = 64

# %%
meanf = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
prior = gpx.gps.Prior(
    mean_function=meanf, kernel=kernel, jitter=constants.NOISE_FLOOR_POWER
)
p = prior * likelihood

num_iters = 500
num_restarts = 30
batch_size = -1

# PROBLEM: cant minibatch with this => very noisy signal
# SO WE LIMIT TO FULL BATCH FOR NOW WITH *LESS* SAMPLES
# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891-prism/c/6968c34f-250c-8325-8472-43c98510b7cf
# we can still impose closure constraint?

lr = 0.001

key = vk()
keys = jax.random.split(key, num_restarts)


def optimize(key):
    key, subkey = jax.random.split(key)

    z = jax.random.uniform(
        subkey, shape=(NUM_INDUCING, 1), minval=0.0, maxval=1.0
    )

    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p, inducing_inputs=z
    )

    opt_posterior, history = gpx.fit(
        model=q,
        objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
        train_data=dataset,
        optim=ox.adam(learning_rate=lr),
        num_iters=num_iters,
        key=key,
        batch_size=batch_size,
        trainable=Parameter,
        log_rate=100,
    )

    return nnx.state(opt_posterior), history


states, histories = jax.vmap(optimize)(keys)

# %%
# pick one posterior (e.g. best idx) and rebuild

i = int(jnp.nanargmin(jnp.array([h[-1] for h in histories])))  # best elbo

graphdef, _ = nnx.split(
    gpx.variational_families.CollapsedVariationalGaussian(
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
Xtest = X[::100, :]

latent_dist = opt_posterior(Xtest, train_data=dataset)
predictive_dist = opt_posterior.posterior.likelihood(latent_dist)

inducing_points = opt_posterior.inducing_inputs.value

samples = latent_dist.sample(key=vk(), sample_shape=(20,))

predictive_mean = predictive_dist.mean
predictive_std = jnp.sqrt(predictive_dist.variance)  # all nans

fig, ax = plt.subplots()

ax.plot(Xtest[:, 1], predictive_mean, ".", label="Predictive mean")

ax.fill_between(
    Xtest[:, 1],
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    alpha=0.2,
    label="Two sigma",
)


ax.vlines(
    x=inducing_points,
    ymin=-1,
    ymax=1,
    alpha=0.3,
    linewidth=0.5,
    label="Inducing point",
)
ax.legend()
ax.set(xlabel=r"$x$", ylabel=r"$f(x)$")
plt.show()

# %%
Z = inducing_points
Kmm = kernel.gram(Z).to_dense()
Lzz = safe_cholesky(Kmm)  # Kmm = Lzz @ Lzz.T


def psi(t):
    x = jnp.asarray(t).reshape(1, -1)
    Kxz = kernel.cross_covariance(x, Z)
    return jax.scipy.linalg.solve_triangular(
        Lzz, Kxz.T, lower=True
    ).squeeze()  # psi = Lzz^{-1} Kzx


t = jnp.linspace(0, 1, 500)
Psi = jax.vmap(psi)(t)

eps = jax.random.normal(vk(), shape=(NUM_INDUCING,))
y = Psi @ eps
plt.plot(t, y)
plt.title("Posterior samples of latent function")
plt.xlabel("tau")
