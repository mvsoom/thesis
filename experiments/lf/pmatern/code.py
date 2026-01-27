# %%
# parameters, export
nu = 100
M = 16
seed = 4283955834
sample_idx = 13

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tinygp import GaussianProcess

from gp.periodic import PeriodicSE
from prism.pmatern import PeriodicMatern
from utils.jax import vk

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu") # GPU = 20x speedup


from dynesty import NestedSampler
from scipy.special import ndtri

from surrogate import source
from utils import time_this

# %%
lf_sample = source.get_lf_samples()[sample_idx]


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


lf_sample["tau"] = warp_time(lf_sample["t"], lf_sample["p"]["T0"])


# %%
def ptform(u):
    z = ndtri(u)
    return 10.0**z


def build_theta(x):
    return {
        "sigma_noise": x[0],
        "sigma_a": x[1],
        "sigma_c": x[2],
    }


def build_kernel(theta):
    if nu < 100:
        k = PeriodicMatern(
            nu=nu,
            scale=theta["sigma_c"],
            M=M,
        )
    else:
        k = PeriodicSE(ell=theta["sigma_c"], period=1.0, J=M // 2)

    return theta["sigma_a"] ** 2 * k


def build_gp(theta):
    k = build_kernel(theta)
    gp = GaussianProcess(
        kernel=k, X=lf_sample["tau"], diag=theta["sigma_noise"] ** 2
    )
    return gp


u = np.random.uniform(size=100)
x = ptform(u)
theta = build_theta(x)
k = build_kernel(theta)


# %%
@jax.jit
def loglikelihood(x):
    theta = build_theta(x)
    gp = build_gp(theta)
    return gp.log_probability(lf_sample["u"])


loglikelihood(x)

# %%
# smoke test
rng = np.random.default_rng(seed)

x = ptform(rng.uniform(size=100))
theta = build_theta(x)
ndim = sum(v.size for v in theta.values())

s = build_gp(theta).sample(vk(), shape=(3,))

plt.title(f"nu: {nu}, M: {M}, sample_idx: {sample_idx}")
plt.plot(lf_sample["tau"], lf_sample["u"], c="black", label="data")
plt.plot(lf_sample["tau"], s.T, label="sample from GP prior")
plt.xlabel("tau (normalized time)")
plt.legend()


# %%
# initialize our nested sampler
nlive = 256

sampler = NestedSampler(
    loglikelihood, ptform, ndim, nlive=nlive, rstate=rng, sample="rwalk"
)

with time_this() as elapsed:
    sampler.run_nested(maxcall=1_000_000, print_progress=True)


# %%
res = sampler.results

res.summary()


# %%
xs = res.samples_equal(rng)[:100]

means = {}
stds = {}

for x in xs:
    theta = build_theta(x)
    for k, v in theta.items():
        if "sigma" in k:
            v = np.log10(v)
            k = f"{k}_log10"
        means[k] = means.get(k, 0.0) + v
        stds[k] = stds.get(k, 0.0) + v * v

N = float(len(xs))

means = {k: v / N for k, v in means.items()}
stds = {k: np.sqrt(stds[k] / N - means[k] * means[k]) for k in means}

print(means)
print(stds)

for x in xs[:5]:
    theta = build_theta(x)

    print(theta)

    gp = build_gp(theta)
    mu, var = gp.predict(lf_sample["u"], lf_sample["tau"], return_var=True)
    std = jnp.sqrt(var) + theta["sigma_noise"]

    plt.fill_between(
        lf_sample["tau"],
        mu - 1.96 * std,
        mu + 1.96 * std,
        alpha=0.2,
    )
    plt.plot(lf_sample["tau"], mu, label="GP posterior mean")

plt.plot(lf_sample["tau"], lf_sample["u"], label="data")

plt.title(f"nu: {nu}, M: {M}")
plt.legend()

# %%
from dynesty import plotting as dyplot

labels = [str(k) for k in theta.keys()]

try:
    fig, ax = dyplot.cornerplot(
        res,
        labels=labels,
        verbose=True,
        quantiles=[0.05, 0.5, 0.95],
    )
except Exception as e:
    print(f"Could not make corner plot: {e}")


# %%
# export
mean = means
std = stds

logz = res.logz[-1]
logzerr = res.logzerr[-1]

log_prob_u = lf_sample["log_prob_u"]

ndim = res.samples.shape[1]
information = res.information[-1]

niter = res.niter
ncall = res.ncall.sum()
walltime = elapsed.walltime
