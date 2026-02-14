# %%
# parameters, export
modality = "modal"
kernel = "pack:1"
normalized = False
single_sigma_c = True
J = 5
iteration = 1
seed = 4283955834

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tinygp import GaussianProcess

from prism.pack import PACK
from utils.jax import vk

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu") # GPU = 20x speedup


from dynesty import NestedSampler
from scipy.special import ndtri

from gfm.lf import lf_modality_waveforms
from utils import time_this

# %%
fs = 20_000  # Hz; from Chien+ 2017

period = 7.0  # msec
N = int(fs * period / 1000)  # samples per period

lfs = lf_modality_waveforms(
    period_ms=period, samples_per_period=N, normalize_power=True
)
d = lfs[modality]

t, du, u = d["t"], d["du"], d["u"]
tc = d["timings"]["Te"] * 1000

fig, ax = plt.subplots(figsize=(10, 5))
plt.title(f"Data to fit: (modality: {modality})")
plt.plot(t, u, label="LF waveform")
plt.plot(t, du, label="LF derivative")
plt.axvline(tc, color="gray", linestyle="--", label="tc")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()

# %%
d = int(kernel[-1])

def ptform(u):
    z = ndtri(u)
    return 10.0**z


def build_theta(x):
    return {
        "sigma_noise": x[0],
        "sigma_a": x[1],
        "sigma_b": x[2],
        "sigma_c": x[3] if single_sigma_c else x[3 : 3 + J],
    }


def build_kernel(theta):
    pack = PACK(
        d=d,
        normalized=normalized,
        J=J,
        period=period,
        sigma_b=theta["sigma_b"],
        sigma_c=theta["sigma_c"],
    )

    return theta["sigma_a"] ** 2 * pack


def build_gp(theta):
    pack = build_kernel(theta)
    gp = GaussianProcess(kernel=pack, X=t, diag=theta["sigma_noise"] ** 2)
    return gp


u = np.random.uniform(size=100)
x = ptform(u)
theta = build_theta(x)
pack = build_kernel(theta)

# %%
@jax.jit
def loglikelihood(x):
    theta = build_theta(x)
    gp = build_gp(theta)
    return gp.log_probability(du)


loglikelihood(x)

# %%
# smoke test
rng = np.random.default_rng(seed)

x = ptform(rng.uniform(size=100))
theta = build_theta(x)
ndim = sum(v.size for v in theta.values())

s = build_gp(theta).sample(vk(), shape=(3,))

plt.title(f"kernel: {kernel}, normalized: {normalized}, J: {J}")
plt.plot(t, s.T, label="sample from GP prior")
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
    mu, var = gp.predict(du, t, return_var=True)
    std = jnp.sqrt(var) + theta["sigma_noise"]

    plt.fill_between(
        t,
        mu - 1.96 * std,
        mu + 1.96 * std,
        alpha=0.2,
    )
    plt.plot(t, mu, label="GP posterior mean")

plt.plot(t, du, label="data")

plt.title(f"kernel: {kernel}, normalized: {normalized}, J: {J}")
plt.legend()

# %%
from dynesty import plotting as dyplot

labels = [
    str(k) for k in theta.keys() for _ in range(J if k == "sigma_c" else 1)
]

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
te = t[np.argmin(du)]
mean = means
std = stds

# %%
# export
logz = res.logz[-1]
logzerr = res.logzerr[-1]

ndim = res.samples.shape[1]
information = res.information[-1]

niter = res.niter
ncall = res.ncall.sum()
walltime = elapsed.walltime
