# %%
# parameters, export
modality = "modal"
kernel = "pack:2"
normalized = True
effective_num_harmonics = 0.6
iteration = 1
seed = 4283955834

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gfm.ack import DiagonalTACK
from gp.blr import blr_from_mercer, log_probability
from pack import PACK

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
sigma_c = 1.0
t1 = 0.0
J = int(np.floor((fs / 2) / (1000 / period)) * effective_num_harmonics)
print(f"Using J={J} harmonics (max frequency={J * (1000 / period)} Hz)")


def ptform(u):
    z = ndtri(u[:3])
    t = period * u[-2:]  # uniform in [0, period]
    x = np.concatenate([10.0**z, t])
    return x


def build_theta(x):
    return {
        "sigma_noise": x[0],
        "sigma_a": x[1],
        "sigma_b": x[2],
        "center": x[4],
        "tc": x[3],
    }


def build_kernel(theta):
    tack = DiagonalTACK(
        d=d,
        normalized=normalized,
        center=theta["center"],
        sigma_b=theta["sigma_b"],
        sigma_c=sigma_c,
    )

    pack = PACK(
        tack,
        period=period,
        t1=t1,
        t2=theta["tc"],
        J=J,
    )

    return theta["sigma_a"] * pack


def build_gp(theta):
    pack = build_kernel(theta)
    return blr_from_mercer(pack, t, noise_variance=theta["sigma_noise"] ** 2)


u = np.random.uniform(size=5)
x = ptform(u)
theta = build_theta(x)
pack = build_kernel(theta)

# %%
# build generative model
# Compute constants (basis does not depend on parameters)
Phi = jax.vmap(pack.compute_phi)(t)

PhiT_Phi = jnp.matmul(Phi.T, Phi)
PhiT_y = jnp.matmul(Phi.T, du)


@jax.jit
def loglikelihood(x):
    theta = build_theta(x)
    pack = build_kernel(theta)

    # Bypass build_gp() to cache building Phi, PhiT_Phi, PhiT_y
    cov_root = pack.compute_weights_root()
    logl = log_probability(
        y=du,
        Phi=Phi,
        cov_root=cov_root,
        noise_variance=theta["sigma_noise"] ** 2,
        PhiT_Phi=PhiT_Phi,
        PhiT_y=PhiT_y,
        jitter=0.0,
    )

    return logl


loglikelihood(x)

# %%
# smoke test
rng = np.random.default_rng(seed)

x = ptform(rng.uniform(size=100))
theta = build_theta(x)
ndim = len(theta)

theta_noiseless = theta.copy()
theta_noiseless["sigma_noise"] = 1e-6

s = build_gp(theta_noiseless).sample(jax.random.PRNGKey(seed), shape=(3,))

plt.title(
    f"kernel: {kernel}, normalized: {normalized}, effective_num_harmonics: {effective_num_harmonics}"
)
plt.plot(t, s.T, label="sample from GP prior")
plt.legend()


# %%
# initialize our nested sampler
nlive = 500

sampler = NestedSampler(
    loglikelihood, ptform, ndim, nlive=nlive, rstate=rng, sample="rwalk"
)

with time_this() as elapsed:
    sampler.run_nested(maxcall=1_000_000, print_progress=False)


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

plt.title(
    f"kernel: {kernel}, normalized: {normalized}, effective_num_harmonics: {effective_num_harmonics}"
)
plt.legend()

# %%
from dynesty import plotting as dyplot

try:
    fig, ax = dyplot.cornerplot(
        res,
        labels=[str(k) for k in theta.keys()],
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
