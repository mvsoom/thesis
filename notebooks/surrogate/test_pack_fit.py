# %%
# parameters, export
kernel = "pack:2"
normalized = True
J = 256
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
from surrogate import source

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu") # GPU = 20x speedup


from dynesty import NestedSampler
from scipy.special import ndtri

from utils import constants, time_this

# %%
lf_samples = source.get_lf_samples(20)


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


samples = []
for lf_sample in lf_samples:
    du = lf_sample["u"]
    t_ms = lf_sample["t"]
    period_ms = lf_sample["p"]["T0"]
    samples.append(
        {
            "period_ms": period_ms,
            "t_ms": t_ms,
            "tau": warp_time(t_ms, period_ms),
            "du": du,
        }
    )

# %%
# resample all samples to a common tau grid
N_tau = int(max(sample["du"].shape[0] for sample in samples))
tau_grid = np.linspace(0.0, 1.0, N_tau)
tau = jnp.asarray(tau_grid)

du_tau = []
for sample in samples:
    du_tau.append(np.interp(tau_grid, sample["tau"], sample["du"]))

du_tau = np.stack(du_tau)

sample0 = samples[0]

fig, ax = plt.subplots(figsize=(10, 5))
plt.title("Data to fit (normalized time)")
plt.plot(tau_grid, du_tau[0], label="LF derivative")
plt.xlabel("Tau (unit period)")
plt.ylabel("Amplitude")
plt.legend()

# %%
d = int(kernel[-1])
sigma_c = 1.0
t1 = 0.0
t2 = 1.0
period_norm = 1.0
print(f"Using J={J} harmonics (max frequency={J} cycles/period)")


def ptform(u):
    z = ndtri(u[:3])
    t = period_norm * u[-1:]  # uniform in [0, 1]
    x = np.concatenate([10.0**z, t])
    return x


def build_theta(x):
    return {
        "sigma_noise": x[0],
        "sigma_a": x[1],
        "sigma_b": x[2],
        "center": x[3],
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
        period=period_norm,
        t1=t1,
        t2=t2,
        J=J,
    )

    return theta["sigma_a"] ** 2 * pack


def build_gp(theta):
    pack = build_kernel(theta)
    return blr_from_mercer(pack, tau, noise_variance=theta["sigma_noise"] ** 2)


u = np.random.uniform(size=5)
x = ptform(u)
theta = build_theta(x)
pack = build_kernel(theta)

# %%
# build generative model
# Compute constants (basis does not depend on parameters)
Phi = jax.vmap(pack.compute_phi)(tau)

PhiT_Phi = jnp.matmul(Phi.T, Phi)
du_stack = jnp.asarray(du_tau)
PhiT_y_all = jnp.matmul(du_stack, Phi)


@jax.jit
def loglikelihood(x):
    theta = build_theta(x)
    pack = build_kernel(theta)

    # Bypass build_gp() to cache building Phi, PhiT_Phi, PhiT_y_all
    cov_root = pack.compute_weights_root()
    logl = jax.vmap(
        lambda y, PhiT_y: log_probability(
            y=y,
            Phi=Phi,
            cov_root=cov_root,
            noise_variance=theta["sigma_noise"] ** 2,
            PhiT_Phi=PhiT_Phi,
            PhiT_y=PhiT_y,
            jitter=constants.NOISE_FLOOR_POWER,
        )
    )(du_stack, PhiT_y_all)

    return jnp.sum(logl)


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

plt.title(f"kernel: {kernel}, normalized: {normalized}, J: {J}")
plt.plot(tau_grid, s.T, label="sample from GP prior")
plt.legend()

# %%
# initialize our nested sampler
nlive = 16

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

# %%
du_test = du_stack[5]

for x in xs[:5]:
    theta = build_theta(x)

    print(theta)

    gp = build_gp(theta)
    mu, var = gp.predict(du_test, tau, return_var=True)
    std = jnp.sqrt(var) + theta["sigma_noise"]

    plt.fill_between(
        tau_grid,
        mu - 1.96 * std,
        mu + 1.96 * std,
        alpha=0.2,
    )
    plt.plot(tau_grid, mu, label="GP posterior mean")

plt.plot(tau_grid, du_test, label="data")

plt.title(f"kernel: {kernel}, normalized: {normalized}, J: {J}")
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
te = tau_grid[np.argmin(du_tau[0])]
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

# %%
# This:
# > logz = np.float64(2665.2054484272)
# add sigma_c
# > logz = 2662.143 +/-  1.008
# add t_c
# > logz = 5137.218 +/-  1.206 => 8 min (nlive=20)
# > effectively `center` ~ 0.5 < `t_c` ~ 0.9!
# add sigma_c AND t_c => 10 min (nlive=24)
# > 5140.146 +/-  1.015

# setting sigma_a = 1 and inferring sigma_c did not work
# but we can calculate


with sigma_c and t_c and d = 2
logz: 5137.365 +/-  1.327

d = 1 # running
logz: 7740.116 +/-  1.261
Walltime: 1106.941 s

d = 0
logz: 8054.234 +/-  1.223
Walltime: 4401.049 s
