"""
Kernel died caused by segfaults:

... 23715it [06:33, 61.08it/s, bound: 48 | nc: 24 | ncall: 544763 | eff(%): 4.353 | loglstar: -inf < 232.221 < inf | logz: 185.21223722it [06:33, 60.14it/s, bound: 48 | nc: 24 | ncall: 544931 | eff(%): 4.353 | loglstar: -inf < 232.221 < inf | logz: 185.22223729it [06:33, 60.62it/s, bound: 48 | nc: 24 | ncall: 545099 | eff(%): 4.353 | loglstar: -inf < 232.222 < inf | logz: 185.231 +/- 0.301 | dlogz: 0.534 > 0.509]Segmentation fault (core dumped)

Both dynesty 2.1.5 and 3.0.0 exhibit this behavior.

Happens both on CPU and GPU with jit compiled.

Culprit may be jax => jitting => leave for now

"""

# %%
# parameters, export
Rd = 0.3
centered = True
iteration = 1
kernel = "tack:2"
normalized = False
seed = 4283955834

# %%
# Parameters
Rd = 1.5
centered = False
iteration = 1
kernel = "tack:3"
normalized = True
seed = 4170760289


# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
jax.config.update("jax_platform_name", "cpu")

from dynesty import NestedSampler

from gfm.kernel import build_theta, instantiate_kernel
from gfm.lf import lf_relaxation_open_phase
from utils import time_this

# %%
tc = 6.0
N = 256

d = lf_relaxation_open_phase(Rd, tc, N)
t, du, u = d["t"], d["du"], d["u"]

plt.title(f"Data to fit: Rd={Rd}")
plt.plot(t, du, label="du")
plt.plot(t, u, label="u")
plt.legend()


# %%
# build generative model
from scipy.special import ndtri
from tinygp.gp import GaussianProcess

hyper = {
    "T": tc,
    "normalized": normalized,
    "centered": centered,
    "center": t.mean(),
}


def build_gp(theta):
    k = instantiate_kernel(kernel, theta, hyper)
    gp = GaussianProcess(kernel=k, X=t, diag=theta["sigma_noise"] ** 2)
    return gp


@jax.jit
def loglikelihood(x):
    theta = build_theta(x, kernel)
    gp = build_gp(theta)
    return gp.log_probability(du)


def ptform(u):
    z = ndtri(u)
    return 10.0**z


# %%
# smoke test
rng = np.random.default_rng(seed)

x = ptform(rng.uniform(size=100))
theta = build_theta(x, kernel)
ndim = len(theta)

theta_noiseless = theta.copy()
theta_noiseless["sigma_noise"] = 1e-6

s = build_gp(theta_noiseless).sample(jax.random.PRNGKey(seed), shape=(3,))

plt.title(f"kernel: {kernel}, centered: {centered}, normalized: {normalized}")
plt.plot(t, s.T, label="sample from GP prior")
plt.legend()

loglikelihood(x)

# %%
# initialize our nested sampler
nlive = 500

sampler = NestedSampler(
    loglikelihood, ptform, ndim, nlive=nlive, rstate=rng, sample="rwalk"
)

with time_this() as elapsed:
    sampler.run_nested(maxcall=1_000_000, print_progress=True)


# %%
res = sampler.results

res.summary()


# %%
xs = res.samples_equal()[:5]

for x in xs:
    theta = build_theta(x, kernel)

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

plt.title(f"kernel: {kernel}, centered: {centered}, normalized: {normalized}")
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
logz = res.logz[-1]
logzerr = res.logzerr[-1]

ndim = res.samples.shape[1]
information = res.information[-1]

niter = res.niter
ncall = res.ncall.sum()
walltime = elapsed.walltime

# %%
import numpy as np
import scrapbook as sb


def _to_py(x):
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, np.generic):
        return x.item()
    return x


def _walk(x):
    x = _to_py(x)
    if isinstance(x, dict):
        return {k: _walk(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_walk(v) for v in x]
    return x


# glue exports
sb.glue("Rd", _walk(Rd))
sb.glue("centered", _walk(centered))
sb.glue("information", _walk(information))
sb.glue("iteration", _walk(iteration))
sb.glue("kernel", _walk(kernel))
sb.glue("logz", _walk(logz))
sb.glue("logzerr", _walk(logzerr))
sb.glue("ncall", _walk(ncall))
sb.glue("ndim", _walk(ndim))
sb.glue("niter", _walk(niter))
sb.glue("normalized", _walk(normalized))
sb.glue("seed", _walk(seed))
sb.glue("walltime", _walk(walltime))
