# %%
# parameters, export
seed = 545465
Rd = 1.2
kernel = "tack:1"
centered = False
normalized = False
iteration = 1

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
jax.config.update("jax_platform_name", "cpu")

from tinygp.kernels import Exp, ExpSineSquared, ExpSquared, Matern32, Matern52

from gfm.ack import STACK, TACK
from gfm.lf import lf_relaxation_open_phase

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
def instantiate_kernel(kernel, theta):
    match kernel:
        case "matern:12":
            k = theta["sigma_a"] * Exp(scale=theta["ell"])
        case "matern:32":
            k = theta["sigma_a"] * Matern32(scale=theta["ell"])
        case "matern:52":
            k = theta["sigma_a"] * Matern52(scale=theta["ell"])
        case "matern:inf":
            k = theta["sigma_a"] * ExpSquared(scale=theta["ell"])
        case "periodickernel":
            # Parametrization (r, T) agrees with src.iklp.periodic.periodic_kernel_generator() [but the latter calculates the time indices t differently; we have PERIOD inclusive and the latter exclusive]
            r = theta["r"]
            T = tc
            gamma = 1.0 / (2.0 * r**2)
            k = theta["sigma_a"] * ExpSineSquared(scale=T, gamma=gamma)
        case _ if "tack" in kernel:
            d = int(kernel[-1])
            center = t.mean() if centered else 0.0

            if "stack" in kernel:
                k = STACK(d=d, normalized=normalized, center=center)
            else:
                LSigma = jnp.diag(
                    jnp.array([theta["sigma_b"], theta["sigma_c"]])
                )
                k = theta["sigma_a"] * TACK(
                    d=d, normalized=normalized, center=center, LSigma=LSigma
                )
        case _:
            raise NotImplementedError(f"Kernel {kernel} not implemented")

    return k


theta = {
    "sigma_a": 5.0,
    "ell": 1.789,
    "r": 0.78113212,
    "sigma_b": 3.0,
    "sigma_c": 0.5,
}

k = instantiate_kernel(kernel, theta)

K = k(t, t)

# %%
from tinygp.gp import GaussianProcess

gp = GaussianProcess(kernel=k, X=t)

s = gp.sample(jax.random.PRNGKey(seed))

plt.title(f"kernel: {kernel}, centered: {centered}, normalized: {normalized}")
plt.plot(t, s, label="sample from GP prior")
plt.legend()


# %%
from scipy.special import ndtri


def build_gp(theta):
    k = instantiate_kernel(kernel, theta)
    gp = GaussianProcess(kernel=k, X=t, diag=theta["sigma_noise"] ** 2)
    return gp


def build_theta(x, kernel):
    match kernel:
        case _ if "matern" in kernel:
            return {
                "sigma_noise": x[0],
                "sigma_a": x[1],
                "ell": x[2],
            }
        case "periodickernel":
            return {
                "sigma_noise": x[0],
                "sigma_a": x[1],
                "r": x[2],
            }
        case _ if "stack" in kernel:
            return {
                "sigma_noise": x[0],
            }
        case _ if "tack" in kernel:
            return {
                "sigma_noise": x[0],
                "sigma_a": x[1],
                "sigma_b": x[2],
                "sigma_c": x[3],
            }
        case _:
            raise NotImplementedError(f"Kernel {kernel} not implemented")


@jax.jit
def loglikelihood(x):
    theta = build_theta(x, kernel)
    gp = build_gp(theta)
    return gp.log_probability(du)


def ptform(u):
    z = ndtri(u)
    return 10.0**z


# %%

x = ptform(np.random.uniform(size=100))
theta = build_theta(x, kernel)
ndim = len(theta)

loglikelihood(x)

# %%

from dynesty import NestedSampler

# initialize our nested sampler
nlive = 500

sampler = NestedSampler(loglikelihood, ptform, ndim, nlive=nlive)

sampler.run_nested(maxiter=10_000, print_progress=False)

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
