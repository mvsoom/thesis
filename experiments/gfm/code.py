# %%
# parameters, export
seed = 545465
examplar_name = "soft_gci"
d = 100
kernel = "tack:1"
centered = False
normalized = False
data_file = "/home/marnix/thesis/experiments/gfm/data/hard_gci/lf.dat"
iteration = 0

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)

from tinygp.kernels import Exp, ExpSineSquared, ExpSquared, Matern32, Matern52

from gfm.ack import TACK

# %%

def get_data():
    data = np.loadtxt(data_file, skiprows=1)
    t = data[:, 0]
    du = data[:, 1]
    u = data[:, 2]
    return t, du, u


t, du, u = get_data()

plt.title(f"Test data: {examplar_name}, d={d}")
plt.plot(t, du, label="du")
plt.plot(t, u, label="u")
plt.legend()


# %%


PERIOD = t[-1] - t[0]


def instantiate_kernel(kernel, theta):
    match kernel:
        case "matern:12":
            k = Exp(scale=theta["ell"])
        case "matern:32":
            k = Matern32(scale=theta["ell"])
        case "matern:52":
            k = Matern52(scale=theta["ell"])
        case "matern:inf":
            k = ExpSquared(scale=theta["ell"])
        case "periodickernel":
            # Parametrization (r, T) agrees with src.iklp.periodic.periodic_kernel_generator() [but the latter calculates the time indices t differently; we have PERIOD inclusive and the latter exclusive]
            r = theta["r"]
            T = PERIOD
            gamma = 1.0 / (2.0 * r**2)
            k = ExpSineSquared(scale=T, gamma=gamma)
        case _ if "tack" in kernel:
            d = int(kernel[-1])
            center = t.mean() if centered else 0.0
            LSigma = jnp.diag(jnp.array([theta["sigma_b"], theta["sigma_c"]]))
            k = TACK(d=d, normalized=normalized, center=center, LSigma=LSigma)
        case _:
            raise NotImplementedError(f"Kernel {kernel} not implemented")

    return theta["sigma_a"] * k


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
    std = jnp.sqrt(var)

    plt.fill_between(
        t,
        mu - 1.96 * std,
        mu + 1.96 * std,
        alpha=0.2,
    )
    plt.plot(t, mu, label="GP posterior mean")

plt.title(f"kernel: {kernel}, centered: {centered}, normalized: {normalized}")
plt.legend()

# %%
from dynesty import plotting as dyplot

fig, ax = dyplot.cornerplot(
    res,
    labels=[str(k) for k in theta.keys()],
    verbose=True,
    quantiles=[0.05, 0.5, 0.95],
)

# %%
# export
logz = res.logz[-1]
logzerr = res.logzerr[-1]
ndim = res.samples.shape[1]
information = res.information[-1]