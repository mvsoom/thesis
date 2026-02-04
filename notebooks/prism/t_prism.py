# %%
%pylab

t = array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    ]
)
y = array(
    [
        0.70329138,
        0.74818803,
        0.75587486,
        0.76342799,
        0.76715587,
        0.76715587,
        0.77085201,
        0.77085201,
        0.77451697,
        0.77815125,
        0.78887512,
        0.79934055,
        0.8260748,
        2.15760785,
        0.8893017,
        0.94448267,
        0.94939001,
        0.96378783,
        0.96614173,
        0.97312785,
        0.97081161,
        0.96614173,
        0.96142109,
        0.96378783,
        0.96378783,
        0.96378783,
        0.96142109,
        0.95664858,
        0.95182304,
        0.95664858,
        0.96142109,
    ]
)

plot(t, y, "o")

# %%
# IMPORTANT: FIXME: data needs to be standardized for t-PRISM approximation to be maximally effective
# We don't learn means AND error in approximation scales like 1/(nu obs_stddev^2) so we zero-mean and keep variances close to 1
y = (y - np.mean(y)) / np.std(y)

plot(t, y, "o")

# %%
import numpy as np
import gpjax as gpx
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero, Constant
from gpjax.variational_families import CollapsedVariationalGaussian

import jax
from prism.svi import (
    init_Z_grid,
)
from utils.jax import vk

mask = np.random.rand(len(t)) <= 0.5
key = vk()

# %%

def collapsed_svi(key=vk(), M=12, jitter=1e-6, lengthscale=1.0, variance=1.0, obs_stddev=1.0):
    Z = init_Z_grid(key, M) * t[-1]

    kernel = RBF(lengthscale=lengthscale, variance=variance)
    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=len(t), obs_stddev=obs_stddev)
    posterior = prior * likelihood

    return CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=Z, jitter=jitter
    )

q = collapsed_svi(key, M=12, jitter=1e-6)


from prism import svi, t_svi

nu = 1e6
num_inner = 3

elbo = svi.collapsed_elbo_masked(q, t, y)
elbo_t = t_svi.collapsed_elbo_masked_t(q, t, y, nu, num_inner=num_inner)

print(f"ELBO  : {elbo:.4f}")
print(f"ELBO_t: {elbo_t:.4f}")
# these matched before your patches

# Apply a random mask
t_mask = np.where(mask, t, np.nan)
y_mask = np.where(mask, y, np.nan)

elbo_m = svi.collapsed_elbo_masked(q, t_mask, y_mask)
elbo_t_m = t_svi.collapsed_elbo_masked_t(q, t_mask, y_mask, nu, num_inner=num_inner)

print(f"ELBO[masked]  : {elbo_m:.4f}")
print(f"ELBO_t[masked]: {elbo_t_m:.4f}")

mu, Sigma = svi.infer_eps_posterior_single(q, t_mask, y_mask)
mu_t, Sigma_t = t_svi.infer_eps_posterior_single_t(q, t_mask, y_mask, nu)

print("Posterior mean max abs diff:", np.max(np.abs(mu - mu_t)))
print("Posterior cov max abs diff :", np.max(np.abs(Sigma - Sigma_t)))

# ALL OK: correspondence works

# %%
# Compare posteriors

# tune to data manually
q = collapsed_svi(key, M=12, jitter=1e-6, lengthscale=10.0, variance=1.0, obs_stddev=0.01)

nu = 1.0

mu, Sigma = svi.infer_eps_posterior_single(q, t, y)
mu_t, Sigma_t = t_svi.infer_eps_posterior_single_t(q, t, y, nu)

from prism.svi import gp_posterior_mean_from_eps

plot(t, y, "o", label="data")

y_star = gp_posterior_mean_from_eps(q, t, mu)
y_star_t = gp_posterior_mean_from_eps(q, t, mu_t)

plot(t, y_star, label="posterior mean")
plot(t, y_star_t, "--", label="posterior mean t")
legend()

# %%
