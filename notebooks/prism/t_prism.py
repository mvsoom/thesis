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
import gpjax as gpx
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.variational_families import CollapsedVariationalGaussian

from prism.svi import (
    init_Z_grid,
)
from utils.jax import vk


def collapsed_svi(key=vk(), M=12, jitter=1e-6):
    Z = init_Z_grid(key, M) * t[-1]

    kernel = RBF()
    prior = gpx.gps.Prior(kernel, Zero())
    likelihood = Gaussian(num_datapoints=len(t))
    posterior = prior * likelihood

    return CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=Z, jitter=jitter
    )


q = collapsed_svi()


# %%
from prism import svi, t_svi

nu = 1e6

elbo = svi.collapsed_elbo_masked(q, t, y)
elbo_t = t_svi.collapsed_elbo_masked_t(q, t, y, nu)

print(f"ELBO  : {elbo:.4f}")
print(f"ELBO_t: {elbo_t:.4f}")
# OK, these match


# %%
# Apply a random mask
import numpy as np

mask = np.random.rand(len(t)) < 0.5
t_mask = np.where(mask, t, np.nan)
y_mask = np.where(mask, y, np.nan)

elbo_m = svi.collapsed_elbo_masked(q, t_mask, y_mask)
elbo_t_m = t_svi.collapsed_elbo_masked_t(q, t_mask, y_mask, nu)

print(f"ELBO  : {elbo_m:.4f}")
print(f"ELBO_t: {elbo_t_m:.4f}")