# %%

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from prism.pack import NormalizedPACK
from prism.svi import pad_waveforms, sample_batch
from surrogate import source
from utils.jax import vk

# %%
lf_samples = source.get_lf_samples()[:500]


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


lf_samples = [
    {**s, "du": s["u"], "tau": warp_time(s["t"], s["p"]["T0"])}
    for s in lf_samples
]

dataset = pad_waveforms(lf_samples)


# %%
BATCH_SIZE = 16
WIDTH = 384

batch = sample_batch(vk(), dataset, BATCH_SIZE, WIDTH)
# leaves shaped (BATCH_SIZE, WIDTH)


plt.plot(batch.tau[0], batch.du[0])
plt.title("Example batch waveform")


# %%
def get_batch(key, index):
    return sample_batch(key, dataset, BATCH_SIZE, WIDTH)


# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891/c/696a3d30-510c-8332-8075-7046937ecb61
# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891/c/696a0429-7d2c-8332-b396-9f357bb92293

# Do PCA with scree plot to determine good proxy for NUM_INDUCING
# We can still do PCA dim reduction INSIDE latent space

# %%
kernel = NormalizedPACK(d=1)

NUM_INDUCING = 32

Z = jnp.linspace(0.0, 1.0, NUM_INDUCING)[:, None]

test_index = 300

t = batch.tau[test_index][:, None]
y = batch.du[test_index]
mask = batch.mask[test_index]

obs_stddev = 0.85
sigma2 = obs_stddev**2

mask = mask.at[:10].set(0.0)  # artificial missing data

print(mask)

# Reference
import gpjax as gpx

tmasked = t[mask.astype(bool)]
ymasked = y[mask.astype(bool)]

meanf = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(
    num_datapoints=len(ymasked), obs_stddev=obs_stddev
)
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
p = prior * likelihood


q = gpx.variational_families.CollapsedVariationalGaussian(
    posterior=p, inducing_inputs=Z
)

d = gpx.Dataset(X=tmasked, y=jnp.reshape(ymasked, (-1, 1)))

reference_elbo = gpx.objectives.collapsed_elbo(q, d)
print("Reference ELBO:", reference_elbo)


import jax.numpy as jnp
import jax.scipy as jsp


def collapsed_elbo_masked_single_like_gpjax(
    kernel, Z, t, y, mask, sigma2, jitter
):
    # Force shapes like GPJax: y is (W,1)
    if y.ndim == 1:
        y = y[:, None]
    if mask.ndim == 1:
        mask = mask[:, None]

    n_eff = jnp.sum(mask)  # scalar (float)

    M = Z.shape[0]

    # Kzz with explicit jitter
    Kzz = kernel.gram(Z).to_dense()
    Kzz = Kzz + jitter * jnp.eye(M)

    # Kzx, Kxx_diag exactly as GPJax
    Kzx = kernel.cross_covariance(Z, t)  # (M,W)
    Kxx_diag = jax.vmap(kernel, in_axes=(0, 0))(t, t)  # (W,)

    # Apply mask (mask is (W,1), broadcast where needed)
    mask_w = mask[:, 0]  # (W,)
    Kzx = Kzx * mask_w[None, :]
    y = y * mask
    Kxx_diag = Kxx_diag * mask_w

    # mean function is zero in your test, so mu=0
    diff = y

    Lz = jnp.linalg.cholesky(Kzz)

    A = jsp.linalg.solve_triangular(Lz, Kzx, lower=True) / jnp.sqrt(
        sigma2
    )  # (M,W)
    AAT = A @ A.T
    B = jnp.eye(M) + AAT
    L = jnp.linalg.cholesky(B)

    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

    L_inv_A_diff = jsp.linalg.solve_triangular(L, A @ diff, lower=True)  # (M,1)

    quad = (
        jnp.sum(diff * diff) - jnp.sum(L_inv_A_diff * L_inv_A_diff)
    ) / sigma2

    two_log_prob = -n_eff * jnp.log(2.0 * jnp.pi * sigma2) - log_det_B - quad
    two_trace = jnp.sum(Kxx_diag) / sigma2 - jnp.trace(AAT)

    return 0.5 * (two_log_prob - two_trace)


elbo2 = collapsed_elbo_masked_single_like_gpjax(
    kernel, Z, t, y, mask, sigma2, jitter=1e-6
)
print("ELBO masked single like GPJax:", elbo2)

# %%
