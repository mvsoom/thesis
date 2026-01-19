# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from tqdm import tqdm

from prism.pack import NormalizedPACK
from surrogate import source


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


def get_waveforms(lf_samples):
    for s in tqdm(lf_samples):
        tau = warp_time(s["t"], s["p"]["T0"])
        du = s["u"]
        yield tau, du


def pad_waveforms(waveforms, width=None, dtype=jnp.float64):
    n = len(waveforms)

    if width is None:
        width = max(len(tau) for tau, du in waveforms)

    X = np.full((n, width), np.nan)
    y = np.full((n, width), np.nan)

    i = 0
    for tau, du in tqdm(waveforms, desc=f"Padding to width={width}"):
        m = min(width, len(tau))
        X[i, :m] = tau[:m]
        y[i, :m] = du[:m]
        i += 1

    return jnp.array(X, dtype=dtype), jnp.array(y, dtype=dtype)


def get_data(n=None, width=None):
    lf_samples = source.get_lf_samples()[:n]

    waveforms = list(get_waveforms(lf_samples))

    X, y = pad_waveforms(waveforms, width=width)
    return X, y


def collapsed_elbo_masked(q, t, y, jitter=1e-6):
    """
    Collapsed Titsias (2009) ELBO for a single waveform (t, y) with NaN-masked padding.

    Parameters
    ----------
    q : CollapsedVariationalGaussian
    t_row : [W]      time inputs (tau), NaN where padded
    y_row : [W]      observations, NaN where padded
    jitter : float   diagonal jitter for Kzz

    Returns
    -------
    elbo : scalar
    """

    mask_w = ~jnp.isnan(y)  # [W] boolean
    mask = mask_w[:, None]  # [W,1]

    # promote t,y to GPJax shapes
    t = t[:, None]  # [W,1]
    y = y[:, None]  # [W,1]

    # zero-out padded entries
    t = jnp.where(mask, t, 0.0)
    y = jnp.where(mask, y, 0.0)

    n_eff = jnp.sum(mask_w)

    # Model
    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs
    sigma2 = q.posterior.likelihood.obs_stddev**2

    M = Z.shape[0]

    # Kernel matrices
    Kzz = kernel.gram(Z).to_dense()
    Kzz = Kzz + jitter * jnp.eye(M)

    Kzx = kernel.cross_covariance(Z, t)  # [M,W]
    Kxx_diag = jax.vmap(kernel, in_axes=(0, 0))(t, t)  # [W]

    # Apply mask
    Kzx = Kzx * mask_w[None, :]
    Kxx_diag = Kxx_diag * mask_w

    # Calculate the ELBO
    Lz = jnp.linalg.cholesky(Kzz)

    A = jsp.linalg.solve_triangular(Lz, Kzx, lower=True) / jnp.sqrt(
        sigma2
    )  # [M,W]

    AAT = A @ A.T
    B = jnp.eye(M) + AAT
    L = jnp.linalg.cholesky(B)

    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

    diff = y  # zero-mean GP
    L_inv_A_diff = jsp.linalg.solve_triangular(L, A @ diff, lower=True)  # [M,1]

    quad = (
        jnp.sum(diff * diff) - jnp.sum(L_inv_A_diff * L_inv_A_diff)
    ) / sigma2

    two_log_prob = -n_eff * jnp.log(2.0 * jnp.pi * sigma2) - log_det_B - quad

    two_trace = jnp.sum(Kxx_diag) / sigma2 - jnp.trace(AAT)

    return 0.5 * (two_log_prob - two_trace)


def batch_collapsed_elbo_masked(q, data, I_total):
    """
    q: CollapsedVariationalGaussian
    data: Dataset with
          data.X shape [B, W]
          data.y shape [B, W]
          NaNs indicate masked points
    I_total: total number of waveforms in the full dataset
    """

    X = data.X  # [B, W]
    y = data.y  # [B, W]
    B = X.shape[0]

    # vmap over waveforms (rows)
    elbos = jax.vmap(
        collapsed_elbo_masked,
        in_axes=(None, 0, 0),
    )(q, X, y)  # X,y are [B,W]

    # unbiased waveform-level scaling
    elbo = (I_total / B) * jnp.sum(elbos)
    return elbo


if __name__ == "__main__":
    X, y = get_data(n=20)
    kernel = NormalizedPACK(d=1)

    NUM_INDUCING = 32
    Z = jnp.linspace(0.0, 1.0, NUM_INDUCING)[:, None]

    obs_stddev = 0.85
    sigma2 = obs_stddev**2

    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    for test_index in [0, 13]:  # [nonpadded, padded]
        tau = X[test_index][:, None]
        du = y[test_index]
        mask = ~jnp.isnan(du)

        print("Case test_index =", test_index)
        print("  # of masked points:", jnp.sum(~mask))

        # Calculate reference ELBO of gpjax
        tmasked = tau[mask]
        ymasked = du[mask]

        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=len(ymasked), obs_stddev=obs_stddev
        )
        p = prior * likelihood
        q = gpx.variational_families.CollapsedVariationalGaussian(
            posterior=p, inducing_inputs=Z
        )

        d = gpx.Dataset(X=tmasked, y=jnp.reshape(ymasked, (-1, 1)))

        reference_elbo = gpx.objectives.collapsed_elbo(q, d)
        print("  Reference ELBO:", reference_elbo)

        # Calculate masked ELBO
        our_elbo = collapsed_elbo_masked(q, tau.squeeze(), du)
        print("  Our masked ELBO:", our_elbo)

# %%
import jax.numpy as jnp


def infer_eps_posterior_single(q, t_row, y_row, jitter=1e-6):
    """
    Infer posterior over whitened amplitudes eps for ONE waveform.

    Parameters
    ----------
    q : trained CollapsedVariationalGaussian
    t_row : [W]   time grid (tau), NaN padded
    y_row : [W]   observations, NaN padded

    Returns
    -------
    mu_eps : [M]
    Sigma_eps : [M, M]
    """

    mask_w = ~jnp.isnan(y_row)  # [W]
    t = t_row[:, None]  # [W,1]
    y = y_row[:, None]  # [W,1]

    t = jnp.where(mask_w[:, None], t, 0.0)
    y = jnp.where(mask_w[:, None], y, 0.0)

    # Model
    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs
    sigma2 = q.posterior.likelihood.obs_stddev**2

    M = Z.shape[0]

    # Psi(t)
    Kzz = kernel.gram(Z).to_dense()
    Kzz = Kzz + jitter * jnp.eye(M)
    Lzz = jnp.linalg.cholesky(Kzz)

    Kzx = kernel.cross_covariance(Z, t)  # [M,W]
    Kzx = Kzx * mask_w[None, :]  # drop padded points

    Psi = jsp.linalg.solve_triangular(Lzz, Kzx, lower=True).T  # [W,M]

    # BLR posterior
    # A = Psi / sigma
    A = Psi / jnp.sqrt(sigma2)  # [W,M]

    # precision = I + A^T A
    precision = jnp.eye(M) + A.T @ A
    Lp = jnp.linalg.cholesky(precision)
    Sigma_eps = jsp.linalg.cho_solve((Lp, True), jnp.eye(M))

    mu_eps = (Sigma_eps @ (Psi.T @ y) / sigma2).squeeze()  # [M]

    return mu_eps, Sigma_eps

