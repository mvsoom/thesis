# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax as ox
import plotly.express as px
from flax import nnx
from gpjax.parameters import Parameter
from tqdm import tqdm

from prism.pack import NormalizedPACK
from utils.jax import nocheck, safe_cholesky


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


def init_Z_grid(key, M):
    """Initialize inducing inputs Z on a jittered uniform grid in [0,1]

    This is needed because uniform random init has (closest point distance) scale like 1/M^2, leading to ill-conditioned Kzz for large M.
    """
    key, subkey = jax.random.split(key)

    z = (jnp.arange(M) + 0.5) / M
    z = z[:, None]

    z = z + 1e-3 / M * jax.random.normal(key, z.shape)  # break symmetry
    shift = jax.random.uniform(subkey, ())  # random phase

    z = jnp.mod(z + shift, 1.0)
    return z


def collapsed_elbo_masked(q, t, y):
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
    jitter = q.jitter

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

    Kzx = kernel.cross_covariance(Z, t)  # [M,W]
    Kxx_diag = jax.vmap(kernel, in_axes=(0, 0))(t, t)  # [W]

    # Apply mask
    Kzx = Kzx * mask_w[None, :]
    Kxx_diag = Kxx_diag * mask_w

    # Calculate the ELBO
    Lz = safe_cholesky(Kzz, jitter=jitter)

    A = jsp.linalg.solve_triangular(Lz, Kzx, lower=True) / jnp.sqrt(
        sigma2
    )  # [M,W]

    AAT = A @ A.T
    B = jnp.eye(M) + AAT
    L = safe_cholesky(B, jitter=jitter)

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
    from surrogate.prism import get_train_data

    X, y, oq = get_train_data(n=20)
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
def optimize(key, model, dataset, lr, batch_size, num_iters, **fit_kwargs):
    key, subkey = jax.random.split(key)

    N = dataset.X.shape[0]
    model = model(subkey)
    optim = ox.adam(learning_rate=lr)

    def cost(q, d):
        return -batch_collapsed_elbo_masked(q, d, N)

    fitted, cost_history = gpx.fit(
        model=model,
        objective=cost,
        train_data=dataset,
        optim=optim,
        num_iters=num_iters,
        key=key,
        batch_size=batch_size,
        trainable=Parameter,
        **fit_kwargs,
    )

    elbo_history = -cost_history
    return fitted, elbo_history


def optimize_restarts(optimize, num_restarts, key):
    subkeys = jax.random.split(key, num_restarts)

    def optimize_state(subkey):
        fitted, elbo_history = optimize(subkey)
        return nnx.state(fitted), elbo_history

    states, elbo_histories = jax.vmap(optimize_state)(subkeys)
    return states, elbo_histories


def optimize_restarts_scan(optimize, num_restarts, key):
    """Restarts with sequential scan to prevent OOM"""
    subkeys = jax.random.split(key, num_restarts)

    def optimize_state(subkey):
        fitted, elbo_history = nocheck(optimize)(subkey)
        return nnx.state(fitted), elbo_history

    states, elbo_histories = jax.lax.map(optimize_state, subkeys)
    return states, elbo_histories


def pick_best(states, elbo_histories, q):
    i = int(np.nanargmax(np.array([h[-1] for h in elbo_histories])))

    graphdef, _ = nnx.split(q)

    best_fitted = nnx.merge(graphdef, jax.tree.map(lambda x: x[i], states))
    best_elbo_history = elbo_histories[i]

    return best_fitted, best_elbo_history

# %%
def infer_eps_posterior_single(q, t_row, y_row):
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
    jitter = q.jitter

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
    Lzz = safe_cholesky(Kzz, jitter=jitter)

    Kzx = kernel.cross_covariance(Z, t)  # [M,W]
    Kzx = Kzx * mask_w[None, :]  # drop padded points

    Psi = jsp.linalg.solve_triangular(Lzz, Kzx, lower=True).T  # [W,M]

    # BLR posterior
    # A = Psi / sigma
    A = Psi / jnp.sqrt(sigma2)  # [W,M]

    # precision = I + A^T A
    precision = jnp.eye(M) + A.T @ A
    Lp = safe_cholesky(precision, jitter=jitter)
    Sigma_eps = jsp.linalg.cho_solve((Lp, True), jnp.eye(M))

    mu_eps = (Sigma_eps @ (Psi.T @ y) / sigma2).squeeze()  # [M]

    return mu_eps, Sigma_eps


def do_prism(q, dataset, device=jax.devices("cpu")[0]):
    """Calculate the amplitude posterior for all waveforms in dataset, thereby refracting the dataset like a prism into latent space"""
    with jax.default_device(device):
        mu_eps, Sigma_eps = jax.vmap(
            infer_eps_posterior_single,
            in_axes=(None, 0, 0),
        )(q, dataset.X, dataset.y)
    return mu_eps, Sigma_eps


def gp_posterior_mean_from_eps(q, t_star, mu_eps):
    """
    GP posterior mean at t_star for ONE waveform,
    given inferred mu_eps.

    t_star : [T] query points
    mu_eps : [M]
    """
    jitter = q.jitter
    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs

    # Kzz and its Cholesky
    Kzz = kernel.gram(Z).to_dense()
    Lzz = safe_cholesky(Kzz, jitter=jitter)

    # E[u | y]
    u_mean = Lzz @ mu_eps  # [M]

    # K_{t*Z}
    t_star = t_star[:, None]
    KtZ = kernel.cross_covariance(t_star, Z)  # [T,M]

    # posterior mean
    f_mean = KtZ @ jsp.linalg.solve(Kzz, u_mean)

    return f_mean.squeeze()


def svi_basis(q, t):
    kernel = q.posterior.prior.kernel
    Z = q.inducing_inputs.value

    Kmm = kernel.gram(Z).to_dense()
    Lzz = safe_cholesky(Kmm)  # Kmm = Lzz @ Lzz.T

    x = jnp.asarray(t).reshape(1, -1)
    Kxz = kernel.cross_covariance(x, Z)
    return jax.scipy.linalg.solve_triangular(
        Lzz, Kxz.T, lower=True
    ).squeeze()  # psi = Lzz^{-1} Kzx

# %%
def reconstruct_waveforms(mu_eps, qsvi, train_data, test_indices, tau_test):
    """Is the learned RKHS rich enough to reconstruct some test waveforms?"""
    f_means = jax.vmap(
        lambda eps: gp_posterior_mean_from_eps(qsvi, tau_test, eps)
    )(mu_eps[test_indices])

    plot_rows = []
    panel_order = []
    for idx, f_mean in zip(test_indices, f_means):
        idx_int = int(idx)
        panel_label = f"test_index={idx_int}"
        panel_order.append(panel_label)

        x_data = np.array(train_data.X[idx_int])
        y_data = np.array(train_data.y[idx_int])
        for x_val, y_val in zip(x_data, y_data):
            plot_rows.append(
                {
                    "tau": x_val,
                    "value": y_val,
                    "series": "Data",
                    "panel": panel_label,
                }
            )

        x_pred = np.array(tau_test)
        y_pred = np.array(f_mean)
        for x_val, y_val in zip(x_pred, y_pred):
            plot_rows.append(
                {
                    "tau": x_val,
                    "value": y_val,
                    "series": "Posterior mean",
                    "panel": panel_label,
                }
            )

    fig = px.line(
        plot_rows,
        x="tau",
        y="value",
        color="series",
        facet_col="panel",
        facet_col_wrap=2,
        category_orders={"panel": panel_order},
        title="Posterior mean vs data (selected test indices)",
        labels={"tau": "tau", "value": "u'(t)", "series": ""},
    )

    return fig

# %%
def whitening_from_cov(Sigma_bar, eps=1e-12):
    w, V = jnp.linalg.eigh(Sigma_bar)
    w = jnp.clip(w, eps, None)
    W = V @ jnp.diag(1.0 / jnp.sqrt(w)) @ V.T
    Wi = V @ jnp.diag(jnp.sqrt(w)) @ V.T
    return W, Wi


def make_whitener(mu_eps, Sigma_eps):
    """
    Build whitening / unwhitening closures from a dataset of (mu_eps, Sigma_eps).

    Returns:
        whiten(eps, Sigma)   -> (eps_w, Sigma_w)
        unwhiten(eps_w, Sigma_w) -> (eps, Sigma)
        offdiag_energy (diagnostic)
    """
    mu0 = mu_eps.mean(axis=0)
    Sigma_bar = Sigma_eps.mean(axis=0)

    W, Wi = whitening_from_cov(Sigma_bar)

    def whiten_fn(mu, Sigma):
        mu_w = (mu - mu0) @ W.T
        Sigma_w = jnp.einsum("ij,njk,kl->nil", W, Sigma, W.T)
        return mu_w, Sigma_w

    def unwhiten_fn(mu_w, Sigma_w):
        mu = mu_w @ Wi.T + mu0
        Sigma = jnp.einsum("ij,njk,kl->nil", Wi, Sigma_w, Wi.T)
        return mu, Sigma

    return whiten_fn, unwhiten_fn


def offdiag_energy_fraction(C):
    diag = jnp.diagonal(C, axis1=1, axis2=2)
    diag_energy = jnp.sum(diag * diag)
    total_energy = jnp.sum(C * C)
    return 1.0 - diag_energy / total_energy


# %%
def latent_pair_density(Xmu, Xvar, pair, nx=200, ny=200, pad=0.5):
    i, j = pair
    mu = Xmu[:, [i, j]]  # (N,2)
    var = Xvar[:, [i, j]]  # (N,2)

    xmin, xmax = mu[:, 0].min() - pad, mu[:, 0].max() + pad
    ymin, ymax = mu[:, 1].min() - pad, mu[:, 1].max() + pad

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    logp = np.full(grid.shape[0], -np.inf)

    for n in range(mu.shape[0]):
        diff = grid - mu[n]
        invvar = 1.0 / var[n]
        quad = diff[:, 0] ** 2 * invvar[0] + diff[:, 1] ** 2 * invvar[1]
        logdet = np.log(var[n]).sum()
        lp = -0.5 * (quad + logdet + 2 * np.log(2 * np.pi))
        logp = np.logaddexp(logp, lp)

    logp -= np.log(mu.shape[0])
    dens = np.exp(logp).reshape(ny, nx)

    extent = [xmin, xmax, ymin, ymax]
    return dens, extent