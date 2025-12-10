# %%
import jax
import matplotlib.pyplot as plt
import numpy as np

from gp.blr import blr_from_mercer
from gp.periodic import SPACK
from utils.jax import vk

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


import jax.numpy as jnp

from utils import lfmodel

# %%
# Generate examplar
T = 6.0  # ms
Rd = 1.0  # 0.3 - 2.7; 1.0 is modal
N = 256


def generate_examplar(T, Rd, N):
    p = lfmodel.convert_lf_params({"T0": T, "Rd": Rd}, "Rd -> T")
    t = np.linspace(0, T, N)
    dt = t[1] - t[0]

    du = np.array(lfmodel.dgf(t, p))
    u = np.cumsum(du) * dt

    # Do gauge
    te = t[np.argmin(du)]  # instant of peak excitation
    t = t - te  # center time axis
    to = 0.0 - te  # time of opening where du is zero

    power = (du**2).sum() * dt / T
    du /= np.sqrt(power)
    u /= np.sqrt(power)

    d = {
        "Rd": Rd,
        "t": t,
        "to": to,
        "du": du,
        "u": u,
    }
    return d


lf = generate_examplar(T, Rd, N)

# Inspect
plt.title(f"Data to fit: Rd={Rd}")
plt.plot(lf["t"], lf["du"], label="du")
plt.plot(lf["t"], lf["u"], label="u")
plt.legend()

# %%

# Fix hyperparams and infer amplitudes
sigma_noise = 1e-2
sigma_noise_db = 20 * np.log10(sigma_noise)
print(f"Using noise std: {sigma_noise} ({sigma_noise_db} dB)")

t1 = 0.0
t2 = T

fs = 16000.0
F0 = 1000.0 / T  # Hz equivalent
num_harmonics = int(np.floor((fs / F0) / 2))

num_periods = 6

N = int((T / 1000) * fs) * num_periods
t, dt = jnp.linspace(
    -T * num_periods / 2, T * num_periods / 2, N * 10, retstep=True
)

d = 1

kernel = SPACK(d, T, num_harmonics, t1, t2)


# %%
def infer_amplitudes(lf, plot=False):
    t = lf["t"]
    dt = t[1] - t[0]
    du = lf["du"]

    gp = blr_from_mercer(kernel, t, noise_variance=sigma_noise**2)

    Phi = gp.state.Phi
    L = gp.cov_root

    if plot:
        Phi_weighted = Phi @ L  # (N, 2W)

        plt.plot(t, du, c="black", label="du (data)")
        plt.plot(t, Phi_weighted, alpha=1.0)
        plt.title(f"Basis functions weighted by prior covariance (d={d})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    conditioned = gp.condition(du)
    cgp = conditioned.gp

    mu, cov_root = cgp.mu, cgp.cov_root

    if plot:
        # draw posterior samples
        num_samples = 6

        plt.figure(figsize=(6, 4))
        plt.plot(t, du, label="data", c="black", linewidth=2)
        plt.plot(t, cgp.mean, label="posterior mean", c="red", linewidth=2)

        for k in range(num_samples):
            du_sample = cgp.sample(vk())
            u_sample = jnp.cumsum(du_sample * dt)
            plt.plot(t, du_sample, c="blue", alpha=0.1)
            plt.plot(t, u_sample, c="green", alpha=0.1)

        plt.legend()
        plt.title(f"Posterior mean and samples (Rd = {lf['Rd']})")
        plt.xlabel("time (ms)")
        plt.ylabel("amplitude")
        plt.show()

    return {"mu": mu, "cov_root": cov_root}


_ = infer_amplitudes(lf, plot=True)

# %%
Rds = np.linspace(0.3, 2.7, 10)
lfs = [generate_examplar(T, Rd, N) for Rd in Rds]

for i, lf in enumerate(lfs):
    polarity = 1  # (-1) ** i
    lf["du"] *= polarity
    lf["u"] *= polarity

posteriors = [infer_amplitudes(lf) for lf in lfs]


# %%
def envelope_gaussians(posteriors):
    """
    Compute the Gaussian q = N(mu_star, Sigma_star) minimizing
        (1/K) sum_i KL(p_i || q)
    where p_i = N(mu_i, L_i L_i.T).

    posteriors: list of dicts with keys "mu" and "cov_root".
    """
    # stack mus: (K,M)
    mus = jnp.stack([p["mu"] for p in posteriors], axis=0)

    # stack cov_roots: (K,M,R)
    Ls = jnp.stack([p["cov_root"] for p in posteriors], axis=0)

    K = mus.shape[0]

    # mixture mean: (M,)
    mu_star = jnp.mean(mus, axis=0)

    # E[ ww^T ] term = Sigma_i + mu_i mu_i^T
    # Sigma_i = L_i L_i^T
    # do: batch matmul (K,M,R) @ (K,R,M) -> (K,M,M)
    Sigma_i = Ls @ jnp.swapaxes(Ls, -2, -1)

    # outer mus: (K,M,M)
    mu_outer = mus[..., None] * mus[:, None, :]

    # second moment average: (M,M)
    second = jnp.mean(Sigma_i + mu_outer, axis=0)

    # covariance = second - mu* mu*.T
    Sigma_star = second - mu_star[:, None] * mu_star[None, :]

    return mu_star, Sigma_star


mu_star, Sigma_star = envelope_gaussians(posteriors)

plt.plot(mu_star, label="Mixture mean")
for i, p in enumerate(posteriors):
    plt.plot(p["mu"], alpha=0.3, label=f"Posterior mean Rd={Rds[i]}")
plt.title("Mixture mean and individual posterior means")
# plt.legend()
plt.show()

# %%
from gp.blr import BayesianLinearRegressor

Sigma_star_root = jnp.linalg.cholesky(
    Sigma_star + 1e-9 * jnp.eye(Sigma_star.shape[0])
)

gpl = BayesianLinearRegressor(
    kernel.compute_phi, t, mu=mu_star, cov_root=Sigma_star_root
)

# %%
from utils.reskew import dgf_polarity

num_samples = 1

plt.figure(figsize=(6, 4))
for i, k in enumerate(range(num_samples)):
    du = gpl.sample(vk())
    u = jnp.cumsum(du) * dt

    plt.plot(t, du, c="blue", alpha=1 / num_samples)

    plt.plot(t, u, c="green", alpha=1 / num_samples)

    polarity = dgf_polarity(du, fs=fs)
    print(f"Polarity[sample {i}]:", polarity)

plt.title("Samples from learned surrogate prior over several periods")
plt.xlabel("time (ms)")
plt.ylabel("amplitude")
plt.show()

# %%

gpl_lf = BayesianLinearRegressor(
    kernel.compute_phi, lf["t"], mu=mu_star, cov_root=Sigma_star_root
)

conditioned = gpl_lf.condition(lf["du"], t).gp

for _ in range(num_samples):
    plt.plot(
        t, conditioned.sample(vk()), "--", label="conditioned sample", c="red"
    )
plt.plot(lf["t"], lf["du"], label="data", c="black")

plt.legend()
plt.title("Learned surrogate prior conditioned on original LF exemplar")
plt.xlabel("time (ms)")
plt.ylabel("amplitude")
plt.show()
# %%
