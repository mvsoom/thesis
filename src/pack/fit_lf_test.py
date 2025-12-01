# %%
import jax
import matplotlib.pyplot as plt
import numpy as np

from gp.mercer import posterior_latent
from gp.periodic import SPACK

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

f_vals = (
    jnp.arange(
        1,
        num_harmonics + 1,
    )
    / T
)  # cycles / ms
omega_vals = 2.0 * jnp.pi * f_vals  # rad / ms

d = 1

kernel = SPACK(d, T, num_harmonics, t1, t2)

# %%
def infer_amplitudes(lf, plot=False):
    t = lf["t"]
    du = lf["du"]

    Phi = jax.vmap(kernel.compute_phi)(t)
    L = kernel.compute_weights_root()

    if plot:
        Phi_weighted = Phi @ L  # (N, 2W)

        plt.plot(t, du, c="black", label="du (data)")
        plt.plot(t, Phi_weighted, alpha=1.0)
        plt.title(f"Basis functions weighted by prior covariance (d={d})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    mu_a, Sigma_a = posterior_latent(du, kernel, t, sigma_noise**2)

    mu_a = L @ mu_a
    Sigma_a = L @ Sigma_a @ L.T

    f_post = Phi @ mu_a  # (N,)

    if plot:
        # draw posterior samples of a
        num_samples = 6
        L_post = jnp.linalg.cholesky(Sigma_a + 1e-9 * jnp.eye(Sigma_a.shape[0]))
        eps = jnp.asarray(
            np.random.randn(Sigma_a.shape[0], num_samples)
        )  # (2W, 6)
        a_samps = mu_a[:, None] + L_post @ eps  # (2W, 6)

        # turn amplitudes into function samples
        f_samps = Phi @ a_samps  # (N, 6)

        plt.figure(figsize=(6, 4))
        plt.plot(t, du, label="data", c="black", linewidth=2)
        plt.plot(t, f_post, label="posterior mean", c="red", linewidth=2)

        for k in range(num_samples):
            u = jnp.cumsum(f_samps[:, k]) * (t[1] - t[0])
            plt.plot(t, f_samps[:, k], c="blue", alpha=0.1)
            plt.plot(t, u, c="green", alpha=0.1)

        Phi_integrated_periods = jax.vmap(
            lambda t: kernel.compute_phi_integrated(t, lf["to"])
        )(t)
        plt.plot(
            t,
            Phi_integrated_periods @ mu_a,
            label="posterior mean (integrated)",
            c="orange",
            linewidth=2,
        )

        plt.legend()
        plt.title(f"Posterior mean and samples (Rd = {lf['Rd']})")
        plt.xlabel("time (ms)")
        plt.ylabel("amplitude")
        plt.show()

        # plot samples over several periods
        t_periods = jnp.linspace(-2 * T, 2 * T, 4 * N, endpoint=False)

        Phi_periods = jax.vmap(kernel.compute_phi)(t_periods)
        Phi_integrated_periods = jax.vmap(
            lambda t: kernel.compute_phi_integrated(t, lf["to"])
        )(t_periods)

        f_samps_periods = Phi_periods @ a_samps  # (4N, 6)
        f_samps_integrated_periods = Phi_integrated_periods @ a_samps  # (4N, 6)

        plt.figure(figsize=(6, 4))
        for k in range(num_samples):
            plt.plot(t_periods, f_samps_periods[:, k], c="blue", alpha=0.1)
            plt.plot(
                t_periods,
                f_samps_integrated_periods[:, k],
                c="green",
                alpha=0.1,
            )

        plt.title(f"Posterior samples over several periods (Rd = {lf['Rd']})")
        plt.xlabel("time (ms)")
        plt.ylabel("amplitude")
        plt.show()

    return {"mu": mu_a, "Sigma": Sigma_a}

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
    # posteriors: list of dicts {'mu': mu_i, 'Sigma': Sigma_i}
    mus = [p["mu"] for p in posteriors]
    Sigmas = [p["Sigma"] for p in posteriors]

    k = len(mus)
    D = mus[0].shape[0]

    # mixture mean
    mu_star = sum(mus) / k

    # mixture second moment
    second = sum(Sigmas[i] + jnp.outer(mus[i], mus[i]) for i in range(k)) / k

    # covariance = second moment - outer(m, m)
    Sigma_star = second - jnp.outer(mu_star, mu_star)

    return mu_star, Sigma_star


mu_star, Sigma_star = envelope_gaussians(posteriors)

plt.plot(mu_star, label="Mixture mean")
for i, p in enumerate(posteriors):
    plt.plot(p["mu"], alpha=0.3, label=f"Posterior mean Rd={Rds[i]}")
plt.title("Mixture mean and individual posterior means")
# plt.legend()
plt.show()

# %%

# draw samples from envelope
# %%
L_star = jnp.linalg.cholesky(Sigma_star + 1e-9 * jnp.eye(Sigma_star.shape[0]))
num_samples = 5
eps = jnp.asarray(np.random.randn(Sigma_star.shape[0], num_samples))  # (2W, 6)
a_samps_star = mu_star[:, None] + L_star @ eps  # (2W, 6)

t_periods = jnp.linspace(-2 * T, 2 * T, 4 * N, endpoint=False)
Phi_periods = jax.vmap(kernel.compute_phi)(t_periods)  # (4N, 2W)
f_samps_periods_star = Phi_periods @ a_samps_star  # (4N, 6)

plt.figure(figsize=(6, 4))
for k in range(num_samples):
    plt.plot(
        t_periods, f_samps_periods_star[:, k], c="blue", alpha=1 / num_samples
    )

    u = jnp.cumsum(f_samps_periods_star[:, k]) * (t_periods[1] - t_periods[0])

    plt.plot(t_periods, u, c="green", alpha=1 / num_samples)

plt.title("Envelope posterior samples over several periods")
plt.xlabel("time (ms)")
plt.ylabel("amplitude")
plt.show()

from utils.reskew import dgf_polarity

polarity = dgf_polarity(f_samps_periods_star[:, 0], fs=fs)
print("Polarity: ", polarity)

# %%
# `a_samps_star` is in a way a new GP with discrete indices...

# top panel
plt.subplot(2, 1, 1)

plt.plot(a_samps_star)
plt.xlabel("Coefficient index")
plt.ylabel("Amplitude")

# bottom panel
plt.subplot(2, 1, 2)

plt.plot(np.abs(a_samps_star))
plt.yscale("log")
plt.xlabel("Coefficient index")
plt.ylabel("abs(Amplitude)")
plt.show()
# %%
