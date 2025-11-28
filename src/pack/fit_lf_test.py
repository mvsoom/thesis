# %%
import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu")


import jax.numpy as jnp

from pack.spectral_factors import build_pack_factors
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
def build_amp_prior_cov(B):
    # B: (M, W)
    Kc = jnp.conj(B).T @ B  # (W, W), complex
    ReK = jnp.real(Kc)
    ImK = jnp.imag(Kc)

    top = jnp.concatenate([ReK, -ImK], axis=1)
    bot = jnp.concatenate([ImK, ReK], axis=1)
    Sigma_a = 0.5 * jnp.concatenate([top, bot], axis=0)  # (2W, 2W)
    return Sigma_a


def build_phi_cos_sin(t_grid, omega_vals):
    phase = omega_vals[None, :] * t_grid[:, None]  # (N, W)
    cos_part = jnp.cos(phase)
    sin_part = jnp.sin(phase)
    scale = 2.0 / T
    Phi = jnp.concatenate(
        [scale * cos_part, -scale * sin_part], axis=1
    )  # (N, 2W)
    return Phi


def build_phi_cos_sin_integrated(t_grid, omega_vals, t0):
    # integration is ∫_{t0}^t u'(τ) dτ, where u' uses build_phi_cos_sin
    phase = omega_vals[None, :] * t_grid[:, None]  # (N, W)
    phase0 = omega_vals[None, :] * t0  # (1, W)

    # a_k block:  (2/T) * (sin(ω t) - sin(ω t0)) / ω
    cos_block = jnp.sin(phase) - jnp.sin(phase0)

    # b_k block:  (2/T) * (cos(ω t) - cos(ω t0)) / ω
    sin_block = jnp.cos(phase) - jnp.cos(phase0)

    scale = (2.0 / T) / omega_vals[None, :]  # (1, W)
    Phi_int = jnp.concatenate(
        [scale * cos_block, scale * sin_block],
        axis=1,
    )  # (N, 2W)

    return Phi_int


# %%

# Fix hyperparams and infer amplitudes
sigma_noise = 1e-2
sigma_noise_db = 20 * np.log10(sigma_noise)
print(f"Using noise std: {sigma_noise} ({sigma_noise_db} dB)")

t1 = 0.0
t2 = T

include_dc = False

fs = 16000.0
F0 = 1000.0 / T  # Hz equivalent
num_harmonics = int(np.floor((fs / F0) / 2))

f_vals = (
    jnp.arange(
        0 if include_dc else 1,
        num_harmonics if include_dc else num_harmonics + 1,
    )
    / T
)  # cycles / ms
omega_vals = 2.0 * jnp.pi * f_vals  # rad / ms

# sample coeffs and synthesize
d = 1
m_max = 500


def infer_amplitudes(lf, plot=False):
    t = lf["t"]
    du = lf["du"]

    _, B = build_pack_factors(d, omega_vals, t1, t2, m_max, n_quad=128)

    Sigma_a = build_amp_prior_cov(B)
    Phi = build_phi_cos_sin(t, omega_vals)

    if plot:
        L = jnp.linalg.cholesky(Sigma_a + 1e-6 * jnp.eye(Sigma_a.shape[0]))
        Phi_weighted = Phi @ L  # (N, 2W)

        plt.plot(t, du, c="black", label="du (data)")
        plt.plot(t, Phi_weighted, alpha=1.0)
        plt.title(f"Basis functions weighted by prior covariance (d={d})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    # y: (N,)
    # Phi: (N, 2W)
    # Sigma_a: (2W, 2W)
    # sigma_noise: scalar

    y = jnp.asarray(du)
    sigma2 = sigma_noise**2
    N, D = Phi.shape

    # Phi * Sigma_a
    Phi_Sa = Phi @ Sigma_a  # (N, 2W)

    # S = Phi Sigma_a Phi^T + sigma2 I
    S = Phi_Sa @ Phi.T + sigma2 * jnp.eye(N)

    # Cholesky of S
    L_S = jnp.linalg.cholesky(S + 1e-9 * jnp.eye(N))

    def solve_S(rhs):
        tmp = jax.scipy.linalg.solve_triangular(L_S, rhs, lower=True)
        return jax.scipy.linalg.solve_triangular(L_S.T, tmp, lower=False)

    # ---- posterior mean of a ----
    alpha = solve_S(y)  # (N,)
    mu_a = Sigma_a @ (Phi.T @ alpha)  # (2W,)

    # posterior mean prediction on training grid
    f_post = Phi @ mu_a  # (N,)

    # S^{-1} (Phi Sigma_a)  -> shape (N, 2W)
    S_inv_Phi_Sa = solve_S(Phi_Sa)

    Sigma_post = Sigma_a - Sigma_a @ (Phi.T @ S_inv_Phi_Sa)  # (2W, 2W)

    if plot:
        # draw posterior samples of a
        num_samples = 6
        L_post = jnp.linalg.cholesky(
            Sigma_post + 1e-9 * jnp.eye(Sigma_post.shape[0])
        )
        eps = jnp.asarray(
            np.random.randn(Sigma_post.shape[0], num_samples)
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

        Phi_integrated_periods = build_phi_cos_sin_integrated(
            t, omega_vals, lf["to"]
        )
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

        Phi_periods = build_phi_cos_sin(t_periods, omega_vals)
        Phi_integrated_periods = build_phi_cos_sin_integrated(
            t_periods, omega_vals, lf["to"]
        )

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

    return {"mu": mu_a, "Sigma": Sigma_post}


Rds = np.linspace(0.3, 2.7, 10)
lfs = [generate_examplar(T, Rd, N) for Rd in Rds]

for i, lf in enumerate(lfs):
    polarity = (-1) ** i
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

# draw samples from envelope
# %%
L_star = jnp.linalg.cholesky(Sigma_star + 1e-9 * jnp.eye(Sigma_star.shape[0]))
num_samples = 1
eps = jnp.asarray(np.random.randn(Sigma_star.shape[0], num_samples))  # (2W, 6)
a_samps_star = mu_star[:, None] + L_star @ eps  # (2W, 6)

t_periods = jnp.linspace(-2 * T, 2 * T, 4 * N, endpoint=False)
Phi_periods = build_phi_cos_sin(t_periods, omega_vals)
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
