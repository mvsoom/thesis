import jax
import jax.numpy as jnp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.jax import nocheck, normalize_density, quantile_sample


def window(t, sigma_w):
    return jnp.exp(-0.5 * (t / sigma_w) ** 2) / jnp.sqrt(
        2 * jnp.pi * sigma_w**2
    )


def compute_mean_yw_power(X, y, freqs, sigma_w):
    w = window(X, sigma_w)

    mask = ~jnp.isnan(y)
    wy = jnp.where(mask, w * y, 0.0)
    X = jnp.where(mask, X, 0.0)

    def body(carry, f):
        phase = jnp.exp(-2j * jnp.pi * f * X)
        Y = jnp.sum(wy * phase, axis=1)

        P = jnp.abs(Y) ** 2

        mean_P = jnp.mean(P)

        return carry, mean_P

    _, out = jax.lax.scan(body, None, freqs)

    return out


def plot_average_psd_db(freqs, average_psd):
    psd_db = 10 * jnp.log10(
        jnp.maximum(average_psd / jnp.max(average_psd), 1e-20)
    )
    return px.line(
        x=freqs,
        y=psd_db,
        labels={"x": "Frequency (harmonic number)", "y": "Power (dB)"},
        title="Average Power Spectral Density",
    )


def mean_spectrum(model, keys, freqs):
    def _mean_spectrum(keys, freqs):
        def body(total, key):
            k = model(key).posterior.prior.kernel
            s = k.bochner_spectrum(freqs)
            return total + s, None

        total0 = jnp.zeros_like(freqs)
        total, _ = jax.lax.scan(body, total0, keys)
        return total / keys.shape[0]

    return nocheck(_mean_spectrum)(keys, freqs)


def plot_spectral_initialization(
    freqs,
    average_psd,
    prior_qsvi,
    S_mean,
    M,
    *,
    plot_eps=1e-12,
    log_y=True,
):
    S_prior_sample = prior_qsvi.posterior.prior.kernel.bochner_spectrum(freqs)

    D_norm = normalize_density(freqs, average_psd)
    S_prior_sample_norm = normalize_density(freqs, S_prior_sample)
    S_mean_norm = normalize_density(freqs, S_mean)

    score = S_prior_sample * average_psd
    score_norm = normalize_density(freqs, score)

    samples = quantile_sample(freqs, score_norm, M)
    score_at_samples = jnp.interp(samples, freqs, score_norm)

    if log_y:
        D_norm = jnp.maximum(D_norm, plot_eps)
        S_mean_norm = jnp.maximum(S_mean_norm, plot_eps)
        S_prior_sample_norm = jnp.maximum(S_prior_sample_norm, plot_eps)
        score_norm = jnp.maximum(score_norm, plot_eps)
        score_at_samples = jnp.maximum(score_at_samples, plot_eps)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Empirical DFT and Prior Bochner Spectra",
            "Inducing-Point Score",
        ],
    )

    fig.add_trace(
        go.Scatter(x=freqs, y=D_norm, name="Empirical DFT", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=S_mean_norm,
            name="Mean Bochner (prior)",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=S_prior_sample_norm,
            name="Sampled Bochner (prior)",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=freqs, y=score_norm, name="Score density"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=samples,
            y=score_at_samples,
            mode="markers",
            name="Sampled inducing points",
            marker=dict(size=7),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=760, title="Spectral Initialization")
    fig.update_xaxes(title="Frequency (harmonic number)", row=2, col=1)
    fig.update_yaxes(title="Normalized density", row=1, col=1)
    fig.update_yaxes(title="Score density", row=2, col=1)

    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=2, col=1)

    return fig


def plot_spectra_after_vi(
    freqs,
    average_psd,
    prior_qsvi,
    qsvi,
    *,
    plot_eps=1e-12,
    log_y=True,
):
    S_empirical = normalize_density(freqs, average_psd)
    S_prior = normalize_density(
        freqs, prior_qsvi.posterior.prior.kernel.bochner_spectrum(freqs)
    )
    S_learned = normalize_density(
        freqs, qsvi.posterior.prior.kernel.bochner_spectrum(freqs)
    )

    if log_y:
        S_empirical = jnp.maximum(S_empirical, plot_eps)
        S_prior = jnp.maximum(S_prior, plot_eps)
        S_learned = jnp.maximum(S_learned, plot_eps)

    fig = go.Figure(
        [
            go.Scatter(
                x=freqs, y=S_empirical, mode="lines", name="Empirical DFT"
            ),
            go.Scatter(
                x=freqs,
                y=S_prior,
                mode="lines",
                name="Prior Bochner (prior_qsvi)",
            ),
            go.Scatter(
                x=freqs,
                y=S_learned,
                mode="lines",
                name="Learned Bochner (qsvi)",
            ),
        ]
    )

    fig.update_layout(
        title="Spectra Before vs After VI",
        xaxis_title="Frequency (harmonic number)",
        yaxis_title="Normalized density",
        yaxis_type="log" if log_y else "linear",
    )
    return fig


def plot_elbo_history(history, title="ELBO during training (best run)"):
    return px.line(
        history,
        title=title,
        labels={"x": "Iteration", "y": "ELBO"},
    )


def plot_prior_samples(tau_test, y):
    return (
        px.line(y)
        .update_traces(x=tau_test)
        .update_layout(
            xaxis_title=r"$\tau$ (cycles)",
            yaxis_title=r"$u'(t)$",
            title="Prior samples of learned latent function distribution",
        )
    )


def plot_basis_functions(tau_test, Psi_test):
    return (
        px.line(Psi_test)
        .update_traces(x=tau_test)
        .update_layout(
            xaxis_title=r"$\tau$ (cycles)",
            yaxis_title=r"$\psi_m(t)$",
            title=r"Learned basis functions $\psi_m(t)$",
        )
    )
