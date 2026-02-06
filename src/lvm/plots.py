# %%

import math

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2, spearmanr

from lvm.qgpvlm import sample_qgpvlm
from prism.svi import (
    latent_pair_density,
)


def pair_plots_oq(qlvm, pairs, showdensity, showscatter, oq=None):
    X_mu = qlvm.X_mu
    X_var = qlvm.X_var

    if oq is None:
        oq = np.full((len(X_mu),), np.nan)

    if showdensity:
        for pair in pairs:
            dens, extent = latent_pair_density(X_mu, X_var, pair)

            i, j = pair
            x_vals = np.linspace(extent[0], extent[1], dens.shape[1])
            y_vals = np.linspace(extent[2], extent[3], dens.shape[0])

            fig = px.imshow(
                np.array(dens),
                x=x_vals,
                y=y_vals,
                color_continuous_scale=px.colors.sequential.Gray,
                title=f"Latent pair density (latent {i} vs latent {j})",
                labels={
                    "x": f"latent {i}",
                    "y": f"latent {j}",
                    "color": "density",
                },
                aspect="auto",
            )

            if showscatter:
                fig.add_scatter(
                    x=np.array(X_mu[:, i]),
                    y=np.array(X_mu[:, j]),
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=np.array(oq),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="oq"),
                    ),
                    name="oq",
                )
                fig.update_traces(
                    showscale=False, selector=dict(type="heatmap")
                )
    elif showscatter:
        scatter_rows = []
        panel_order = []
        for pair in pairs:
            i, j = pair
            panel_label = f"latent {i} vs latent {j}"
            panel_order.append(panel_label)
            for x_val, y_val, color_val in zip(
                np.array(X_mu[:, i]), np.array(X_mu[:, j]), np.array(oq)
            ):
                scatter_rows.append(
                    {
                        "x": x_val,
                        "y": y_val,
                        "oq": color_val,
                        "panel": panel_label,
                    }
                )

        fig = px.scatter(
            scatter_rows,
            x="x",
            y="y",
            color="oq",
            facet_col="panel",
            facet_col_wrap=2,
            category_orders={"panel": panel_order},
            color_continuous_scale="Viridis",
            title="Latent pair projections colored by oq",
            labels={"x": "latent value", "y": "latent value", "oq": "oq"},
        )
    return fig


def single_plot_oq(qlvm, top3, oq=None):
    X_mu = qlvm.X_mu
    X_var = qlvm.X_var

    if oq is None:
        oq = np.full((len(X_mu),), np.nan)

    i, j, k = top3

    fig = px.scatter_3d(
        x=X_mu[:, i],
        y=X_mu[:, j],
        z=X_mu[:, k],
        color=oq,
        color_continuous_scale="Viridis",
        opacity=0.7,
    )

    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        scene=dict(
            xaxis_title=f"latent {i}",
            yaxis_title=f"latent {j}",
            zaxis_title=f"latent {k}",
        ),
        title="Latent space (top 3 dims) colored by oq",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return fig


# %%
def xd_gmm_plots(gmm, history, qlvm, top3):
    K = gmm.K
    params = gmm.params
    r = gmm.r

    yield px.line(
        y=np.array(history),
        title=f"[K={K}] Log likelihood during EM",
        labels={"x": "Iteration", "y": "Log likelihood"},
    )

    # plot cumulative histogram of background responsibilities
    r_bg = r[:, 0]
    yield px.histogram(
        x=np.array(r_bg),
        nbins=100,
        cumulative=True,
        histnorm="probability",
        title=f"[K={K}] Cumulative histogram of background responsibilities",
        labels={
            "x": "Responsibility of background component",
            "y": "Cumulative density",
        },
    )

    is_outlier = r_bg > 0.95

    labels = np.argmax(
        r[:, 1:], axis=1
    )  # cluster index 0..K-1 (ignoring background)

    labels = np.array(labels)

    yield px.bar(
        x=np.arange(r.shape[1]),
        y=np.array(r.sum(axis=0) / r.sum()),
        title=f"[K={K}] Component responsibility mass",
        labels={"x": "Component index", "y": "Responsibility mass"},
    )

    X_mu = qlvm.X_mu

    try:
        i, j, k = top3  # will fail if Q < 3

        # Colors: clusters, but gray outliers
        plot_color = labels.astype(str)
        plot_color = np.where(is_outlier, "outlier", plot_color)

        fig = px.scatter_3d(
            x=X_mu[:, i],
            y=X_mu[:, j],
            z=X_mu[:, k],
            color=plot_color,
            opacity=0.7,
        )

        # Make outliers visually obvious
        fig.update_traces(
            selector=dict(name="outlier"),
            marker=dict(size=4, color="black", symbol="x"),
        )

        fig.update_traces(marker=dict(size=3))

        fig.update_layout(
            scene=dict(
                xaxis_title=f"latent {i}",
                yaxis_title=f"latent {j}",
                zaxis_title=f"latent {k}",
            ),
            title=f"[K={K}] Latent space clusters with outliers",
            margin=dict(l=0, r=0, b=0, t=30),
        )

        yield fig

        # Use the same top3 dims
        dims = (i, j, k)

        # 99% chi-square radius in 3D
        radius = np.sqrt(chi2.ppf(0.95, 3))

        # Color map reused from clusters
        unique_labels = np.unique(labels)
        palette = px.colors.qualitative.Dark24
        color_map = {str(l): palette[l % len(palette)] for l in unique_labels}

        fig = go.Figure()

        # Optional: faint points for context
        fig.add_trace(
            go.Scatter3d(
                x=X_mu[:, i],
                y=X_mu[:, j],
                z=X_mu[:, k],
                mode="markers",
                marker=dict(
                    size=2,
                    color=[color_map[str(l)] for l in labels],
                    opacity=0.12,
                ),
                showlegend=False,
            )
        )

        # Parametric unit sphere grid
        Nu, Nv = 50, 25
        u = np.linspace(0, 2 * np.pi, Nu)
        v = np.linspace(0, np.pi, Nv)
        uu, vv = np.meshgrid(u, v)

        # Unit sphere (3, Nv, Nu)
        sphere = np.stack(
            [
                np.cos(uu) * np.sin(vv),
                np.sin(uu) * np.sin(vv),
                np.cos(vv),
            ],
            axis=0,
        )

        # Plot each cluster ellipsoid (skip background)
        for k_idx in range(params.mu.shape[0]):
            mu = np.array(params.mu)[k_idx][list(dims)]
            cov = np.array(params.cov)[k_idx][np.ix_(dims, dims)]

            # Eigen-decomposition
            w, V = np.linalg.eigh(cov)

            # Transform unit sphere -> ellipsoid
            A = V @ np.diag(np.sqrt(w)) * radius

            # Apply transform, keeping grid
            ell = mu[:, None, None] + np.einsum("ij,jmn->imn", A, sphere)

            X = ell[0]
            Y = ell[1]
            Z = ell[2]

            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    opacity=0.25,
                    showscale=False,
                    surfacecolor=np.zeros_like(X),
                    colorscale=[
                        [0, color_map[str(k_idx)]],
                        [1, color_map[str(k_idx)]],
                    ],
                    name=f"cluster {k_idx}",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title=f"latent {i}",
                yaxis_title=f"latent {j}",
                zaxis_title=f"latent {k}",
            ),
            title=f"[K={K}] Cluster ellipsoids in latent space",
            margin=dict(l=0, r=0, b=0, t=30),
        )

        yield fig
    except Exception as e:
        print(f"Could not make 3D cluster plot: {e}")


# %%
def sample_latent_gmm_pointwise(gmm, plvm, psi, tau_test, unwhiten, nsamples=6):
    K = gmm.K
    fig = go.Figure()

    # fixed basis once
    Psi = jax.vmap(psi)(tau_test)

    for _ in range(nsamples):
        # sample component and latent
        k = np.random.choice(K, p=gmm.pi)
        mu_k = gmm.params.mu[k]
        cov_k = gmm.params.cov[k]

        z = np.random.multivariate_normal(mu_k, cov_k)

        mu_y, diag_y = plvm.predict_f_meanvar_batch(z, z * 0)
        Sigma_y = jax.vmap(jnp.diag)(diag_y[0])

        mu_eps_sample, _ = unwhiten(mu_y, Sigma_y)

        f_sample = Psi @ mu_eps_sample.squeeze()

        fig.add_trace(
            go.Scattergl(
                x=np.array(tau_test),
                y=f_sample,
                mode="lines",
                line=dict(width=1),
                name=f"cluster {k}",
                legendgroup=str(k),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"[K={K}] {nsamples} samples from latent GMM",
        xaxis_title="tau",
        yaxis_title="u'(t)",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return fig


def plot_cluster_means_in_data_space(qgp, tau_test):
    K = qgp.gmm.K

    Psi = jax.vmap(qgp.psi)(tau_test)  # [T, M]

    # sort components by mixture weight
    order = jnp.argsort(-qgp.pi)
    pis = qgp.pi[order]
    mus = qgp.mu[order]
    covs = qgp.cov[order]

    colors = pc.qualitative.Set2
    fig = go.Figure()

    for i, (pi_k, mu_k, Sigma_k) in enumerate(zip(pis, mus, covs)):
        color = colors[i % len(colors)]
        group = f"comp{i}"

        mean = Psi @ mu_k
        var = jnp.einsum("tm,mn,tn->t", Psi, Sigma_k, Psi)
        std = jnp.sqrt(jnp.maximum(var, 0.0))

        upper = mean + 2 * std
        lower = mean - 2 * std

        label = f"k={i}, π={pi_k:.2f}"

        # upper envelope (anchor for fill)
        fig.add_trace(
            go.Scattergl(
                x=tau_test,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                legendgroup=group,
                hoverinfo="skip",
            )
        )

        # lower envelope + fill
        fig.add_trace(
            go.Scattergl(
                x=tau_test,
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.25)"),
                line=dict(width=0),
                showlegend=False,
                legendgroup=group,
                hoverinfo="skip",
            )
        )

        # mean line (only legend entry)
        fig.add_trace(
            go.Scattergl(
                x=tau_test,
                y=mean,
                mode="lines",
                line=dict(color=color, width=3),
                name=label,
                legendgroup=group,
            )
        )

    fig.update_layout(
        xaxis_title="tau",
        yaxis_title="u'(tau)",
        title=f"[K={K}] Learned means ± 2σ of qGPLVM components in data space",
        legend_title="Components (sorted by π)",
        legend=dict(
            groupclick="togglegroup"  # THIS is the crucial line
        ),
    )

    return fig

def plot_cluster_samples_in_data_space(key, gqp, tau_test, nsamples=6):
    K = gqp.gmm.K

    samples = sample_qgpvlm(key, gqp, tau_test, nsamples)  # (nsamples, Ntest)

    ncols = math.ceil(math.sqrt(nsamples))
    nrows = math.ceil(nsamples / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    x = np.array(tau_test)

    for i in range(nsamples):
        f_sample = np.array(samples[i])

        r = i // ncols + 1
        c = i % ncols + 1

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=f_sample,
                mode="lines",
                line=dict(width=1),
                showlegend=False,
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        title=f"[K={K}] Samples from qGPLVM",
        xaxis_title="tau",
        yaxis_title="u'(t)",
        margin=dict(l=0, r=0, b=0, t=30),
        height=250 * nrows,
        width=300 * ncols,
    )

    return fig


def plot_logl_histogram(log_prob_gmm, log_prob_u, n_eff, K):
    gmm = (log_prob_gmm / n_eff).ravel()
    lf = (log_prob_u / n_eff).ravel()

    df = pd.DataFrame(
        {
            "loglik": np.concatenate([gmm, lf]),
            "model": (["GMM"] * gmm.size) + (["LF"] * lf.size),
        }
    )

    return px.histogram(
        df,
        x="loglik",
        color="model",
        nbins=100,
        barmode="overlay",
        opacity=0.5,
        title=f"[K={K}] Normalized log likelihoods per sample",
        labels={"loglik": "Log likelihood per sample", "count": "Count"},
    )


def plot_oq_vs_loglik(oq, loglik, title=None):
    oq = np.asarray(oq).ravel()
    loglik = np.asarray(loglik).ravel()

    if oq.size != loglik.size:
        raise ValueError("oq and loglik must have the same number of points")

    if title is None:
        title = "Open quotient vs log likelihood"

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=oq,
                y=loglik,
                mode="markers",
                marker=dict(size=4, opacity=0.6),
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="open quotient (oq)",
        yaxis_title="normalized log likelihood",
        margin=dict(l=0, r=0, b=0, t=30),
        height=450,
        width=600,
    )
    fig.update_xaxes(type="log")
    return fig


def oq_sensitivity_spearman(oq, ll):
    oq = np.asarray(oq)
    ll = np.asarray(ll)
    rho, p = spearmanr(oq, ll)
    return {"oq_sensitivity": float(rho), "oq_sensitivity_p": float(p)}
