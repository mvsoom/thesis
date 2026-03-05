# %%
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ack.pack import PACK
from prism.harmonic import SHMPeriodic, SHMPeriodicFFT

# %%
# Configuration
PERIOD = 1.0
VARIANCE = 1.0
MAX_HARMONIC = 80
H_VIEW = 40


def cosine_coeffs_from_line_masses(A):
    """Convert SHM line masses A to cosine-series coefficients a_j."""
    A = np.asarray(A, dtype=float)
    a = np.zeros_like(A)
    a[0] = A[0] / (2.0 * np.pi)
    a[1:] = A[1:] / np.pi
    return np.clip(a, 0.0, None)


def normalize_mass(a):
    a = np.asarray(a, dtype=float)
    total = np.sum(a)
    return a / (total + 1e-15)


def harmonic_index(mu, period=PERIOD):
    """Angular frequency mu -> harmonic index j."""
    return np.asarray(np.rint(mu / (2.0 * np.pi / period)), dtype=int)


def stem_points(x, y):
    """Build line-segment points for a stem plot."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs = np.empty(3 * x.size, dtype=float)
    ys = np.empty(3 * y.size, dtype=float)
    xs[0::3] = x
    xs[1::3] = x
    xs[2::3] = np.nan
    ys[0::3] = 0.0
    ys[1::3] = y
    ys[2::3] = np.nan
    return xs, ys


def add_stem_trace(fig, x, y, *, name, color, row=None, col=None, dash="solid"):
    xs, ys = stem_points(x, y)
    line_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=1.4, dash=dash),
        showlegend=False,
        hoverinfo="skip",
    )
    marker_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color=color, size=6),
        name=name,
        hovertemplate="j=%{x}<br>mass=%{y:.3e}<extra></extra>",
    )

    if row is None or col is None:
        fig.add_trace(line_trace)
        fig.add_trace(marker_trace)
    else:
        fig.add_trace(line_trace, row=row, col=col)
        fig.add_trace(marker_trace, row=row, col=col)


def pack_weights(J, mode):
    if mode == "uniform":
        w = np.ones(J + 1)
    elif mode == "dc-heavy":
        w = np.full(J + 1, 0.15 / max(J, 1))
        w[0] = 0.85
    elif mode == "low-harm-heavy":
        w = np.zeros(J + 1)
        w[0] = 0.15
        w[1] = 0.75
        if J > 1:
            w[2:] = 0.10 / (J - 1)
    elif mode == "high-harm-heavy":
        w = np.zeros(J + 1)
        w[0] = 0.15
        w[-1] = 0.75
        if J > 1:
            w[1:-1] = 0.10 / (J - 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return w / np.sum(w)


def pack_line_spectrum(
    *,
    d,
    J,
    weights,
    num_harmonics=MAX_HARMONIC,
    period=PERIOD,
    variance=VARIANCE,
):
    kernel = PACK(
        d=d,
        J=J,
        period=period,
        variance=variance,
        weights=jnp.asarray(weights),
    )
    shm = SHMPeriodicFFT(
        kernel=kernel,
        num_harmonics=num_harmonics,
        N_min=4096,
        oversamp=8,
        clip_negative=True,
    )
    A, mu = shm.compute_shm()
    a = cosine_coeffs_from_line_masses(A)
    p = normalize_mass(a)
    h = harmonic_index(mu, period=period)
    return h, a, p


def periodic_line_spectrum(
    *,
    lengthscale,
    num_harmonics=MAX_HARMONIC,
    period=PERIOD,
    variance=VARIANCE,
):
    kernel = SHMPeriodic(
        variance=variance,
        lengthscale=lengthscale,
        period=period,
        num_harmonics=num_harmonics,
    )
    A, mu = kernel.compute_shm()
    a = cosine_coeffs_from_line_masses(A)
    p = normalize_mass(a)
    h = harmonic_index(mu, period=period)
    return h, a, p


def tail_stats(p):
    p = np.asarray(p, dtype=float)
    cdf = np.cumsum(p)
    j95 = int(np.searchsorted(cdf, 0.95, side="left"))
    j99 = int(np.searchsorted(cdf, 0.99, side="left"))
    return {
        "tail_j_ge_2": float(np.sum(p[2:])),
        "tail_j_ge_5": float(np.sum(p[5:])),
        "j95": j95,
        "j99": j99,
    }


colors = {
    "d0": "#1f77b4",
    "d1": "#ff7f0e",
    "d2": "#2ca02c",
    "d3": "#d62728",
    "periodic_a": "#111111",
    "periodic_b": "#7f7f7f",
    "uniform": "#1f77b4",
    "dc-heavy": "#d62728",
    "low-harm-heavy": "#2ca02c",
    "high-harm-heavy": "#9467bd",
}


# %%
# 1) Compare across PACK order d, with Periodic baseline
J_fixed = 4
w_uniform = pack_weights(J_fixed, "uniform")
d_values = [0, 1, 2, 3]

fig_d = go.Figure()

for d in d_values:
    h, _, p = pack_line_spectrum(d=d, J=J_fixed, weights=w_uniform)
    mask = h <= H_VIEW
    add_stem_trace(
        fig_d,
        h[mask],
        p[mask],
        name=f"PACK d={d}, J={J_fixed}",
        color=colors[f"d{d}"],
    )

for ell, name, dash, color in [
    (0.12, "Periodic ls=0.12", "dot", colors["periodic_a"]),
    (0.40, "Periodic ls=0.40", "dash", colors["periodic_b"]),
]:
    h, _, p = periodic_line_spectrum(lengthscale=ell)
    mask = h <= H_VIEW
    add_stem_trace(
        fig_d,
        h[mask],
        p[mask],
        name=name,
        color=color,
        dash=dash,
    )

fig_d.update_layout(
    title="Line Spectrum: PACK across d (uniform weights) + Periodic baselines",
    xaxis_title="Harmonic index j",
    yaxis_title="Normalized line mass",
    yaxis_type="log",
    height=560,
)
fig_d.show()


# %%
# 2) Compare influence of J for d=0 and d=1
J_values = [1, 2, 4, 8]
d_compare = [0, 1]

fig_J = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[f"PACK d={d}" for d in d_compare],
    shared_yaxes=True,
)

palette_J = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for col, d in enumerate(d_compare, start=1):
    for J, color in zip(J_values, palette_J, strict=True):
        w = pack_weights(J, "uniform")
        h, _, p = pack_line_spectrum(d=d, J=J, weights=w)
        mask = h <= H_VIEW
        add_stem_trace(
            fig_J,
            h[mask],
            p[mask],
            name=f"J={J}",
            color=color,
            row=1,
            col=col,
        )

    h_per, _, p_per = periodic_line_spectrum(lengthscale=0.20)
    mask = h_per <= H_VIEW
    add_stem_trace(
        fig_J,
        h_per[mask],
        p_per[mask],
        name="Periodic ls=0.20",
        color=colors["periodic_a"],
        dash="dot",
        row=1,
        col=col,
    )

fig_J.update_xaxes(title_text="Harmonic index j")
fig_J.update_yaxes(title_text="Normalized line mass", type="log")
fig_J.update_layout(
    title="Influence of J on PACK line spectrum",
    height=540,
)
fig_J.show()


# %%
# 3) Influence of simplex weights for fixed J
J_weight = 4
weight_modes = ["uniform", "dc-heavy", "low-harm-heavy", "high-harm-heavy"]

fig_w = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["PACK d=0", "PACK d=1"],
    shared_yaxes=True,
)

for col, d in enumerate([0, 1], start=1):
    for mode in weight_modes:
        w = pack_weights(J_weight, mode)
        h, _, p = pack_line_spectrum(d=d, J=J_weight, weights=w)
        mask = h <= H_VIEW
        add_stem_trace(
            fig_w,
            h[mask],
            p[mask],
            name=f"{mode} ({np.array2string(w, precision=2)})",
            color=colors[mode],
            row=1,
            col=col,
        )

fig_w.update_xaxes(title_text="Harmonic index j")
fig_w.update_yaxes(title_text="Normalized line mass", type="log")
fig_w.update_layout(
    title="Influence of weights (J=4)",
    height=580,
)
fig_w.show()


# %%
# 4) J=1 tail analysis for d=0 and d=1
J_tail = 1
w0_grid = [0.10, 0.30, 0.50, 0.70, 0.90]

print("J=1 tail summary (mass in higher harmonics created by nonlinearity):")
print("columns: d, w0, w1, tail(j>=2), tail(j>=5), j95, j99")

rows = []
for d in [0, 1]:
    for w0 in w0_grid:
        w = np.array([w0, 1.0 - w0])
        h, _, p = pack_line_spectrum(d=d, J=J_tail, weights=w)
        stats = tail_stats(p)
        row = {
            "d": d,
            "w0": w0,
            "w1": 1.0 - w0,
            **stats,
        }
        rows.append(row)
        print(
            f"d={d}, w0={w0:.2f}, w1={1.0 - w0:.2f}, "
            f"tail(j>=2)={stats['tail_j_ge_2']:.3f}, "
            f"tail(j>=5)={stats['tail_j_ge_5']:.3f}, "
            f"j95={stats['j95']}, j99={stats['j99']}"
        )

fig_tail = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["J=1, d=0", "J=1, d=1"],
    shared_yaxes=True,
)

for col, d in enumerate([0, 1], start=1):
    for w0, color in zip(
        [0.10, 0.50, 0.90], ["#1f77b4", "#ff7f0e", "#d62728"], strict=True
    ):
        w = np.array([w0, 1.0 - w0])
        h, _, p = pack_line_spectrum(d=d, J=1, weights=w)
        mask = h <= H_VIEW
        add_stem_trace(
            fig_tail,
            h[mask],
            p[mask],
            name=f"w0={w0:.2f}, w1={1.0 - w0:.2f}",
            color=color,
            row=1,
            col=col,
        )

fig_tail.update_xaxes(title_text="Harmonic index j")
fig_tail.update_yaxes(title_text="Normalized line mass", type="log")
fig_tail.update_layout(
    title="J=1 tails for d=0 and d=1",
    height=540,
)
fig_tail.show()
