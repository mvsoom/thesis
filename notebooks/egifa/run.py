# %%
import numpy as np
import plotly.graph_objects as go
import scipy.io
import scipy.io.wavfile
from plotly.subplots import make_subplots

from egifa.data import get_voiced_meta
from egifa.evaluate import get_voiced_runs

runs = list(get_voiced_runs())


# %%
def _as_ms(sample_idx, fs):
    return 1e3 * np.asarray(sample_idx, dtype=np.float64) / float(fs)


def _load_full_recording_from_group(group):
    fs, speech = scipy.io.wavfile.read(group["wav"])
    if speech.ndim > 1:
        speech = speech[:, 0]
    speech = speech.astype(np.float64)
    peak = np.max(np.abs(speech))
    if peak > 0:
        speech /= peak

    mat = scipy.io.loadmat(group["mat"])
    gf = np.squeeze(mat["glottal_flow"]).astype(np.float64)
    return fs, speech, gf


def plot_random_run_hierarchy(runs, seed=None):
    if len(runs) == 0:
        print("No runs available.")
        return None

    rng = np.random.default_rng(seed)
    run = runs[int(rng.integers(len(runs)))]
    group = run["group"]
    frame = run["frame"]

    fs_file, speech_full, gf_full = _load_full_recording_from_group(group)
    if int(group["fs"]) != int(fs_file):
        raise ValueError(
            f"Group fs ({group['fs']}) and file fs ({fs_file}) mismatch."
        )

    t_file_ms = _as_ms(np.arange(len(speech_full)), fs_file)

    # Get all voiced groups for this file to show hierarchy:
    # file -> groups -> frame.
    groups_in_file = list(get_voiced_meta(path_contains=group["wav"]))
    group_intervals_ms = []
    for g in groups_in_file:
        g0 = float(_as_ms(g["t_samples"][0], fs_file))
        g1 = float(_as_ms(g["t_samples"][-1], fs_file))
        group_intervals_ms.append((g0, g1))

    group_start_ms = float(_as_ms(group["t_samples"][0], fs_file))
    group_end_ms = float(_as_ms(group["t_samples"][-1], fs_file))
    frame_start_ms = float(_as_ms(frame["t_samples"][0], fs_file))
    frame_end_ms = float(_as_ms(frame["t_samples"][-1], fs_file))

    group_t_ms = _as_ms(group["smooth"]["t_samples"], fs_file)
    frame_t_ms = _as_ms(frame["t_samples"], fs_file)
    frame_gci_ms = _as_ms(frame["gci"], fs_file)
    frame_goi_ms = _as_ms(frame["goi"], fs_file)

    fig = make_subplots(
        rows=8,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.02,
        subplot_titles=[
            "Entire file: speech",
            "Entire file: glottal flow",
            "Selected voiced group: smoothed speech",
            "Selected voiced group: smoothed gf",
            "Selected voiced group: smoothed dgf",
            "Selected frame: speech",
            "Selected frame: gf",
            "Selected frame: dgf",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=t_file_ms,
            y=speech_full,
            mode="lines",
            name="file speech",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_file_ms,
            y=gf_full,
            mode="lines",
            name="file gf",
            line=dict(color="#ff7f0e"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=group_t_ms,
            y=group["smooth"]["speech"],
            mode="lines",
            name="group speech (smooth)",
            line=dict(color="#1f77b4"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=group_t_ms,
            y=group["smooth"]["gf"],
            mode="lines",
            name="group gf (smooth)",
            line=dict(color="#ff7f0e"),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=group_t_ms,
            y=group["smooth"]["dgf"],
            mode="lines",
            name="group dgf (smooth)",
            line=dict(color="#2ca02c"),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frame_t_ms,
            y=frame["speech"],
            mode="lines",
            name="frame speech",
            line=dict(color="#1f77b4"),
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame_t_ms,
            y=frame["gf"],
            mode="lines",
            name="frame gf",
            line=dict(color="#ff7f0e"),
        ),
        row=7,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame_t_ms,
            y=frame["dgf"],
            mode="lines",
            name="frame dgf",
            line=dict(color="#2ca02c"),
        ),
        row=8,
        col=1,
    )

    # Group intervals over full-file context
    for g0, g1 in group_intervals_ms:
        fig.add_vrect(
            x0=g0,
            x1=g1,
            fillcolor="rgba(120,120,120,0.08)",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_vrect(
            x0=g0,
            x1=g1,
            fillcolor="rgba(120,120,120,0.08)",
            line_width=0,
            row=2,
            col=1,
        )

    # Highlight selected group on file context
    for row in (1, 2):
        fig.add_vrect(
            x0=group_start_ms,
            x1=group_end_ms,
            fillcolor="rgba(0,150,255,0.20)",
            line_width=1,
            line_color="rgba(0,150,255,0.5)",
            row=row,
            col=1,
        )

    # Highlight selected frame in file/group context
    for row in (1, 2, 3, 4, 5):
        fig.add_vrect(
            x0=frame_start_ms,
            x1=frame_end_ms,
            fillcolor="rgba(255,0,0,0.20)",
            line_width=1,
            line_color="rgba(255,0,0,0.5)",
            row=row,
            col=1,
        )

    # Frame GCIs / GOIs on detailed rows
    for x in frame_gci_ms:
        for row in (6, 7, 8):
            fig.add_vline(
                x=float(x),
                line_color="green",
                line_width=1,
                opacity=0.35,
                row=row,
                col=1,
            )
    for x in frame_goi_ms:
        for row in (6, 7, 8):
            fig.add_vline(
                x=float(x),
                line_color="purple",
                line_width=1,
                opacity=0.25,
                row=row,
                col=1,
            )

    # Match axes by hierarchy level.
    for row in (2, 3, 4, 5):
        fig.update_xaxes(matches="x", row=row, col=1)
    for row in (7, 8):
        fig.update_xaxes(matches="x6", row=row, col=1)

    fig.update_xaxes(
        title_text="absolute time (ms): file/group context",
        row=5,
        col=1,
    )
    fig.update_xaxes(
        title_text="absolute time (ms): frame detail",
        row=8,
        col=1,
        range=[frame_start_ms, frame_end_ms],
    )

    fig.update_yaxes(title_text="speech", row=1, col=1)
    fig.update_yaxes(title_text="gf", row=2, col=1)
    fig.update_yaxes(title_text="speech", row=3, col=1)
    fig.update_yaxes(title_text="gf", row=4, col=1)
    fig.update_yaxes(title_text="dgf", row=5, col=1)
    fig.update_yaxes(title_text="speech", row=6, col=1)
    fig.update_yaxes(title_text="gf", row=7, col=1)
    fig.update_yaxes(title_text="dgf", row=8, col=1)

    fig.update_layout(
        height=1800,
        hovermode="x unified",
        title=(
            "EGIFA hierarchy: file > voiced group > frame | "
            f"{group['name']} | f0={group['f0_hz']} Hz | "
            f"group={group['group']} frame={frame['frame_index']} "
            f"restart={frame['restart_index']}"
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.03,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.show()

    return run


run = plot_random_run_hierarchy(runs)

print("fs:", run["frame"]["fs"])
print("frame index:", run["frame"]["frame_index"])

print()

print("Periods (ms) in frame:", run["frame"]["periods_ms"])
print("OQ values in frame:", run["frame"]["oq"])


# %%
# Distribution summary over the current test set (runs)
# Actual frame-level pitch comes from periods_ms.

actual_f0 = []
actual_oq = []
nominal_f0 = []

for r in runs:
    periods_ms = np.asarray(r["frame"]["periods_ms"], dtype=float)
    oq = np.asarray(r["frame"]["oq"], dtype=float)

    periods_ms = periods_ms[np.isfinite(periods_ms) & (periods_ms > 0)]
    oq = oq[np.isfinite(oq)]

    actual_f0.append(
        1000.0 / np.mean(periods_ms) if len(periods_ms) else np.nan
    )
    actual_oq.append(np.mean(oq) if len(oq) else np.nan)
    nominal_f0.append(float(r["group"]["f0_hz"]))

actual_f0 = np.asarray(actual_f0, dtype=float)
actual_oq = np.asarray(actual_oq, dtype=float)
nominal_f0 = np.asarray(nominal_f0, dtype=float)

mask_f0 = np.isfinite(actual_f0)
mask_oq = np.isfinite(actual_oq)
mask_nom = np.isfinite(nominal_f0)

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    subplot_titles=[
        "Actual F0 distribution",
        "Actual OQ distribution",
    ],
)

fig.add_trace(
    go.Histogram(
        x=actual_f0[mask_f0],
        nbinsx=50,
        name="actual F0",
        opacity=0.8,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=actual_oq[mask_oq],
        nbinsx=50,
        name="actual OQ",
        opacity=0.8,
    ),
    row=1,
    col=2,
)

for f0_nom in np.unique(nominal_f0[mask_nom]):
    fig.add_vline(
        x=float(f0_nom),
        line=dict(color="crimson", width=1, dash="dot"),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="crimson", width=1, dash="dot"),
        name="nominal F0 markers",
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.update_xaxes(title_text="F0 (Hz)", row=1, col=1)
fig.update_yaxes(title_text="count", row=1, col=1)
fig.update_xaxes(title_text="OQ", row=1, col=2)
fig.update_yaxes(title_text="count", row=1, col=2)

fig.update_layout(
    height=450,
    hovermode="closest",
    title=f"EGIFA test-set frame distributions (n={len(runs)})",
)
fig.show()


# %%
# Relationship plots: OQ vs nominal F0, and OQ vs pressure

actual_oq = []
nominal_f0 = []
pressure_pa = []

for r in runs:
    oq = np.asarray(r["frame"]["oq"], dtype=float)
    oq = oq[np.isfinite(oq)]

    actual_oq.append(np.mean(oq) if len(oq) else np.nan)
    nominal_f0.append(float(r["group"]["f0_hz"]))
    pressure_pa.append(float(r["group"]["pressure_pa"]))

actual_oq = np.asarray(actual_oq, dtype=float)
nominal_f0 = np.asarray(nominal_f0, dtype=float)
pressure_pa = np.asarray(pressure_pa, dtype=float)

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    subplot_titles=[
        "Actual OQ vs nominal F0",
        "Actual OQ vs pressure_pa",
    ],
)

for f0_nom in np.sort(np.unique(nominal_f0[np.isfinite(nominal_f0)])):
    idx = (
        np.isfinite(actual_oq)
        & np.isfinite(nominal_f0)
        & (nominal_f0 == f0_nom)
    )
    if not np.any(idx):
        continue
    fig.add_trace(
        go.Box(
            x=np.full(np.sum(idx), f"{int(round(f0_nom))} Hz"),
            y=actual_oq[idx],
            name=f"{int(round(f0_nom))} Hz",
            boxpoints="outliers",
            marker=dict(size=3),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

for p in np.sort(np.unique(pressure_pa[np.isfinite(pressure_pa)])):
    idx = np.isfinite(actual_oq) & np.isfinite(pressure_pa) & (pressure_pa == p)
    if not np.any(idx):
        continue
    fig.add_trace(
        go.Box(
            x=np.full(np.sum(idx), f"{int(round(p))} Pa"),
            y=actual_oq[idx],
            name=f"{int(round(p))} Pa",
            boxpoints="outliers",
            marker=dict(size=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig.update_xaxes(title_text="nominal F0", row=1, col=1)
fig.update_yaxes(title_text="actual OQ", row=1, col=1)
fig.update_xaxes(title_text="pressure_pa", row=1, col=2)
fig.update_yaxes(title_text="actual OQ", row=1, col=2)

fig.update_layout(
    height=500,
    hovermode="closest",
    title=f"EGIFA frame-level OQ distributions (n={len(runs)})",
)
fig.show()



