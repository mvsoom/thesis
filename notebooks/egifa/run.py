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
