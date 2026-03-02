# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io.wavfile
import scipy.signal
from plotly.subplots import make_subplots

from commonvoice.data import get_voiced_meta
from utils import time_this
from utils.audio import frame_signal


# %%
def _as_ms(sample_idx, fs):
    return 1e3 * np.asarray(sample_idx, dtype=np.float64) / float(fs)


def _load_wav_resampled(path):
    fs, x = scipy.io.wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak

    if fs != 20_000:
        x = scipy.signal.resample_poly(x, 20_000, fs)
        fs = 20_000

    return fs, x


def _paired_gf_path(speech_wav_path):
    speech_wav_path = Path(speech_wav_path)
    if speech_wav_path.name.endswith("_speech.wav"):
        return speech_wav_path.with_name(
            speech_wav_path.name.replace("_speech.wav", "_gf.wav")
        )

    raise ValueError(f"Cannot infer paired *_gf.wav path for {speech_wav_path}")


def _load_full_recording_from_group(group):
    speech_path = Path(group["wav"])
    gf_path = _paired_gf_path(speech_path)

    fs_speech, speech = _load_wav_resampled(speech_path)
    fs_gf, gf = _load_wav_resampled(gf_path)

    if fs_speech != fs_gf:
        raise ValueError(
            f"Sampling rate mismatch for {speech_path} and {gf_path}: "
            f"{fs_speech} != {fs_gf}"
        )

    return fs_speech, speech, gf


def get_voiced_runs(
    path_contains=None,
    frame_len_msec=128.0,
    hop_msec=64.0,
    num_vi_restarts=1,
    dtype=np.float64,
    max_groups=None,
):
    for group_index, v in enumerate(
        get_voiced_meta(path_contains=path_contains)
    ):
        if max_groups is not None and group_index >= max_groups:
            break

        t = v["smooth"]["t_samples"].astype(dtype)
        x = v["smooth"]["speech"].astype(dtype)
        u = v["smooth"]["gf"].astype(dtype)
        du = v["smooth"]["dgf"].astype(dtype)
        tau = v["smooth"]["tau"].astype(dtype)
        assert len(x) == len(u) == len(du) == len(t) == len(tau)

        fs = float(v["fs"])
        fs_smooth = float(v["smooth"]["fs"])
        frame_len = int(frame_len_msec / 1000 * fs_smooth)
        hop = int(hop_msec / 1000 * fs_smooth)

        if len(t) < frame_len:
            continue

        t_frames = frame_signal(t, frame_len, hop)
        x_frames = frame_signal(x, frame_len, hop)
        u_frames = frame_signal(u, frame_len, hop)
        du_frames = frame_signal(du, frame_len, hop)
        tau_frames = frame_signal(tau, frame_len, hop)

        for frame_index, (
            t_frame,
            x_frame,
            u_frame,
            du_frame,
            tau_frame,
        ) in enumerate(
            zip(t_frames, x_frames, u_frames, du_frames, tau_frames)
        ):
            t_ms = 1e3 * t_frame / fs

            t_min, t_max = t_frame[0], t_frame[-1]
            loc = np.where((t_min <= v["gci"]) & (v["gci"] <= t_max))[0]

            gci = v["gci"][loc]
            goi = v["goi"][loc]
            oq = v["oq"][loc[:-1]]
            periods_ms = v["periods_ms"][loc[:-1]]

            for restart_index in range(num_vi_restarts):
                frame = {
                    "fs": fs_smooth,
                    "t_ms": t_ms,
                    "t_samples": t_frame,
                    "tau": tau_frame,
                    "speech": x_frame,
                    "gf": u_frame,
                    "dgf": du_frame,
                    "gci": gci,
                    "goi": goi,
                    "oq": oq,
                    "periods_ms": periods_ms,
                    "frame_index": frame_index,
                    "restart_index": restart_index,
                }
                yield {"group": v, "frame": frame}


PATH_CONTAINS = None
MAX_GROUPS = None  # set an int for faster iteration while exploring

with time_this():
    runs = list(
        get_voiced_runs(
            path_contains=PATH_CONTAINS,
            max_groups=MAX_GROUPS,
        )
    )


# %%
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
            "CommonVoice hierarchy: file > voiced group > frame | "
            f"{group['name']} | group={group['group']} "
            f"frame={frame['frame_index']} restart={frame['restart_index']}"
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

if run is not None:
    print("fs:", run["frame"]["fs"])
    print("frame index:", run["frame"]["frame_index"])

    print()

    print("Periods (ms) in frame:", run["frame"]["periods_ms"])
    print("OQ values in frame:", run["frame"]["oq"])


# %%
# Distribution summary over the current run set.
# Actual frame-level pitch comes from periods_ms.

actual_f0 = []
actual_oq = []

for r in runs:
    periods_ms = np.asarray(r["frame"]["periods_ms"], dtype=float)
    oq = np.asarray(r["frame"]["oq"], dtype=float)

    periods_ms = periods_ms[np.isfinite(periods_ms) & (periods_ms > 0)]
    oq = oq[np.isfinite(oq)]

    actual_f0.append(
        1000.0 / np.mean(periods_ms) if len(periods_ms) else np.nan
    )
    actual_oq.append(np.mean(oq) if len(oq) else np.nan)

actual_f0 = np.asarray(actual_f0, dtype=float)
actual_oq = np.asarray(actual_oq, dtype=float)

mask_f0 = np.isfinite(actual_f0)
mask_oq = np.isfinite(actual_oq)

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    subplot_titles=[
        "Estimated F0 distribution from LBGID",
        "Estimated OQ distribution from LBGID",
    ],
)

fig.add_trace(
    go.Histogram(
        x=actual_f0[mask_f0],
        nbinsx=50,
        name="F0",
        opacity=0.8,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=actual_oq[mask_oq],
        nbinsx=50,
        name="OQ",
        opacity=0.8,
    ),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="F0 (Hz)", row=1, col=1)
fig.update_yaxes(title_text="count", row=1, col=1)
fig.update_xaxes(title_text="OQ", row=1, col=2)
fig.update_yaxes(title_text="count", row=1, col=2)

fig.update_layout(
    height=450,
    hovermode="closest",
    title=f"CommonVoice run-level frame distributions (n={len(runs)})",
)
fig.show()
