# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io.wavfile
from plotly.subplots import make_subplots

from egifa.data import get_voiced_meta
from gci.estimate import gci_estimates_from_quickgci


def _load_wav_normalized(path):
    fs, speech = scipy.io.wavfile.read(path)
    if speech.ndim > 1:
        speech = speech[:, 0]

    speech = speech.astype(np.float32)
    peak = np.max(np.abs(speech))
    if peak > 0:
        speech /= peak

    return fs, speech


def _full_quickgci(wav_path, fs, cache):
    wav_path = Path(wav_path)
    if wav_path not in cache:
        fs_wav, speech_wav = _load_wav_normalized(wav_path)
        if fs_wav != fs:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: {fs_wav} != {fs}"
            )
        cache[wav_path] = np.asarray(
            gci_estimates_from_quickgci(speech_wav, fs=fs_wav), dtype=np.int64
        )
    return cache[wav_path]


def sample_random_window(voiced_meta, window_ms=32.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if len(voiced_meta) == 0:
        raise ValueError("No voiced metadata available.")

    for _ in range(512):
        v = voiced_meta[int(rng.integers(len(voiced_meta)))]
        fs = int(v["fs"])
        frame_len = max(2, int(round(window_ms * fs / 1000.0)))
        n = len(v["speech"])
        if n >= frame_len:
            start_local = int(rng.integers(0, n - frame_len + 1))
            stop_local = start_local + frame_len
            return v, start_local, stop_local

    raise ValueError(
        "Could not sample a voiced segment long enough for window."
    )


def plot_random_selected_window(
    voiced_meta,
    window_ms=32.0,
    quickgci_cache=None,
):
    if quickgci_cache is None:
        quickgci_cache = {}

    v, start_local, stop_local = sample_random_window(
        voiced_meta, window_ms=window_ms
    )

    fs = int(v["fs"])
    t_samples = np.asarray(v["t_samples"], dtype=np.int64)
    t_win = (1e3 * t_samples[start_local:stop_local] / fs).astype(np.float32)

    speech_win = np.asarray(
        v["speech"][start_local:stop_local], dtype=np.float32
    )
    gf_win = np.asarray(v["gf"][start_local:stop_local], dtype=np.float32)
    tau_win = np.asarray(v["tau"][start_local:stop_local], dtype=np.float32)

    start_abs = int(t_samples[start_local])
    stop_abs = int(t_samples[stop_local - 1] + 1)

    ref_abs = np.asarray(v["gci"], dtype=np.int64)
    ref_abs = ref_abs[(ref_abs >= start_abs) & (ref_abs < stop_abs)]

    est_full = _full_quickgci(v["wav"], fs, quickgci_cache)
    est_abs = est_full[(est_full >= start_abs) & (est_full < stop_abs)]

    ref_t_ms = (1e3 * ref_abs / fs).astype(np.float32)
    est_t_ms = (1e3 * est_abs / fs).astype(np.float32)
    seg0 = int(t_samples[0])
    ref_tau = (
        tau_win[(ref_abs - seg0) - start_local]
        if len(ref_abs)
        else np.asarray([], dtype=np.float32)
    )
    est_tau = (
        tau_win[(est_abs - seg0) - start_local]
        if len(est_abs)
        else np.asarray([], dtype=np.float32)
    )

    oq = np.mean(v["oq"]) if "oq" in v else np.nan

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            (
                f"EGIFA: {Path(v['wav']).stem} | "
                f"group={v['group']} | oq ≈ {oq:.2f}"
            ),
            "",
        ],
    )

    fig.add_trace(
        go.Scattergl(
            x=t_win,
            y=speech_win,
            mode="lines",
            opacity=0.7,
            name="speech",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=tau_win,
            y=gf_win,
            mode="lines",
            opacity=0.7,
            name="u(t)",
            customdata=t_win,
            hovertemplate="tau=%{x:.4f}<br>u=%{y:.4f}<br>t=%{customdata:.3f} ms<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=ref_tau,
            y=np.zeros_like(ref_tau),
            mode="markers",
            marker=dict(color="green", size=6),
            name="reference GCI",
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=est_tau,
            y=np.zeros_like(est_tau),
            mode="markers",
            marker=dict(color="red", size=6),
            name="estimated GCI",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    shapes = []
    for t in ref_t_ms:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y domain",
                x0=float(t),
                x1=float(t),
                y0=0,
                y1=1,
                line=dict(color="green", width=1),
                opacity=0.25,
                layer="below",
            )
        )
    for t in est_t_ms:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y domain",
                x0=float(t),
                x1=float(t),
                y0=0,
                y1=1,
                line=dict(color="red", width=1, dash="dash"),
                opacity=0.35,
                layer="below",
            )
        )
    for tau in ref_tau:
        shapes.append(
            dict(
                type="line",
                xref="x2",
                yref="y2 domain",
                x0=float(tau),
                x1=float(tau),
                y0=0,
                y1=1,
                line=dict(color="green", width=1),
                opacity=0.25,
                layer="below",
            )
        )
    for tau in est_tau:
        shapes.append(
            dict(
                type="line",
                xref="x2",
                yref="y2 domain",
                x0=float(tau),
                x1=float(tau),
                y0=0,
                y1=1,
                line=dict(color="red", width=1, dash="dash"),
                opacity=0.35,
                layer="below",
            )
        )

    fig.update_layout(
        shapes=shapes,
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(
        title="t (ms)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title="tau (cycles)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="s(t)", row=1, col=1)
    fig.update_yaxes(title_text="u(t)", row=2, col=1)

    fig.show()
    return {
        "meta": v,
        "window_local": (start_local, stop_local),
        "window_abs": (start_abs, stop_abs),
        "ref_abs": ref_abs,
        "est_abs": est_abs,
    }


# %%
WINDOW_MS = 32.0
PATH_CONTAINS = "vowel"

voiced_meta = list(get_voiced_meta(path_contains=PATH_CONTAINS))
quickgci_cache = {}

# %%

window = plot_random_selected_window(
    voiced_meta=voiced_meta,
    window_ms=WINDOW_MS,
    quickgci_cache=quickgci_cache,
)