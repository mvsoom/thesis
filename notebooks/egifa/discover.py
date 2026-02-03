# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
import scipy.io.wavfile
from plotly.subplots import make_subplots

from utils import __datadir__, pyglottal

EGIFA_DIR = __datadir__("EGIFA")


def _list_egifa_stems():
    stems = []
    for subset in ("speech", "vowel"):
        subset_dir = EGIFA_DIR / subset
        if not subset_dir.exists():
            continue
        for wav_path in sorted(subset_dir.glob("*.wav")):
            mat_path = wav_path.with_suffix(".mat")
            if mat_path.exists():
                stems.append((subset, wav_path.with_suffix("")))
    if not stems:
        raise FileNotFoundError(
            f"No EGIFA .wav/.mat pairs found under {EGIFA_DIR}"
        )
    return stems


def _load_egifa_stem(stem: Path):
    fs, s = scipy.io.wavfile.read(stem.with_suffix(".wav"))
    if s.ndim > 1:
        s = s[:, 0]
    s = s.astype(np.float32)
    s_max = np.max(np.abs(s))
    if s_max > 0:
        s = s / s_max

    mat = scipy.io.loadmat(stem.with_suffix(".mat"))
    gf = np.squeeze(mat["glottal_flow"]).astype(np.float32)
    gf_max = np.max(np.abs(gf))
    if gf_max > 0:
        gf = gf / gf_max

    gci = np.squeeze(mat["gci"]).astype(np.int64)
    return fs, s, gf, gci, mat


def plot_random_egifa(window_ms=None):
    stems = _list_egifa_stems()
    subset, stem = stems[np.random.randint(len(stems))]
    fs, s, gf, gci, _ = _load_egifa_stem(stem)

    t_s = 1e3 * np.arange(len(s)) / fs
    t_gf = 1e3 * np.arange(len(gf)) / fs
    gci_t = 1e3 * gci / 20000

    estimated_gci = pyglottal.quick_gci(
        s,
        fs=fs,
        fmin=20,
        fmax=400,
        theta=-np.pi / 2,
        reps=2,
    )
    est_t = 1e3 * estimated_gci / fs

    if window_ms is None or t_s[-1] <= window_ms:
        t1, t2 = t_s[0], t_s[-1]
    else:
        t1 = np.random.uniform(0, t_s[-1] - window_ms)
        t2 = t1 + window_ms

    mask_s = (t1 <= t_s) & (t_s <= t2)
    mask_gf = (t1 <= t_gf) & (t_gf <= t2)
    mask_ref = (t1 <= gci_t) & (gci_t <= t2)
    mask_est = (t1 <= est_t) & (est_t <= t2)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f"EGIFA {subset}: {stem.name}",
            "Ground truth u(t)",
        ],
    )

    fig.add_trace(
        go.Scattergl(
            x=t_s[mask_s],
            y=s[mask_s],
            mode="lines",
            opacity=0.7,
            name="speech",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=t_gf[mask_gf],
            y=gf[mask_gf],
            mode="lines",
            opacity=0.7,
            name="u(t)",
        ),
        row=2,
        col=1,
    )

    shapes = []

    for t in gci_t[mask_ref]:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                line=dict(color="green", width=1),
                opacity=0.25,
                layer="below",
            )
        )

    for t in est_t[mask_est]:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                line=dict(color="red", width=1, dash="dash"),
                opacity=0.35,
                layer="below",
            )
        )

    fig.update_layout(
        shapes=shapes,
        height=650,
        hovermode="x unified",
    )
    fig.update_xaxes(
        title="Time (ms)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    fig.update_yaxes(title_text="s(t)", row=1, col=1)
    fig.update_yaxes(title_text="u(t)", row=2, col=1)

    fig.show()


plot_random_egifa()
