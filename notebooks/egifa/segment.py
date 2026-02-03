# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
import scipy.io.wavfile
from plotly.subplots import make_subplots
from tqdm import tqdm

from utils import __datadir__, pyglottal
from utils.audio import frame_signal

EGIFA_DIR = __datadir__("EGIFA")
VOWEL_DIR = EGIFA_DIR / "vowel"


def _list_vowel_stems():
    stems = []
    for wav_path in sorted(VOWEL_DIR.glob("*.wav")):
        mat_path = wav_path.with_suffix(".mat")
        if mat_path.exists():
            stems.append(wav_path.with_suffix(""))
    if not stems:
        raise FileNotFoundError(f"No EGIFA vowels found under {VOWEL_DIR}")
    return stems


def _parse_f0_from_stem(stem: Path) -> float:
    for token in stem.name.split("_"):
        if token.endswith("hz"):
            return float(token[:-2])
    raise ValueError(f"Could not parse f0 from filename: {stem.name}")


def _load_vowel(stem: Path):
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

    gci_ref = np.squeeze(mat["gci"]).astype(np.int64)
    return fs, s, gf, gci_ref


# %%
WINDOW_MS = 32.0
HOP_MS = 16.0
TOL_FRAC = 0.2
SEED = 0

vowel_stems = _list_vowel_stems()

frames = []


def _tau_from_gci(frame_len: int, gci_idx):
    gci = np.asarray(gci_idx, dtype=int)
    gci = gci[(gci >= 0) & (gci < frame_len)]
    gci = np.unique(gci)

    tau_full = np.full(frame_len, np.nan, dtype=np.float32)
    mask = np.zeros(frame_len, dtype=bool)

    if len(gci) < 2:
        return tau_full, mask, gci

    for k in range(len(gci) - 1):
        a = gci[k]
        b = gci[k + 1]
        if b <= a:
            continue
        idx = np.arange(a, b, dtype=int)
        tau_full[idx] = k + (idx - a) / (b - a)
        mask[idx] = True

    tau_full[gci[-1]] = len(gci) - 1
    mask[gci[-1]] = True

    tau_masked = tau_full.copy()
    tau_masked[~mask] = np.nan
    return tau_masked, mask, gci


def _apply_nan_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32).copy()
    out[~mask] = np.nan
    return out


def estimate_oq(u: np.ndarray, gci=None) -> float:
    u = np.asarray(u, dtype=np.float32)
    valid = np.isfinite(u)
    if not np.any(valid):
        return np.nan
    u_max = np.nanmax(u)
    u_min = np.nanmin(u)
    height = u_min + 0.025 * (u_max - u_min)
    return float(np.mean(u[valid] > height))


for stem in tqdm(vowel_stems, desc="EGIFA vowels"):
    fs, s, gf, gci_ref = _load_vowel(stem)
    f0 = _parse_f0_from_stem(stem)

    period_samples = fs / f0
    frame_len = int(round(WINDOW_MS * fs / 1000.0))
    hop = int(round(HOP_MS * fs / 1000.0))

    frames_s = frame_signal(s, frame_len, hop)
    frames_u = frame_signal(gf, frame_len, hop)

    if len(s) < frame_len:
        starts = np.array([0], dtype=int)
    else:
        starts = np.arange(0, len(s) - frame_len + 1, hop, dtype=int)
    ends = starts + frame_len

    gci_est = pyglottal.quick_gci(
        s,
        fs=fs,
        fmin=20,
        fmax=400,
        theta=-np.pi / 2,
        reps=2,
    )
    gci_est = np.asarray(gci_est, dtype=int)

    left = np.searchsorted(gci_est, starts, side="left")
    right = np.searchsorted(gci_est, ends, side="left")
    counts = right - left

    expected = int(np.round((WINDOW_MS / 1000.0) * f0))
    candidate_idx = np.where(counts == expected)[0]

    for idx in candidate_idx:
        gci_idx = gci_est[left[idx] : right[idx]]
        if expected > 1:
            if len(gci_idx) < 2:
                continue
            diffs = np.diff(gci_idx)
            if not np.all(
                np.abs(diffs - period_samples) <= TOL_FRAC * period_samples
            ):
                continue

        start = starts[idx]
        ref = (
            gci_ref[(gci_ref >= start) & (gci_ref < start + frame_len)] - start
        )
        est = gci_idx - start

        tau_masked, mask, _ = _tau_from_gci(frame_len, est)
        if not np.any(mask):
            continue

        s_masked = _apply_nan_mask(frames_s[idx], mask)
        u_masked = _apply_nan_mask(frames_u[idx], mask)
        t_abs = 1e3 * (start + np.arange(frame_len)) / fs

        frames.append(
            dict(
                stem=stem,
                t=t_abs.astype(np.float32),
                tau=tau_masked,
                s=s_masked,
                u=u_masked,
                oq=estimate_oq(u_masked),
            )
        )

if frames:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(frames))
    split = int(0.8 * len(frames))
    train_frames = [frames[i] for i in perm[:split]]
    test_frames = [frames[i] for i in perm[split:]]
    print(
        f"Selected frames: {len(frames)} | "
        f"train={len(train_frames)} test={len(test_frames)}"
    )
else:
    train_frames = []
    test_frames = []
    print("No frames selected.")


# %%
EGIFA_GCI_FS = 20_000


def plot_random_selected_frame():
    if len(frames) == 0:
        print("No frames selected.")
        return

    rng = np.random.default_rng()
    frame = frames[int(rng.integers(len(frames)))]

    frame_s = frame["s"]
    frame_u = frame["u"]
    tau = frame["tau"]
    t_ms = frame["t"]
    n = np.arange(len(frame_s))

    fs, _ = scipy.io.wavfile.read(frame["stem"].with_suffix(".wav"))
    mat = scipy.io.loadmat(frame["stem"].with_suffix(".mat"))
    gci_ref = np.squeeze(mat["gci"]).astype(np.int64)
    gci_ref_ms = 1e3 * gci_ref / EGIFA_GCI_FS
    ref_idx = gci_ref_ms[(gci_ref_ms >= t_ms[0]) & (gci_ref_ms <= t_ms[-1])]

    est_idx = np.where(
        np.isfinite(tau) & np.isclose(tau, np.round(tau), atol=1e-6)
    )[0]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"EGIFA vowel: {frame['stem'].name} | OQ={frame['oq']:.2f}",
        ],
    )

    fig.add_trace(
        go.Scattergl(
            x=t_ms,
            y=frame_s,
            mode="lines",
            opacity=0.7,
            name="speech",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=tau,
            y=frame_u,
            mode="lines",
            opacity=0.7,
            name="u(t)",
        ),
        row=2,
        col=1,
    )

    shapes = []
    for t_ref in ref_idx:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t_ref,
                x1=t_ref,
                y0=0,
                y1=1,
                line=dict(color="green", width=1),
                opacity=0.25,
                layer="below",
            )
        )
    for idx in est_idx:
        if np.isfinite(tau[idx]):
            shapes.append(
                dict(
                    type="line",
                    xref="x2",
                    yref="paper",
                    x0=tau[idx],
                    x1=tau[idx],
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
        row=1,
        col=1,
        title="t (ms)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    fig.update_xaxes(
        row=2,
        col=1,
        title="tau (cycles)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )

    fig.update_yaxes(title_text="s(t)", row=1, col=1)
    fig.update_yaxes(title_text="u(t)", row=2, col=1)

    for t_ref in ref_idx:
        fig.add_annotation(
            x=t_ref,
            y=1.02,
            xref="x",
            yref="paper",
            text="ref",
            showarrow=False,
            font=dict(color="green", size=9),
        )
    for idx in est_idx:
        if np.isfinite(tau[idx]):
            fig.add_annotation(
                x=tau[idx],
                y=1.02,
                xref="x2",
                yref="paper",
                text="est",
                showarrow=False,
                font=dict(color="red", size=9),
            )

    fig.show()
    return frame


frame = plot_random_selected_frame()

# %%
