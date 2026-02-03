# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
import scipy.io.wavfile
import scipy.signal
from plotly.subplots import make_subplots

from utils import __datadir__

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


def _load_egifa(stem: Path):
    fs, s = scipy.io.wavfile.read(stem.with_suffix(".wav"))
    if s.ndim > 1:
        s = s[:, 0]
    s = s.astype(np.float32)

    mat = scipy.io.loadmat(stem.with_suffix(".mat"))
    gf = np.squeeze(mat["glottal_flow"]).astype(np.float32)
    upper = np.squeeze(mat["upper_area"]).astype(np.float32)
    lower = np.squeeze(mat["lower_area"]).astype(np.float32)
    return fs, s, gf, upper, lower


# %%
WINDOW_MS = 32.0
HOP_MS = 16.0

stems = _list_egifa_stems()
# for a subset:
# stems = [s for s in stems if s[0] == "vowel"]
stems = [s for s in stems if s[0] == "speech"]

subset, stem = stems[np.random.randint(len(stems))]
fs, s, gf, upper, lower = _load_egifa(stem)

frame_len = int(round(WINDOW_MS * fs / 1000.0))
hop = int(round(HOP_MS * fs / 1000.0))

if len(s) <= frame_len:
    start = 0
else:
    starts = np.arange(0, len(s) - frame_len + 1, hop, dtype=int)
    start = int(np.random.choice(starts))

end = min(start + frame_len, len(s))
n = np.arange(start, end)
t_ms = 1e3 * n / fs

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=[
        f"EGIFA {subset}: {stem.name}",
        "Glottal flow u(t)",
        "Glottal flow derivative u'(t)",
        "Upper & lower area",
    ],
)

fig.add_trace(
    go.Scattergl(
        x=t_ms,
        y=s[start:end],
        mode="lines",
        opacity=0.7,
        name="speech",
    ),
    row=1,
    col=1,
)

u_seg = gf[start:end]

u_seg -= np.min(u_seg)  # Simple breathy DC offset canceler
u_max = np.max(u_seg)
tol_value = 0.05 * u_max

du_dt = np.gradient(u_seg, 1 / fs)


# Criterion for selecting:
#
# Find the smallest regions who are the locus of
# - u(t) being near its minimum (below tol_value)
# - endpoints having maximal product of derivatives (one going up, one going down) to capture strong energy change
#
# This heuristic works amazingly well in practice on the vowels
# and tol_value can be set in function of maximum OQ attainable
# (so compare with physical measurements) on a cleaned test set
# Main failure mode is incomplete offsets (which we discard anyway)
# and very breathy sounds, where tol_value is too low for some regions
# (we can extrapolate)
#
# Operates currently on 32 msec intervals; longer intervals will need local
# tolerance for breathy signals
#
# Asssuming positive polarity, ie u(t) >~ 0:
#    left endpoint = GCI
#   right endpoint = GOI

low_idx = np.where(u_seg <= tol_value)[0]
subintervals = []
if low_idx.size:
    splits = np.where(np.diff(low_idx) > 1)[0] + 1
    regions = np.split(low_idx, splits)
    for region in regions:
        if region.size < 2:
            continue

        d = du_dt[region]
        energy = d[:, None] * d[None, :]  # NxN

        valid = energy < 0.0
        valid = np.triu(valid, k=0)

        if not valid.any():
            continue

        # mask invalid pairs to -inf so they never win
        masked = np.where(valid, np.abs(energy), -np.inf)

        # find best pair
        flat_idx = np.argmax(masked)
        i, j = np.unravel_index(flat_idx, masked.shape)

        # (i, j) are indices into `region`
        left = int(region[i]) + 1
        right = int(region[j]) - 1

        subintervals.append((left, right))

fig.add_trace(
    go.Scattergl(
        x=t_ms,
        y=u_seg,
        mode="lines",
        opacity=0.7,
        name="u(t)",
        customdata=np.sign(du_dt),
        hovertemplate="t=%{x:.3f} ms<br>u=%{y:.6g}<br>sign(u')=%{customdata:.0f}<extra></extra>",
    ),
    row=2,
    col=1,
)

for k, (left, right) in enumerate(subintervals):
    fig.add_trace(
        go.Scattergl(
            x=[t_ms[left], t_ms[right]],
            y=[tol_value, tol_value],
            mode="lines+markers",
            name="u-min+tol" if k == 0 else None,
            marker=dict(color="black", size=6),
            line=dict(color="black", width=2),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0.0, 0.0],
                arrayminus=[tol_value, tol_value],
                thickness=1,
                width=6,
                color="black",
            ),
            showlegend=(k == 0),
        ),
        row=2,
        col=1,
    )

fig.add_trace(
    go.Scattergl(
        x=t_ms,
        y=du_dt,
        mode="lines",
        opacity=0.6,
        name="u'(t)",
        line=dict(color="#2ca02c"),
    ),
    row=3,
    col=1,
)

fig.add_trace(
    go.Scattergl(
        x=t_ms,
        y=upper[start:end],
        mode="lines",
        opacity=0.7,
        name="upper",
        line=dict(color="#1f77b4"),
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=t_ms,
        y=lower[start:end],
        mode="lines",
        opacity=0.7,
        name="lower",
        line=dict(color="#ff7f0e"),
    ),
    row=4,
    col=1,
)

fig.update_layout(
    height=750,
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
fig.update_yaxes(title_text="u'(t)", row=3, col=1)
fig.update_yaxes(title_text="Area", row=4, col=1)

fig.show()

# %%
