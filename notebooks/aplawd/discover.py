# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aplawd import APLAWD, APLAWD_Markings
from utils import __datadir__, pyglottal

aplawd_db = APLAWD(__datadir__("APLAWDW/dataset"))
markings_db = APLAWD_Markings(__datadir__("APLAWDW/markings/aplawd_gci"))


# %%
# check reference markings: need some local alignment by shift ~ 0.9 msec
def plot_recording_and_degg(recordings, markings, key=None):
    aplawd_db = recordings
    markings_db = markings

    if key is None:
        key = np.random.choice(aplawd_db.keys())

    # recording = aplawd_db.load_shifted(key)
    recording = aplawd_db.load(key)
    recording_gci = markings_db.load(recording.name)

    # NOTE: use Serwys parameters: https://chatgpt.com/s/t_697e19dc95988191b1d2af6b89e81f3d
    # NOTE 2: needs positive polarity
    estimated_gci = pyglottal.quick_gci(
        recording.s,
        fs=recording.fs,
        fmin=20,
        fmax=400,
        theta=-np.pi / 2,
        reps=2,
    )

    t_s = 1e3 * np.arange(len(recording.s)) / recording.fs
    t_d = 1e3 * np.arange(len(recording.d)) / recording.fs

    # t_s -= 0.9  # shift to align with markings

    ref_t = t_d[recording_gci]
    est_t = t_s[estimated_gci]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"APLAWD waveform: {recording.name}", "DEGG"],
    )

    fig.add_trace(
        go.Scattergl(
            x=t_s, y=recording.s, mode="lines", opacity=0.6, name="speech"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=t_d, y=recording.d, mode="lines", opacity=0.6, name="DEGG"
        ),
        row=2,
        col=1,
    )

    # reference markers
    fig.add_trace(
        go.Scattergl(
            x=ref_t,
            y=np.zeros_like(ref_t),
            mode="markers",
            marker=dict(color="green", size=6),
            name="reference GCI",
        ),
        row=2,
        col=1,
    )

    # estimated markers
    fig.add_trace(
        go.Scattergl(
            x=est_t,
            y=np.zeros_like(est_t),
            mode="markers",
            marker=dict(color="red", size=6),
            name="estimated GCI",
        ),
        row=2,
        col=1,
    )

    shapes = []

    # grey reference lines
    for t in ref_t:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                line=dict(color="grey", width=1),
                opacity=0.25,
                layer="below",
            )
        )

    # dashed estimated lines
    for t in est_t:
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

    fig.update_layout(shapes=shapes)

    # better zoom feel
    fig.update_xaxes(
        title="time (ms)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )

    fig.update_yaxes(autorange=True)

    fig.update_layout(
        height=700,
        hovermode="x unified",
    )

    fig.show()


plot_recording_and_degg(aplawd_db, markings_db)

# %%
# Test quick_gci
key = np.random.choice(aplawd_db.keys())

# recording = aplawd_db.load_shifted(key) # with 0.95 msec shift
recording = aplawd_db.load(key)
recording_gci = markings_db.load(recording.name)

t_s = 1e3 * np.arange(len(recording.s)) / recording.fs
t_d = 1e3 * np.arange(len(recording.d)) / recording.fs
gci_t = t_d[recording_gci]

# shift
t_s -= 0.9

pyglottal.quick_gci(recording.s, fs=recording.fs)
