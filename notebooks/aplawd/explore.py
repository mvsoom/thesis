# %%
import numpy as np
import plotly.graph_objects as go
from IPython.display import Audio, display
from plotly.subplots import make_subplots

from aplawd.data import get_meta
from gci.estimate import gci_estimates_from_quickgci

meta_list = get_meta()
meta_with_speech = [m for m in meta_list if "speech" in m]

# %%


def plot_recording_and_degg(index=None):
    if index is None:
        index = np.random.randint(len(meta_with_speech))

    meta = meta_with_speech[index]

    speech = meta["speech"]
    degg = meta["degg"]
    fs = meta["fs"]
    recording_gci = meta["markings"]
    name = meta.get("name", meta.get("key", "APLAWD"))

    estimated_gci = gci_estimates_from_quickgci(speech, fs)

    t_s = 1e3 * np.arange(len(speech)) / fs
    t_d = 1e3 * np.arange(len(degg)) / fs

    # t_s -= 0.9  # shift to align with markings

    ref_t = t_d[recording_gci]
    est_t = t_s[estimated_gci]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"APLAWD waveform: {name}", "DEGG"],
    )

    fig.add_trace(
        go.Scattergl(x=t_s, y=speech, mode="lines", opacity=0.6, name="speech"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(x=t_d, y=degg, mode="lines", opacity=0.6, name="DEGG"),
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
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
        ),
    )

    fig.show()
    return meta


meta = plot_recording_and_degg()

display(Audio(meta["speech"], rate=meta["fs"]))
