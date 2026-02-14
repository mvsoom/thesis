# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from IPython.display import Audio, display
from plotly.subplots import make_subplots

from egifa.data import get_meta
from gci.lbgid import level_based_glottal_instant_detection

# %%
meta = get_meta("speech")


# %%
def plot_egifa(meta, index=None):
    if index is None:
        index = np.random.randint(len(meta))

    m = meta[index]

    speech, gf, fs = m["speech"], m["gf"], m["fs"]
    t = 1e3 * np.arange(len(speech)) / fs

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=speech,
            mode="lines",
            opacity=0.7,
            name="speech",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=gf,
            mode="lines",
            opacity=0.7,
            name="u(t)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=Path(m["wav"]).stem,
        height=650,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(
        title="Time (ms)",
        type="linear",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    fig.update_yaxes(title_text="normalized s(t)", row=1, col=1)
    fig.update_yaxes(title_text="u(t)", row=2, col=1)
    fig.show()

    return m


m = plot_egifa(meta)

display(Audio(m["speech"], rate=m["fs"]))

# %%
from scipy.ndimage import gaussian_filter1d

gf = m["gf"]
fs = m["fs"]

t_ms = 1e3 * np.arange(len(gf)) / fs

instants, aux = level_based_glottal_instant_detection(gf, fs, return_aux=True)

roof = aux["roof"]
floor = aux["floor"]
level = aux["level"]

tmin = 500
tmax = tmin + 1024
mask = (t_ms >= tmin) & (t_ms <= tmax)


# see [Smoothing DGF] in .chat for physiological motivation of both Gaussian smoothing AND sigma := 2.0 (which corresponds to ~0.1 msec temporal resolution of GCI events)
dgf_smoothed = gaussian_filter1d(gf, sigma=2.0, order=1, mode="nearest") * fs

fig = go.Figure()
fig.add_trace(
    go.Scattergl(x=t_ms[mask], y=gf[mask], mode="lines", name="gf", opacity=0.8)
)
fig.add_trace(
    go.Scattergl(
        x=t_ms[mask], y=roof[mask], mode="lines", name="roof", opacity=0.8
    )
)
fig.add_trace(
    go.Scattergl(
        x=t_ms[mask], y=floor[mask], mode="lines", name="floor", opacity=0.8
    )
)
fig.add_trace(
    go.Scattergl(
        x=t_ms[mask],
        y=level[mask],
        mode="lines",
        name="step",
        opacity=0.8,
        line=dict(color="black", width=1),
    )
)

for k, (left, right) in enumerate(instants):
    if t_ms[left] < tmin:
        continue
    if t_ms[right] > tmax:
        continue
    fig.add_trace(
        go.Scattergl(
            x=[t_ms[left], t_ms[right]],
            y=[floor[left], floor[right]],
            mode="lines+markers",
            name="(gci, goi) pairs" if k == 0 else None,
            marker=dict(color="black", size=6),
            line=dict(color="black", width=2),
            showlegend=(k == 0),
        )
    )


fig.show()

# %%
import matplotlib.pyplot as plt

T = np.diff([gci for gci, goi in instants])
plt.plot(np.log10(T))

# %%
import matplotlib.pyplot as plt

T = np.diff(np.concatenate(instants))
plt.plot(np.log10(T))