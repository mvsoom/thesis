# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from commonvoice.data import get_voiced_meta
from utils import time_this

# %%

with time_this():  # 20 sec
    meta = list(get_voiced_meta())


# %%

v = meta[np.random.randint(len(meta))]

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("Speech", "Glottal Flow", "Smoothed DGF"),
)

fig.add_trace(
    go.Scatter(
        x=v["t_samples"],
        y=v["speech"],
        mode="lines",
        name="original speech",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=v["smooth"]["t_samples"],
        y=v["smooth"]["speech"],
        mode="lines",
        name="smoothed speech",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=v["t_samples"],
        y=v["gf"],
        mode="lines",
        name="original gf",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=v["smooth"]["t_samples"],
        y=v["smooth"]["gf"],
        mode="lines",
        name="smoothed gf",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=v["smooth"]["t_samples"],
        y=v["smooth"]["dgf"],
        mode="lines",
        name="smoothed dgf",
    ),
    row=3,
    col=1,
)

fig.update_xaxes(title="t_samples", row=3, col=1)
fig.update_yaxes(title_text="speech", row=1, col=1)
fig.update_yaxes(title_text="gf", row=2, col=1)
fig.update_yaxes(title_text="dgf", row=3, col=1)
fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
)
fig.show()

# %%

# plot histogram of OQ

oq = np.concatenate([v["oq"] for v in meta])
fig = px.histogram(oq, nbins=50, title=f"OQ distribution (mean={oq[0]:.3f})")
fig.update_xaxes(title="OQ")
fig.update_yaxes(title="Count")
fig.show()

# %%
# plot histogram of period in ms
# really bad

f0_hz = np.concatenate([1000 / v["periods_ms"] for v in meta])
fig = px.histogram(
    f0_hz, nbins=50, title=f"F0 distribution (mean={f0_hz.mean():.2f} Hz)"
)
fig.update_xaxes(title="Period (ms)")
fig.update_yaxes(title="Count")
fig.show()
