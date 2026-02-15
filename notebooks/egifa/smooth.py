# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from egifa.data import get_voiced_meta

v = next(get_voiced_meta())

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
