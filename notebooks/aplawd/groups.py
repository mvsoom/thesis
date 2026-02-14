"""Plot grouped pitch tracks from APLAWD metadata."""

# %%
import numpy as np
import pandas as pd
import plotly.express as px

from aplawd.data import get_meta_grouped

meta = [m for m in get_meta_grouped() if "groups" in m]

# %%
# It works well. The condition weight > 1 is conservative, but plenty of data survives
# Parity means: when parity flips (0 -> 1 or 1 -> 0), a new group has started. It's just a plotting trick

def _track_dataframe(m):
    tau = np.arange(len(m["periods_ms"])) + 0.5
    periods = np.asarray(m["periods_ms"])
    groups = np.asarray(m["groups"]).astype(int)
    weights = np.asarray(m["weights"])

    if groups.shape[0] >= tau.shape[0]:
        groups = groups[: tau.shape[0]]
        weights = weights[: tau.shape[0]]
    else:
        tau = tau[: groups.shape[0]]
        periods = periods[: groups.shape[0]]
        weights = weights[: groups.shape[0]]

    valid = (
        ~np.isnan(tau) & ~np.isnan(periods) & (groups >= 0) & ~np.isnan(weights)
    )

    df = pd.DataFrame(
        {
            "tau": tau[valid],
            "period_ms": periods[valid],
            "group": groups[valid],
            "weight": weights[valid],
        }
    )
    df["group_str"] = df["group"].astype(str)
    df["parity"] = df["group"] % 2
    return df


def plot_track(index=None, parity=True):
    if not meta:
        raise ValueError("No grouped metadata available.")
    if index is None:
        index = np.random.randint(len(meta))

    m = meta[index]
    df = _track_dataframe(m)
    name = m.get("name", m.get("key", "APLAWD"))

    if parity:
        fig = px.scatter(
            df,
            x="tau",
            y="period_ms",
            color="parity",
            color_discrete_map={0: "#1f77b4", 1: "#ff7f0e"},
            hover_data={"weight": True, "group": True, "parity": True},
        )
    else:
        fig = px.scatter(
            df,
            x="tau",
            y="period_ms",
            color="group_str",
            hover_data={"weight": True, "group": True},
        )

    fig.add_scatter(
        x=df["tau"],
        y=df["period_ms"],
        mode="lines",
        line=dict(color="rgba(0,0,0,0.2)"),
        showlegend=False,
    )

    fig.update_layout(
        title=f"Pitch track: {name}",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2,
            yanchor="top",
        ),
    )
    fig.update_xaxes(title="tau")
    fig.update_yaxes(title="period (ms)")
    return fig


def plot_random_tracks(n=3, parity=True):
    for _ in range(n):
        plot_track(parity=parity).show()


plot_random_tracks(n=3, parity=True)
