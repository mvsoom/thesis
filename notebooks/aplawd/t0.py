"""Investigate GCI estimation errors and jitter from APLAWD"""

# %%
import numpy as np
import plotly.express as px
from tqdm import tqdm

from aplawd.data import get_praatgci_meta, get_quickgci_meta
from utils import (
    align_and_intersect,
)
from utils.constants import MIN_NUM_PERIODS


def subgroups(groups):
    for gid in np.unique(groups):
        if gid == -1:
            continue
        idx = np.flatnonzero(groups == gid)
        yield idx


def get_errors(meta):
    errors = []
    for m in tqdm(meta):
        for idx in subgroups(m["groups"]):
            if len(idx) - 1 < MIN_NUM_PERIODS:
                continue

            markings = m["markings"][idx] * 1000 / m["markings_fs"]
            gcis = m["gcis"] * 1000 / m["fs"]

            aligned = align_and_intersect(markings, gcis)

            error = aligned[0] - aligned[1]  # error = (true - estimated)
            errors.append(error)

    errors = np.concatenate(errors)
    return errors


# %%
meta = get_quickgci_meta()
errors = get_errors(meta)

# 2 is smallest distance allowed (max 500 Hz F0)
mask = (-2 <= errors) & (errors <= 2)

fig = px.histogram(x=errors[mask])
fig.add_vline(
    x=-0.95,
    line_color="red",
    annotation_text="Approximate larynx-to-mic delay (Naylor et al. 2007)",
)
fig.update_layout(title="QuickGCI (true - estimate) errors")
fig.show()

# %%
meta = get_praatgci_meta()
errors = get_errors(meta)

mask = (-10 <= errors) & (errors <= 10)

fig = px.histogram(x=errors[mask])
fig.update_layout(title="Praat_ToPulse (true - estimate) errors")
fig.show()


# %%
# Measure APLAWD jitter over voiced groups
def yield_voiced_periods(meta):
    for m in meta:
        for idx in subgroups(m["groups"]):
            if len(idx) - 1 < MIN_NUM_PERIODS:
                continue

            markings = m["markings"][idx] * 1000 / m["markings_fs"]
            periods = np.diff(markings)
            yield periods


def measure_jitter(x):
    # https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local__absolute____.html
    return float(np.mean(np.abs(np.diff(x))))


from aplawd.data import get_meta_grouped

meta = [m for m in get_meta_grouped() if "groups" in m]

# measure jitter **over voiced groups**
jitter = np.array(
    [measure_jitter(periods) for periods in yield_voiced_periods(meta)]
)

jitter_usec = jitter * 1000.0

# %%
fig = px.histogram(x=jitter_usec)
fig.add_vline(x=83.200, line_dash="dash", annotation_text="Pathology threshold")
fig.update_layout(
    title="Measured jitter over voiced groups from APLAWD",
    xaxis_title="Local absolute jitter (microseconds)",
)
fig.show()
