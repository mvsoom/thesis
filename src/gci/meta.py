# %%
import jax.numpy as jnp
import numpy as np
from gpjax import Dataset

from prism.svi import pad_waveforms
from prism.t_svi import do_t_prism
from utils import (
    load_egg,
)

PITCH_TRACK_MODEL_NAME = "svi/aplawd/runs/M=64_iter=1_kernelname=rbf.ipynb"


def pitch_track_model(model_name=PITCH_TRACK_MODEL_NAME):
    return load_egg(model_name)


def get_data_periods(
    meta,
    n=None,
    offset=0,
    width=None,
    dtype=np.float64,
    with_metadata=False,
):
    periods_list = [m["periods_ms"] for m in meta]

    if n is None:
        periods_list = periods_list[offset:]
        meta = meta[offset:]
    else:
        periods_list = periods_list[offset : offset + n]
        meta = meta[offset : offset + n]

    indices_list = [0.5 + np.arange(len(p)) for p in periods_list]

    pairs = list(zip(indices_list, periods_list))

    X, y = pad_waveforms(pairs, width=width, dtype=dtype)

    if with_metadata:
        for m, tau in zip(meta, indices_list):
            if isinstance(meta, dict):
                m["tau"] = tau
        return X, y, meta

    return X, y


def group_based_on_weights(w):
    good = w > 1.0  # conservative
    mask = w == 0.0

    g = np.abs(np.diff(good, axis=1, prepend=0))
    g = g.astype(bool)
    g = g | (~good)
    g = np.cumsum(g, axis=1)
    g = g - g[:, 0][:, None]  # start from 0 for each waveform

    groups = g
    groups[mask] = -1  # Mask out the masked points with group -1

    return groups


def get_meta_grouped(meta, model_name=PITCH_TRACK_MODEL_NAME, **kwargs):
    """Group GCIs into voiced groups based on a pitch track model"""
    MODEL = pitch_track_model(model_name)

    X, y, meta = get_data_periods(meta, with_metadata=True, **kwargs)
    X = jnp.array(X, dtype=jnp.float64)
    y = jnp.array(y, dtype=jnp.float64)

    y = jnp.log10(y)
    y = MODEL["whiten"](y)

    # Heavy calculation
    _, _, weights = do_t_prism(MODEL["qsvi"], Dataset(X, y))

    weights = np.array(weights)
    groups = group_based_on_weights(weights)
    for m, g, w in zip(meta, groups, weights):
        m["groups"] = g
        m["weights"] = w

    return meta

def subgroups(groups):
    for gid in np.unique(groups):
        if gid == -1:
            continue
        idx = np.flatnonzero(groups == gid)
        yield idx


if __name__ == "__main__":
    model = pitch_track_model()
    print(model)