# %%
"""Shared MATLAB engine + conversion helpers."""

from __future__ import annotations

import numpy as np


MATLAB_ENGINE = None


def matlab_engine():
    global MATLAB_ENGINE
    if MATLAB_ENGINE is None:
        import matlab.engine

        MATLAB_ENGINE = matlab.engine.start_matlab()
    return MATLAB_ENGINE


def add_path_recursive(path):
    eng = matlab_engine()
    eng.addpath(eng.genpath(str(path)), nargout=0)
    return eng


def matlab_col(x):
    import matlab

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return matlab.double(x.tolist())


def matlab_row(x):
    import matlab

    x = np.asarray(x, dtype=float).reshape(-1)
    return matlab.double([x.tolist()])


def numpy_vector(x):
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def matlab_idx_to_numpy0(x):
    idx = numpy_vector(x)
    if idx.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.rint(idx).astype(np.int64) - 1


def numpy0_to_matlab_idx_row(x):
    idx = np.asarray(x, dtype=np.int64).reshape(-1)
    return matlab_row(idx + 1)

