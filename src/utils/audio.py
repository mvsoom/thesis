import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited polyphase resampling that preserves dtype."""
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_out, sr_in)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down, axis=0).astype(x.dtype)


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Return a view of `x` with shape (n_frames, frame_len).

    - Frames start every `hop` samples.
    - Any “ragged” tail that can’t make a full window is dropped.
    - If `len(x) < frame_len`, zero-pad *before* the data and return exactly one frame.
    """
    n = len(x)
    if n < frame_len:
        buf = np.zeros(frame_len, dtype=x.dtype)
        buf[-n:] = x
        return buf[None, :]

    windows = sliding_window_view(x, frame_len)
    return windows[::hop]
