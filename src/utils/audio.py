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


def frame_signal_with_origins(x, frame_len, hop, origins):
    """
    Return frames of length `frame_len`, stepping every `hop`,
    but each frame start is snapped to the nearest allowed origin.

    origins: sorted list/array of allowed frame positions.
    """
    x = np.asarray(x)
    n = len(x)
    origins = np.asarray(origins)

    # special case: too short
    if n < frame_len:
        buf = np.zeros(frame_len, dtype=x.dtype)
        buf[-n:] = x
        return buf[None, :]

    # naive un-snapped frame starts
    raw_starts = np.arange(0, n - frame_len + 1, hop)

    # nearest origin for each raw start
    # binary search
    idx = np.searchsorted(origins, raw_starts)

    left = origins[np.clip(idx - 1, 0, len(origins) - 1)]
    right = origins[np.clip(idx, 0, len(origins) - 1)]

    nearest = np.where(
        np.abs(raw_starts - left) <= np.abs(raw_starts - right), left, right
    )

    # output frames
    frames = np.empty((len(nearest), frame_len), dtype=x.dtype)

    for i, o in enumerate(nearest):
        lo = max(o, 0)
        hi = min(o + frame_len, n)

        if lo == o and hi == o + frame_len:
            # fast path: no padding
            frames[i] = x[o : o + frame_len]
        else:
            # pad when origin is out-of-bounds or near border
            buf = np.zeros(frame_len, dtype=x.dtype)
            # x[lo:hi] should go at offset (lo - o)
            offset = lo - o
            buf[offset : offset + (hi - lo)] = x[lo:hi]
            frames[i] = buf

    return frames


def compute_gci_origins(one_gci, n, fs, true_pitch):
    gci_t = one_gci / fs
    T = 1.0 / true_pitch
    duration = n / fs

    k_start = int(np.ceil((0 - gci_t) / T))
    k_end = int(np.floor((duration - gci_t) / T))

    gci_times = gci_t + np.arange(k_start, k_end + 1) * T
    gci_samples = (gci_times * fs).round().astype(int)
    gci_samples = np.clip(gci_samples, 0, n - 1)

    return np.sort(gci_samples)


def fit_affine_lag_nrmse(x, y, maxlag):
    """Find best affine+lag fit of inferred signal x to ground truth y

    Find min_(a, b, lag) RMSE of (a * x[t - lag] + b) and y[t]

    For each lag k in [-maxlag, maxlag], we align x and y on their
    overlapping support, then fit an affine map a*x + b -> y
    by least squares. We compute the RMSE for that lag and pick the lag with smallest RMSE, then normalize and return normalized RMSE \in [0,1]
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    original = None
    best = None

    for k in range(-maxlag, maxlag + 1):
        if k >= 0:
            xs, ys = x[k:], y[: n - k]
        else:
            xs, ys = x[: n + k], y[-k:]
        m = len(xs)
        if m == 0:
            continue

        xm, ym = xs.mean(), ys.mean()
        xc, yc = xs - xm, ys - ym
        varx = np.dot(xc, xc) / m
        covxy = np.dot(xc, yc) / m
        a = 0.0 if varx == 0.0 else covxy / varx
        b = ym - a * xm

        err = ys - (a * xs + b)
        rmse = np.sqrt(np.dot(err, err) / m)

        sy = np.sqrt(np.dot(yc, yc) / m)  # std of overlapping y
        nrmse = np.inf if sy == 0.0 else rmse / sy

        if k == 0:
            original = dict(
                lag=k, a=a, b=b, rmse=rmse, nrmse=nrmse, aligned=aligned
            )

        if (best is None) or (nrmse < best["nrmse"]):
            aligned = np.full(n, np.nan)
            if k >= 0:
                aligned[: n - k] = a * x[k:] + b
            else:
                aligned[-k:] = a * x[: n + k] + b
            best = dict(
                lag=k, a=a, b=b, rmse=rmse, nrmse=nrmse, aligned=aligned
            )
    return best, original


def power_spectrum_db(x, fs):
    """
    Compute single-sided power spectrum of real signal x.

    Parameters
    ----------
    x : array_like
        Input signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    f : ndarray
        Frequency bins in Hz.
    power_db : ndarray
        Power spectral density in dB (10*log10).
    """
    x = np.asarray(x)
    n = len(x)
    X = np.fft.rfft(x)
    power = (np.abs(X) ** 2) / n
    f = np.fft.rfftfreq(n, 1.0 / fs)
    power_db = 10 * np.log10(power + 1e-12)
    return f, power_db
