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

        if (best is None) or (nrmse < best["nrmse"]):
            aligned = np.full(n, np.nan)
            if k >= 0:
                aligned[: n - k] = a * x[k:] + b
            else:
                aligned[-k:] = a * x[: n + k] + b
            best = dict(
                lag=k, a=a, b=b, rmse=rmse, nrmse=nrmse, aligned=aligned
            )
    return best


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
