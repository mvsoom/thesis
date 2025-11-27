"""Port of https://github.com/covarep/covarep/blob/master/glottalsource/polarity_reskew.m"""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, ellip, ellipord, filtfilt, get_window, lfilter
from scipy.stats import skew


def lpc_autocorr(x, order):
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        a = np.zeros(order + 1)
        a[0] = 1.0
        return a
    r = np.correlate(x, x, mode="full")
    mid = len(x) - 1
    r = r[mid : mid + order + 1]  # r[0..order]
    sol = solve_toeplitz((r[:-1], r[:-1]), r[1:])
    a = np.concatenate(([1.0], -sol))
    return a


def lpc_residual(sig, win_len, shift, order):
    sig = np.asarray(sig, float)
    n = len(sig)
    win = get_window("hann", win_len, fftbins=False)
    res = np.zeros(n)
    start = 0

    while start + win_len <= n:
        seg = sig[start : start + win_len] * win
        a = lpc_autocorr(seg, order)
        invf = lfilter(a, [1.0], seg)
        num = np.sum(seg**2)
        den = np.sum(invf**2)
        if den > 0:
            invf *= np.sqrt(num / den)
        res[start : start + win_len] += invf
        start += shift

    m = np.max(np.abs(res))
    if m > 0:
        res /= m
    return res


def polarity_reskew(speech_signal, fs):
    """Estimate the speech polarity using the RESKEW method from Drugman (2020)

    Polarity detection is based on the difference in skew of the LPC residuals of the original and a highpass-filtered version of the speech signal.

    From the paper:
    > It is worth noting that this calculation is not performed on a frame basis, but on the whole speech signal.

    Note: don't use this on DGF or GF signals, as these are already LPC-like residuals.
    """
    s = np.asarray(speech_signal, float)

    # Highpass filter as in Drugman (480–500 Hz transition)
    wp = 480.0 / (fs / 2.0)
    ws = 500.0 / (fs / 2.0)
    rp, rs = 3.0, 60.0
    n, wn = ellipord(wp, ws, rp, rs)
    b, a = ellip(n, rp, rs, wn, btype="high")
    s_hp = filtfilt(b, a, s)

    win_len = int(0.025 * fs)
    shift = int(0.005 * fs)
    order = int(fs / 1000.0) + 2

    # Two residuals: standard + HP-residual
    r1 = lpc_residual(s, win_len, shift, order)
    r2 = lpc_residual(s_hp, win_len, shift, order)

    r1 = np.nan_to_num(r1)
    r2 = np.nan_to_num(r2)

    # Drugman's decision rule
    pol = np.sign(skew(r1) - skew(r2))
    return int(pol if pol != 0 else 1)


def dgf_polarity(dgf, fs):
    dgf = np.asarray(dgf, float)

    # light smoothing to suppress tiny spikes
    # (but NOT enough to smear the closure peak)
    win = max(3, int(0.0005 * fs))  # ~0.5 ms
    if win % 2 == 0:
        win += 1
    k = np.ones(win) / win
    dgf_s = np.convolve(dgf, k, mode="same")

    # high-pass filter tuned to emphasize closure
    # closure dominates above ~1 kHz; below that is open-phase slope
    hp_cut = 1000.0 / (fs / 2.0)
    b, a = butter(4, hp_cut, btype="high")
    dgf_hp = filtfilt(b, a, dgf_s)

    # nan-safe
    dgf_s = np.nan_to_num(dgf_s)
    dgf_hp = np.nan_to_num(dgf_hp)

    # compare skewness: same idea as RESKEW
    s1 = skew(dgf_s)
    s2 = skew(dgf_hp)

    pol = np.sign(s1 - s2)
    return int(pol if pol != 0 else 1)
