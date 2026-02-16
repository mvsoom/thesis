from warnings import warn

import numpy as np
from scipy import signal
from scipy.integrate import simpson
from scipy.optimize import linear_sum_assignment
from scipy.signal import freqz


def ar_power_spectrum(a, fs, n_fft=4096, fmax=None, db=False):
    f = np.fft.rfftfreq(int(n_fft), d=1.0 / fs)
    if fmax is not None:
        keep = f <= float(fmax)
        f = f[keep]
    w = 2.0 * np.pi * f / float(fs)
    k = np.arange(1, a.size + 1)
    A = 1.0 - np.exp(-1j * np.outer(w, k)).dot(a)
    Pxx = 1.0 / (np.abs(A) ** 2)
    if db:
        Pxx = 10.0 * np.log10(Pxx)
    return f, Pxx


def band_ratio_db(f, P, low_max=200.0, mid_max=5000.0):
    # masks
    low = (f >= 0.0) & (f < low_max)
    mid = (f >= low_max) & (f < mid_max)
    high = f >= mid_max

    # integrate (sum is fine since grid uniform)
    E_low = np.sum(P[low])
    E_mid = np.sum(P[mid])
    E_high = np.sum(P[high])

    # ratios (mid relative to others)
    R_mid_low_db = 10.0 * np.log10(E_mid / E_low)
    R_mid_high_db = 10.0 * np.log10(E_mid / E_high)

    return R_mid_low_db, R_mid_high_db


def ar_gain_energy(a, n_w=16384):
    """Impulse response energy"""
    if not ar_stat_score(a) > 0.0:
        return np.inf
    # freqz uses H(z)=B(z)/A(z); we want B=1, A=1 - sum a_k z^-k
    w = np.linspace(0.0, np.pi, int(n_w), endpoint=True)
    # freqz expects A as [1, -a1, -a2, ...]
    _, H = freqz(b=[1.0], a=np.r_[1.0, -a], worN=w)
    H2 = (H.conj() * H).real
    # (1/π) ∫_0^π |H|^2 dω  ≈  (1/π) * simpson(H2, w)
    return float(simpson(H2, w) / np.pi)


def ar_stat_score(a):
    """
    Stationarity score in [0,1] for AR: x[t] = sum_k a[k]*x[t-k] + e[t].
    s = max(0, 1 - rho), where rho is the max pole radius of 1 - sum_k a z^{-k}.
    1.0 => comfortably stationary; 0.0 => pole on/outside unit circle.
    """
    a = np.asarray(a, dtype=float)
    if np.isnan(a).any():
        return np.nan
    if a.size == 0:
        return 1.0
    roots = np.roots(np.r_[1.0, -a])
    rho = 0.0 if roots.size == 0 else float(np.max(np.abs(roots)))
    return float(np.clip(1.0 - rho, 0.0, 1.0))


def get_polyorder(width, df):
    n = int(width // df)
    return n if n % 2 else n + 1


def get_bandwidths_at_FWHM(envelope, peaks):
    prominences, left_bases, right_bases = signal.peak_prominences(
        envelope, peaks
    )

    # To get the FWHM, we can use Scipy's peak_widths() if we trick it into
    # believing the peaks have prominence of 3 dB, and then measure peak width at
    # 100% of this prominence.
    prominences_hack = np.repeat(3.0, len(peaks))
    prominence_data = (prominences_hack, left_bases, right_bases)

    widths, width_heights, left_lps, right_lps = signal.peak_widths(
        envelope, peaks, rel_height=1.0, prominence_data=prominence_data
    )

    return widths


def stack_formant_samples(samples_list):
    lengths, counts = np.unique(
        [len(s) for s in samples_list], return_counts=True
    )
    if len(counts) == 1:
        return np.vstack(samples_list)
    else:
        n = len(samples_list)
        warn(
            f"The number of formants Q is not the same across all {n} samples: "
            f"filling out missing values with nans\n"
            f"Histogram of Q: {lengths // 2, counts}"
        )

        max_Q = max(lengths) // 2

        def nanresize(a):
            return np.pad(a, (0, max_Q - len(a)), constant_values=np.nan)

        def align(s):
            bandwidth, center = np.split(s, 2)
            return np.hstack([nanresize(bandwidth), nanresize(center)])

        aligned_samples_list = [align(s) for s in samples_list]
        return np.vstack(aligned_samples_list)


def estimate_formants(
    freq,  # Hz
    power_spectrum,  # dB
    df_upsample=1.0,  # Hz
    freq_bounds=(100.0, 6000.0),  # Hz
    filter_window_length=250.0,  # Hz
    peak_prominence=3.0,  # dB
    peak_distance=200.0,  # Hz
):
    """Heuristic to estimate formants from power spectrum in dB

    Note: see paretochain/uninformative project for source and notebook.
    """

    # Obtain spectral envelope by second-order Savitzky-Golay filter
    df = freq[1] - freq[0]
    envelope = signal.savgol_filter(
        power_spectrum, get_polyorder(filter_window_length, df), 2
    )

    # Upsample envelope to have a greater precision on the frequency axis
    n_upsample = int(len(envelope) * df / df_upsample)
    envelope_up, freq_up = signal.resample(
        envelope, n_upsample, freq, window="hamming"
    )

    # Discard upsampling artifacts at low and high frequencies
    keep = (freq_bounds[0] < freq_up) & (freq_up < freq_bounds[1])
    freq_up = freq_up[keep]
    envelope_up = envelope_up[keep]

    # Find peaks indices
    peaks, _ = signal.find_peaks(
        envelope_up,
        distance=peak_distance // df,
        prominence=peak_prominence,
    )

    # Get formant center frequencies and bandwidths (defined by FWHM)
    centers = freq_up[peaks]
    bandwidths = get_bandwidths_at_FWHM(envelope_up, peaks)

    return centers, bandwidths


def match_formants(est_f, true_f, est_bw=None, max_dev_hz=None, miss_cost=None):
    """
    Assign estimated formants to ground-truth F1..Fk (k=len(true_f)) by
    minimizing sum_j |f_est(i_j) - f_true(j)| over a one-to-one matching.
    Unmatched truths get NaN. Bandwidths are not used in the cost, but are
    aligned and returned.

    Inputs
    ------
    est_f : array_like, shape (m,)
        Estimated formant frequencies [Hz].
    true_f : array_like, shape (k,)
        Ground-truth formants [Hz], ordered (F1..Fk).
    est_bw : array_like or None, shape (m,)
        Estimated bandwidths [Hz] (optional; only aligned on output).
    max_dev_hz : float or None
        If set, disallow matches with |f_est - f_true| > max_dev_hz.
    miss_cost : float or None
        Penalty to leave a true formant unmatched. Defaults to
        (max_dev_hz or max(true_f, default=1.0))*10.

    Returns
    -------
    out : dict
        {
          "matched_freqs": np.ndarray (k,), estimates aligned to F1..Fk (NaN if unmatched),
          "matched_bws":   np.ndarray (k,), aligned bandwidths (NaN if unmatched),
          "assign_idx":    np.ndarray (k,), index into est_f for each true (or -1),
          "total_cost":    float, sum of assignment costs,
        }
    """
    est_f = np.asarray(est_f, dtype=float)
    true_f = np.asarray(true_f, dtype=float)
    m, k = len(est_f), len(true_f)

    if est_bw is None:
        est_bw = np.full(m, np.nan)
    else:
        est_bw = np.asarray(est_bw, dtype=float)

    if miss_cost is None:
        base = (
            max_dev_hz
            if max_dev_hz is not None
            else (float(true_f.max()) if k else 1.0)
        )
        miss_cost = 10.0 * base

    # Cost matrix: rows = truths, cols = estimates plus k dummy "miss" columns
    C = np.full((k, m + k), miss_cost, dtype=float)
    for j in range(k):
        for i in range(m):
            df = abs(est_f[i] - true_f[j])
            if (max_dev_hz is not None) and (df > max_dev_hz):
                C[j, i] = miss_cost
            else:
                C[j, i] = df  # frequency-only cost

    row_ind, col_ind = linear_sum_assignment(C)

    assign_idx = np.full(k, -1, dtype=int)
    matched_freqs = np.full(k, np.nan)
    matched_bws = np.full(k, np.nan)
    total_cost = 0.0

    for r, c in zip(row_ind, col_ind):
        cost = C[r, c]
        total_cost += cost
        if c < m and cost < miss_cost:
            assign_idx[r] = c
            matched_freqs[r] = est_f[c]
            matched_bws[r] = est_bw[c]

    return {
        "matched_freqs": matched_freqs,
        "matched_bws": matched_bws,
        "assign_idx": assign_idx,
        "total_cost": float(total_cost),
    }

