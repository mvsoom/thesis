from warnings import warn

import numpy as np
from scipy import signal


def ar_power_spectrum(a, fs, n_fft=4096, fmax=None, convention="plus"):
    """
    Return PSD proportional to 1/|A(e^{-jω})|^2 for an AR(P) model.

    a: length-P array of AR coefficients excluding a0.
       convention="plus"      means A(z)=1 + a1 z^{-1} + ... + aP z^{-P}
       convention="lpc"       means 'predictor coeffs' (Levinson): A(z)=1 - a1 z^{-1} - ... - aP z^{-P}
    """
    a = np.asarray(a, float)
    if convention == "plus":
        den = np.r_[1.0, a]
    elif convention == "lpc":
        den = np.r_[1.0, -a]
    else:
        raise ValueError("convention must be 'plus' or 'lpc'")

    f = np.fft.rfftfreq(int(n_fft), d=1.0 / fs)
    if fmax is not None:
        keep = f <= float(fmax)
        f = f[keep]
    w = 2.0 * np.pi * f / float(fs)  # rad/sample
    _, H = signal.freqz(b=[1.0], a=den, worN=w)
    Pxx = np.abs(H) ** 2  # ∝ 1/|A|^2

    return f, Pxx


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
