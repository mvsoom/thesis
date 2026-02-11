# %%

import numpy as np
from scipy import signal


def anchor(idx, N):
    if idx.size == 0:
        return np.array([0, N - 1], dtype=int)
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != N - 1:
        idx = np.r_[idx, N - 1]
    return idx


def find_extrema(gf, fs, relative_prominence=0.1):
    # peaks are at least 2.5 msec apart
    distance = int(fs * 2.5e-3)

    range = np.percentile(gf, 95) - np.percentile(gf, 5)
    prominence = relative_prominence * range

    peaks, _ = signal.find_peaks(gf, distance=distance, prominence=prominence)

    troughs, _ = signal.find_peaks(
        -gf, distance=distance, prominence=prominence
    )

    N = len(gf)
    peaks = anchor(peaks, N)
    troughs = anchor(troughs, N)

    return peaks, troughs


def interpolate_everywhere(x, signal):
    N = len(signal)
    return np.interp(np.arange(N), x, signal[x])


def level_based_glottal_instant_detection(gf, fs, level=0.01, return_aux=False):
    """
    Level-based Glottal Instant Detection (LGID).

    Detects glottal closure (GCI) and opening (GOI) instants by:

    1. Estimating adaptive upper and lower envelopes via peak
       and trough interpolation.
    2. Defining a relative amplitude level within the local
       dynamic range.
    3. Segmenting low-level regions based on this adaptive
       threshold.
    4. Pairing opposite-sign derivative extrema inside each
       region to estimate (GCI, GOI) events.

    The method is invariant to slow amplitude modulation and
    robust to global gain changes.

    Returns
    -------
    instants : list[tuple[int, int]]
        Detected (gci, goi) index pairs.
    """
    peaks, troughs = find_extrema(gf, fs)

    roof = interpolate_everywhere(peaks, gf)
    floor = interpolate_everywhere(troughs, gf)

    roof = np.maximum(roof, gf - 1e-12)
    floor = np.minimum(floor, gf + 1e-12)

    level = floor + level * (roof - floor)

    # need to run GCI detection on "hard" derivative because we want large peaks in DGF
    dgf = np.gradient(gf, 1 / fs)

    sub_idx = np.where(gf <= level)[0]
    instants = []
    if sub_idx.size:
        splits = np.where(np.diff(sub_idx) > 1)[0] + 1
        regions = np.split(sub_idx, splits)
        for region in regions:
            if region.size < 2:
                continue

            # find the pair of points in region with maximally "opposed" energy
            # where energy is defined as the product of the DGF values (which will be negative for a GOI and positive for a GCI)
            d = dgf[region]
            energy = d[:, None] * d[None, :]

            valid = energy < 0.0
            valid = np.triu(valid, k=0)

            if not valid.any():
                continue

            # mask invalid pairs to -inf so they never win
            masked = np.where(valid, np.abs(energy), -np.inf)

            # find best pair
            flat_idx = np.argmax(masked)
            i, j = np.unravel_index(flat_idx, masked.shape)

            # (i, j) are indices into `region`
            gci = int(region[i]) + 1
            goi = int(region[j]) - 1

            instants.append((gci, goi))  # where gci < goi

    aux = {
        "roof": roof,
        "floor": floor,
        "level": level,
        "dgf": dgf,
    }

    return instants if not return_aux else (instants, aux)
