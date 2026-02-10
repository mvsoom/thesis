# %%
import random
import warnings

import numpy as np

import aplawd
from aplawd import MARKINGS_FS_HZ, APLAWD_Markings
from prism.svi import pad_waveforms
from utils import (
    __cache__,
    __datadir__,
    __memory__,
    constants,
)


def _collect_periods_and_meta():
    markings_db = APLAWD_Markings(__datadir__("APLAWDW/markings/aplawd_gci"))
    recordings = aplawd.APLAWD(__datadir__("APLAWDW/dataset"))

    period_list = []
    meta_list = []
    for key in markings_db.keys():
        markings = markings_db.load(key)
        markings_ms = markings / MARKINGS_FS_HZ * 1000  # msec
        period_samples = np.diff(markings)
        period_ms = np.diff(markings_ms)
        period_list.append(period_ms)

        meta = {
            "key": key,
            "markings": markings,
            "markings_ms": markings_ms,
            "period_samples": period_samples,
            "periods_ms": period_ms,
            "markings_fs": MARKINGS_FS_HZ,
        }

        try:
            k = recordings.load(key)
        except Exception as e:
            warnings.warn(f"{key}: Failed to load APLAWD recording: {e}")
            k = None

        if k is not None:
            wav_file = None
            if hasattr(recordings, "_wav_dict"):
                wav_file = recordings._wav_dict.get(key)

            meta.update(
                {
                    "speech": k.s,
                    "egg": k.e,
                    "degg": k.d,
                    "fs": k.fs,
                    "egg_file": k.file,
                    "wav_file": wav_file,
                    "name": k.name,
                }
            )

        meta_list.append(meta)

    return period_list, meta_list


def get_list_of_periods(with_metadata=False):
    period_list, meta_list = _collect_periods_and_meta()
    if with_metadata:
        return period_list, meta_list
    return period_list


def get_data_periods(
    n=None, offset=0, width=None, dtype=np.float64, with_metadata=False
):
    period_list, meta_list = _collect_periods_and_meta()
    if n is None:
        period_list = period_list[offset:]
        meta_list = meta_list[offset:]
    else:
        period_list = period_list[offset : offset + n]
        meta_list = meta_list[offset : offset + n]

    indices_list = [0.5 + np.arange(len(p)) for p in period_list]

    pairs = list(zip(indices_list, period_list))

    X, y = pad_waveforms(pairs, width=width, dtype=dtype)

    if with_metadata:
        for meta, tau in zip(meta_list, indices_list):
            if isinstance(meta, dict):
                meta["tau"] = tau
        return X, y, meta_list

    return X, y


def get_whitener(y):
    mean = np.nanmean(y)
    std = np.nanstd(y)

    def whiten(y):
        return (y - mean) / std

    def unwhiten(y):
        return y * std + mean

    return whiten, unwhiten


if __name__ == "__main__":
    X, y = get_data_periods()

    # Choose a width for easier inference time
    lens = np.sum(~np.isnan(y), axis=1)

    print(f"Number of waveforms: {len(y)}")
    print(f"Average length: {np.mean(lens):.1f} samples")
    print(f"Max length: {np.max(lens)} samples")
    print(f"Min length: {np.min(lens)} samples")

    print("95% quantile of length:", np.quantile(lens, 0.95))
    print(
        "99% quantile of length:", np.quantile(lens, 0.99)
    )  # cut at width=320

    # We know physiological cutoff at ~50 Hz
    # This corresponds to 99% quantile
    p99 = np.nanquantile(y, 0.99)
    f0_cutoff = 1000 / p99
    print(f"F0 cutoff (99% quantile): {f0_cutoff:.2f} Hz")

    print("Number of events:", np.sum(~np.isnan(y)))
    print(
        "Events > cutoff (99%):", np.sum(y >= p99)
    )  # 1% => implies value of nu in t-PRISM

# %%
from utils import load_egg

PITCH_TRACK_MODEL = load_egg(
    "svi/aplawd/runs/M=64_iter=1_kernelname=matern:32.ipynb"
)


def load_recording_and_markers(recordings, markings, key):
    k = recordings.load(key)
    try:
        m = markings.load(key)
    except FileNotFoundError:
        m = None
    return k, m


def split_markings_into_voiced_groups(m, fs):
    """Split the markings into groups where the waveform is voiced"""
    max_period_length_msec = constants.MAX_PERIOD_LENGTH_MSEC
    min_num_periods = constants.MIN_NUM_PERIODS

    periods = np.diff(m) / fs * 1000  # msec
    split_at = np.where(periods > max_period_length_msec)[0] + 1
    for group in np.split(m, split_at):
        if len(group) <= min_num_periods + 1:
            continue
        else:
            yield group


def align_and_intersect(a, b):
    """Align two arrays containing sample indices as closely as possible"""
    a, b = a.copy(), b.copy()
    dist = np.abs(a[:, None] - b[None, :])
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    d = j - i
    if d >= 0:
        intersect = min(len(a), len(b) - d)
        a = a[0:intersect]
        b = b[d : d + intersect]
    elif d < 0:
        d = np.abs(d)
        intersect = min(len(a) - d, len(b))
        a = a[d : d + intersect]
        b = b[0:intersect]
    return a, b


def _realistic_periods(
    true_group,
    estimated_group,
    fs,
    min_period_length_msec,
    max_period_length_msec,
):
    """Check that the periods are within the physiological bounds"""
    min_period_length_msec = constants.MIN_PERIOD_LENGTH_MSEC
    max_period_length_msec = constants.MAX_PERIOD_LENGTH_MSEC

    true_periods = np.diff(true_group) / fs * 1000  # msec
    estimated_periods = np.diff(estimated_group) / fs * 1000  # msec

    if np.any(true_periods > max_period_length_msec) or np.any(
        true_periods < min_period_length_msec
    ):
        return False

    if np.any(estimated_periods > max_period_length_msec) or np.any(
        estimated_periods < min_period_length_msec
    ):
        return False

    return True


def yield_training_pairs(recordings, markings, estimate_gcis):
    """Yield all training pairs consisting of the true and Praat-estimated pitch periods in msec"""
    min_num_periods = constants.MIN_NUM_PERIODS

    for key in recordings.keys():
        k, m = load_recording_and_markers(recordings, markings, key)
        if m is None:
            # Only a few dozen
            warnings.warn(
                f"{k.name}: Discarded entire recording because of ground truth markings are missing"
            )
            continue

        try:
            gci_estimates = estimate_gcis(k.s, k.fs)
        except Exception as e:
            # Occurs when the recording is too short
            warnings.warn(
                f"{k.name}: Discarded entire recording because of Exception in `estimate_gcis()`: {e}"
            )
            continue

        voiced_groups = split_markings_into_voiced_groups(m, k.fs)

        # We call the APLAWD markings the 'true' group markings
        for true_group in voiced_groups:
            if len(true_group) < min_num_periods + 1:
                continue  # Discard voiced groups which are a priori too short

            # Intersect the current ground truth group as well as possible with Praat estimates
            true_group, estimated_group = align_and_intersect(
                true_group, gci_estimates
            )
            assert len(true_group) == len(estimated_group)
            if len(true_group) < min_num_periods + 1:
                continue

            if not _realistic_periods(
                true_group,
                estimated_group,
                k.fs,
            ):
                warnings.warn(
                    f"{k.name}: Discarded voiced group of GCIs because one of the synced periods is not within `{min | max}_period_length_msec`"
                )
                continue

            yield true_group, estimated_group


@__memory__.cache
def get_aplawd_training_pairs(return_praat_gci_error=False):
    """
    Get the training pairs from the APLAWD database.

    A training pair consists of (1) the ground truth pitch periods derived from the
    manually verified GCI markings and (2) the pitch periods as estimated
    from Praat's pulses. The latter (2) is aligned as closely as possible to (1).
    """
    # Get the recordings and the GCI markings
    recordings = aplawd.APLAWD(__datadir__("APLAWDW/dataset"))
    markings = aplawd.APLAWD_Markings(
        __datadir__("APLAWDW/markings/aplawd_gci")
    )

    # Get pairs of 'true' pitch periods and the ones estimated by Praat based on the recordings
    training_pairs = list(
        yield_training_pairs(recordings, markings, return_praat_gci_error)
    )

    return training_pairs


@__memory__.cache
def get_aplawd_training_pairs_subset(
    subset_size=5000,
    max_num_periods=100,
    seed=411489,
    return_praat_gci_error=False,
):
    """Select a subset of the training pairs with a max number of pitch periods"""
    training_pairs = get_aplawd_training_pairs(return_praat_gci_error)

    random.seed(seed)
    subset = random.choices(
        list(filter(lambda s: len(s[0]) <= max_num_periods, training_pairs)),
        k=subset_size,
    )

    return subset


def _moving_average(x, w):
    # https://stackoverflow.com/a/54628145/6783015
    return np.convolve(x, np.ones(w), "valid") / w


def _iqr(x):
    return float(np.diff(np.quantile(x, [0.25, 0.75])))


@__cache__
def fit_praat_relative_gci_error():
    fullset = get_aplawd_training_pairs_subset(return_praat_gci_error=True)

    praat_periods = np.concatenate([f[1] for f in fullset])
    praat_gci_errors = np.concatenate(
        [_moving_average(f[2], 2) for f in fullset]
    )

    rel_error = praat_gci_errors / praat_periods

    # Convert median and IQR to the Gaussian counterparts
    # See https://en.wikipedia.org/wiki/Interquartile_range#Distributions
    # for the 1.349 coefficient
    mu = np.median(rel_error)
    sigma = _iqr(rel_error) / 1.349

    return mu, sigma
