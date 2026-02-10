# %%
import numpy as np

from aplawd import APLAWD_Markings
from prism.svi import pad_waveforms
from utils import __datadir__


def get_list_of_periods():
    period_list = []

    markings_db = APLAWD_Markings(__datadir__("APLAWDW/markings/aplawd_gci"))

    for key in markings_db.keys():
        m = markings_db.load(key)
        period_ms = np.diff(m) / 20000 * 1000
        period_list.append(period_ms)

    return period_list


def get_data_periods(n=None, offset=0, width=None, dtype=np.float64):
    if n is None:
        period_list = get_list_of_periods()[offset:]
    else:
        period_list = get_list_of_periods()[offset : offset + n]
    indices_list = [0.5 + np.arange(len(p)) for p in period_list]

    pairs = list(zip(indices_list, period_list))

    X, y = pad_waveforms(pairs, width=width, dtype=dtype)
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
