# %%
import warnings

import numpy as np
from tqdm import tqdm

import aplawd
from aplawd import MARKINGS_FS_HZ, APLAWD_Markings
from gci.meta import get_data_periods as gci_get_data_periods
from gci.meta import get_meta_grouped as gci_get_meta_grouped
from utils import (
    __datadir__,
    __memory__,
)


def get_meta():
    """Return APLAWD data = pairs of (speech, GCI markers). Note speech is NOT shifted"""
    markings_db = APLAWD_Markings(__datadir__("APLAWDW/markings/aplawd_gci"))
    recordings = aplawd.APLAWD(__datadir__("APLAWDW/dataset"))

    meta_list = []
    for key in tqdm(markings_db.keys(), desc="Loading APLAWD metadata"):
        markings = markings_db.load(key)
        markings_ms = markings / MARKINGS_FS_HZ * 1000  # msec
        period_samples = np.diff(markings)
        period_ms = np.diff(markings_ms)

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

    return meta_list


def get_data_periods(
    n=None,
    offset=0,
    width=None,
    dtype=np.float64,
    with_metadata=False,
):
    return gci_get_data_periods(
        get_meta(),
        n=n,
        offset=offset,
        width=width,
        dtype=dtype,
        with_metadata=with_metadata,
    )


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
@__memory__.cache
def get_meta_grouped():
    """Get APLAWD metadata with each GCI marked assigned to a group

    Groups are defined based on the weights (E[lambda]) from t-PRISM: a group is a contiguous sequence of GCIs where the weight is > 1.0, separated by GCIs with weight <= 1.0.
    This is a conservative choice which works quite well in practice.
    A more lenient choice could be weight > 0.5.

    Group numbers are assigned in order of occurrence, starting from 0 and incrementing by 1 at each transition from good to bad, bad to bad, or bad to good.
    GCIs with weight == 0.0 have been masked out by pad_waveforms() and occupy group number -1.
    """
    width = 320  # cutoff at 99% quantile
    return gci_get_meta_grouped(get_meta(), width=width)


if __name__ == "__main__":
    meta = get_meta_grouped()
    print(f"Computed groups for {len(meta)} APLAWD recordings")

# %%
from gci.estimate import gci_estimates_from_praat, gci_estimates_from_quickgci


@__memory__.cache
def get_gci_meta(gci_estimator):
    meta = [m for m in get_meta_grouped() if "groups" in m]

    for m in tqdm(meta, desc="Estimating GCIs"):
        m["gcis"] = gci_estimator(m["speech"], m["fs"])
        m["gcis_ms"] = m["gcis"] * 1000 / m["fs"]

    return meta


def get_quickgci_meta():  # 10 min
    return get_gci_meta(gci_estimates_from_quickgci)


def get_praatgci_meta():  # 30 sec
    return get_gci_meta(gci_estimates_from_praat)

def get_dypsagoi_meta():  # 15 min, needs MATLAB
    from gci.egifa import gci_estimates_from_dypsagoi

    return get_gci_meta(gci_estimates_from_dypsagoi)


def get_sedreams_meta():  # 10 min, needs MATLAB
    from gci.egifa import gci_estimates_from_sedreams

    return get_gci_meta(gci_estimates_from_sedreams)


if __name__ == "__main__":
    for f in [
        get_quickgci_meta,
        get_praatgci_meta,
        get_dypsagoi_meta,
        get_sedreams_meta,
    ]:
        meta = f()
        print(
            f"Computed {f.__name__} estimates for {len(meta)} APLAWD recordings"
        )