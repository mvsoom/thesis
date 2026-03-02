# %%
from pathlib import Path

import numpy as np
import scipy.io
import scipy.io.wavfile
from tqdm import tqdm

from egifa.data import (
    _shuffle_iterable,
    _split_idx_by_max_samples,
    nanwhiten,
    smooth_and_dgf,
)
from gci.meta import get_meta_grouped as gci_get_meta_grouped
from gci.meta import subgroups
from prism.svi import pad_waveforms
from utils import __datadir__, __memory__
from utils.constants import MIN_NUM_PERIODS

COMMONVOICE_DIR = __datadir__("CommonVoice")


def find_speech_gf_pairs(root):
    speech = {}
    gf = {}

    for p in Path(root).rglob("*.wav"):
        if not p.is_file():
            continue

        s = p.name
        if s.endswith("_speech.wav"):
            k = p.stem.removesuffix("_speech")
            speech[k] = p
        elif s.endswith("_gf.wav"):
            k = p.stem.removesuffix("_gf")
            gf[k] = p

    pairs = [(speech[k], gf[k]) for k in (speech.keys() & gf.keys())]

    return pairs


def num_data():
    return len(list(find_speech_gf_pairs(COMMONVOICE_DIR)))


def parse_filename(path):
    stem = Path(path).stem  # eg. 00001_common_voice_de_17298954_speech
    stem = stem.removesuffix("_speech")
    return {
        "name": stem,
    }


def _load_wav(path):
    fs, x = scipy.io.wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak

    # resample to 20 khz as these are long segments and our implementation of LBGID is O(N²) (just so that we can use numpy for speed)
    x = scipy.signal.resample_poly(x, 20_000, fs)
    fs = 20_000

    return fs, x


def _load_gf(path):
    return _load_wav(path)


def get_meta(path_contains=None):
    pairs = find_speech_gf_pairs(COMMONVOICE_DIR)
    for wav, gf in tqdm(pairs, desc="CommonVoice"):
        if path_contains is not None and path_contains not in str(wav):
            continue

        p = parse_filename(wav)
        fs, speech = _load_wav(wav)
        fs_gf, gf = _load_gf(gf)

        assert fs == fs_gf, f"Sampling rates do not match for {wav} and {gf}"

        meta = {
            "wav": str(wav),
            "gf": str(gf),
            "name": p["name"],
            "speech": speech,
            "fs": fs,
            "gf": gf,
        }

        yield meta


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    meta = get_meta()
    total = num_data()
    print(f"Number of CommonVoice recordings: {total}")

    # plot speech and gf for a random recording
    m = next(meta)
    t = np.arange(len(m["speech"])) / m["fs"]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, m["speech"])
    plt.title(f"Speech: {m['name']} [fs={m['fs']} Hz]")
    plt.subplot(2, 1, 2)
    plt.plot(t, m["gf"])
    plt.title("Glottal flow")
    plt.tight_layout()
    plt.show()

# %%
import multiprocessing as mp

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from gci.lbgid import level_based_glottal_instant_detection


def _proc_lgid(m):
    gf = m["gf"]
    fs = m["fs"]

    instants, aux = level_based_glottal_instant_detection(
        gf, fs, return_aux=True
    )

    if not instants.size:
        # there quite a few zero amplitude files
        return None

    gci = instants[:, 0]
    gci_ms = gci * 1e3 / fs

    m["lgid_aux"] = aux
    m["instants"] = instants
    m["instants_ms"] = 1e3 * instants / fs
    m["period_samples"] = np.diff(gci)
    m["periods_ms"] = np.diff(gci_ms)

    return m


@__memory__.cache
def get_meta_gid():
    n_jobs = min(12, mp.cpu_count())

    results = Parallel(
        n_jobs=n_jobs,
        return_as="generator",
        pre_dispatch="128*n_jobs",
    )(delayed(_proc_lgid)(m) for m in get_meta())

    meta = []
    for r in tqdm(results, total=num_data(), desc="LBGID"):
        if r is not None:
            meta.append(r)

    return meta


if __name__ == "__main__":
    meta = get_meta_gid()
    print(f"Computed LGID for {len(meta)} CommonVoice recordings")


# %%
from itertools import chain


@__memory__.cache
def get_meta_grouped():
    meta = get_meta_gid()

    # chunk this per 1000
    chunks = [meta[i : i + 1000] for i in range(0, len(meta), 1000)]
    results = [
        gci_get_meta_grouped(chunk) for chunk in tqdm(chunks, desc="Grouping")
    ]
    return list(chain.from_iterable(results))


if __name__ == "__main__":
    meta = get_meta_grouped()
    print(f"Computed groups for {len(meta)} CommonVoice recordings")


# %%
def get_voiced_meta(path_contains=None, max_samples=None, **smooth_dgf_kwargs):
    meta = get_meta_grouped()
    for m in tqdm(meta):
        if path_contains is not None and path_contains not in m["wav"]:
            continue

        anchors = m["instants"][:, 0]  # anchor at GCIs

        for group, idx in enumerate(subgroups(m["groups"])):
            if len(idx) < 2:  # at least a single period
                continue
            for subidx in _split_idx_by_max_samples(idx, anchors, max_samples):
                if len(subidx) - 1 < MIN_NUM_PERIODS:
                    continue

                start = anchors[subidx[0]]
                stop = anchors[subidx[-1]]  # exclude very last period

                v = {}

                v["wav"] = m["wav"]
                v["gf"] = m["gf"]
                v["name"] = m["name"]
                v["fs"] = m["fs"]

                v["group"] = group
                v["anchor_idx"] = subidx

                v["t_samples"] = np.arange(start, stop)
                v["speech"] = m["speech"][start:stop]
                v["gf"] = m["gf"][start:stop]
                v["lgid_aux"] = {
                    k: arr[start:stop] for k, arr in m["lgid_aux"].items()
                }

                gci = m["instants"][subidx, 0]
                goi = m["instants"][subidx, 1]

                v["gci"] = gci
                v["goi"] = goi

                # tau by simple interpolation
                x, y = gci, np.arange(len(gci))
                v["tau"] = np.interp(v["t_samples"], x, y)

                period = np.diff(gci)

                v["period_samples"] = period
                v["periods_ms"] = period / m["fs"] * 1000
                v["oq"] = 1.0 - (goi - gci)[:-1] / period

                # smooth according to physiological plausibility
                s = smooth_and_dgf(v, **smooth_dgf_kwargs)

                v["smooth"] = {}

                v["smooth"]["t_samples"] = s["t_samples"]  # 0 = start of file
                v["smooth"]["tau"] = s["tau"]  # 0 = start of voiced group
                v["smooth"]["gf"] = s["gf"]
                v["smooth"]["dgf"] = s["dgf"]
                v["smooth"]["speech"] = s["speech"]
                v["smooth"]["fs"] = s["fs"]

                yield v


def get_data(
    n=None,
    offset=0,
    path_contains=None,
    width=None,
    dtype=None,
    with_metadata=False,
    **smooth_dgf_kwargs,
):
    """Get data for learning test learning u'(t) from CommonVoice

    Data is shuffled, whitened and nan-padded to `width`.
    They are (tau, dgf) pairs, where tau is the normalized time within the glottal cycle (0 at GCI, 1 at next GCI, etc) and dgf is the smoothed and resampled derivative of Gaussian glottal flow.
    """
    meta = get_voiced_meta(
        path_contains=path_contains, max_samples=width, **smooth_dgf_kwargs
    )
    vs = _shuffle_iterable(list(meta))
    vs = vs[offset:] if n is None else vs[offset : offset + n]

    def pairs():
        for v in vs:
            tau = v["smooth"]["tau"]
            dgf = v["smooth"]["dgf"]

            dgf = nanwhiten(dgf)

            yield tau, dgf

    X, y = pad_waveforms(list(pairs()), width=width, dtype=dtype)

    if with_metadata:
        return X, y, vs

    return X, y


if __name__ == "__main__":
    X, y, meta = get_data(with_metadata=True)
    print(f"Got {len(X)} waveforms of shape {X.shape[1:]} with metadata")
