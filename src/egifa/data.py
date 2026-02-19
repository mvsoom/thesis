# %%
from pathlib import Path

import numpy as np
import scipy.io
import scipy.io.wavfile
from tqdm import tqdm

from gci.meta import get_meta_grouped as gci_get_meta_grouped
from utils import __datadir__, __memory__

EGIFA_DIR = __datadir__("EGIFA")


def find_wav_mat_pairs(root):
    wav = {}
    mat = {}
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() == ".wav":
            wav[p.stem] = p
        elif p.suffix.lower() == ".mat":
            mat[p.stem] = p

    # match by identical stem
    pairs = []
    for k in wav.keys() & mat.keys():
        pairs.append((wav[k], mat[k]))

    return pairs


def parse_filename(path):
    """Convert eg "modal_e_180hz_2000pa.wav" to {"name": "modal_e", "hz": 180, "pa": 2000}"""
    stem = Path(path).stem
    parts = stem.split("_")

    pa = int(parts[-1].removesuffix("pa"))
    hz = int(parts[-2].removesuffix("hz"))
    name = "_".join(parts[:-2])

    return {"name": name, "hz": hz, "pa": pa}


def _load_wav(path):
    fs, x = scipy.io.wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak

    return fs, x


def _load_gf(path):
    mat = scipy.io.loadmat(path)
    gf = np.squeeze(mat["glottal_flow"]).astype(np.float64)
    return gf


def get_meta(path_contains=None):
    meta_list = []
    pairs = find_wav_mat_pairs(EGIFA_DIR)
    for wav, mat in tqdm(pairs, desc="EGIFA"):
        if path_contains is not None and path_contains not in str(wav):
            continue

        p = parse_filename(wav)
        fs, speech = _load_wav(wav)
        gf = _load_gf(mat)

        meta = {
            "wav": str(wav),
            "mat": str(mat),
            "name": p["name"],
            "f0_hz": p["hz"],
            "pressure_pa": p["pa"],
            "speech": speech,
            "fs": fs,
            "gf": gf,
        }

        meta_list.append(meta)

    return meta_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    meta = get_meta()
    print(f"Loaded {len(meta)} EGIFA recordings")

    # count number of vowel and speech
    n_vowel = sum(1 for m in meta if "vowel" in m["wav"])
    n_speech = len(meta) - n_vowel
    print(f"{n_vowel} vowel recordings, {n_speech} speech recordings")

    # plot speech and gf for a random recording
    m = meta[np.random.randint(len(meta))]
    t = np.arange(len(m["speech"])) / m["fs"]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, m["speech"])
    plt.title(f"Speech: {m['name']} ({m['f0_hz']} Hz, {m['pressure_pa']} Pa)")
    plt.subplot(2, 1, 2)
    plt.plot(t, m["gf"])
    plt.title("Glottal Flow")
    plt.tight_layout()
    plt.show()

# %%
from gci.lbgid import level_based_glottal_instant_detection


@__memory__.cache
def get_meta_gid():  # 5 min
    meta = get_meta()
    for m in tqdm(meta, desc="LGID"):
        gf = m["gf"]
        fs = m["fs"]

        instants, aux = level_based_glottal_instant_detection(
            gf, fs, return_aux=True
        )

        gci = instants[:, 0]
        gci_ms = gci * 1e3 / fs

        m["lgid_aux"] = aux

        m["instants"] = instants
        m["instants_ms"] = 1e3 * instants / fs
        m["period_samples"] = np.diff(gci)
        m["periods_ms"] = np.diff(gci_ms)

    return meta


if __name__ == "__main__":
    meta = get_meta_gid()
    print(f"Computed LGID for {len(meta)} EGIFA recordings")


# %%
@__memory__.cache
def get_meta_grouped():
    return gci_get_meta_grouped(get_meta_gid())


if __name__ == "__main__":
    meta = get_meta_grouped()
    print(f"Computed groups for {len(meta)} EGIFA recordings")


# %%
from fractions import Fraction

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly

from gci.meta import subgroups
from utils import __datadir__, __memory__
from utils.constants import MIN_NUM_PERIODS


def _split_idx_by_max_samples(idx, anchors, max_samples):
    """
      Yield contiguous chunks of anchor indices whose span in samples
      does not exceed `max_samples`.

      Parameters
      ----------
      idx : array-like of int
          Indices into `anchors`. Must be sorted ascending.
      anchors : array-like
          Anchor sample positions (e.g. GCI sample indices).
      max_samples : int or None
          Maximum allowed span:

              anchors[chunk[-1]] - anchors[chunk[0]] <= max_samples

          If None, yield entire idx.

      Border behaviour
      ----------------
      Chunks overlap by ONE anchor so that no period is lost when
      reconstructing waveform segments.

          [ a0 a1 a2 a3 | a3 a4 a5 | a5 a6 ... ]
            chunk1        chunk2      chunk3

      Meaning:

      - chunk2 starts at the LAST anchor of chunk1 (anchor duplicated,
    but inter-anchor periods remain disjoint).
      - Every inter-anchor interval (period) is covered exactly once.
      - No cycles are dropped between chunks.
      - Only anchors (not periods) are duplicated.

      Yields
      ------
      chunk : np.ndarray[int]
          Subarray of `idx`.
    """
    idx = np.asarray(idx, dtype=int)

    if max_samples is None:
        yield idx
        return

    n = len(idx)
    i = 0

    while i < n:
        start_anchor = anchors[idx[i]]
        j = i + 1

        # grow chunk until span exceeded
        while j < n:
            span = anchors[idx[j]] - start_anchor
            if span > max_samples:
                break
            j += 1

        if j == i + 1:
            # window too small to contain one period
            i += 1
            continue

        yield idx[i:j]

        if j >= n:
            break

        # overlap by one anchor to preserve periods
        i = j - 1


def smooth_and_dgf(v, delta_ms=0.1, k=2.0):
    """Compute smoothed versions gf and derive dgf

    See .chat:[Smoothing DGF]

    We smooth according to 0.1 msec temporal resolution of GCI events [Herbst et al. (2014); Orlikoff et al. (2012)].

    This corresponds to Gaussian smoothing of the glottal flow, followed by resampling to a target sampling density of ~2 samples per 0.1 msec window (ie k := 2.0)

    This happens to land at 20 kHz resampling rate, which is native EGIFA rate.

    delta_ms:
        physiologically meaningful temporal scale (≈ closure duration)

    k:
        desired sampling density per delta window, ie how many samples per effective time scale
    """
    t_samples = v["t_samples"]
    tau = v["tau"]
    gf = v["gf"]
    speech = v["speech"]
    fs = v["fs"]

    # smoothed Gaussian filter and derivative filter
    t_sec = delta_ms * 1e-3
    sigma = (t_sec * fs) / 2.355

    gf_smooth = gf_smooth = gaussian_filter1d(
        gf, sigma=sigma, order=0, mode="nearest"
    )
    dgf = (
        gaussian_filter1d(
            gf,
            sigma=sigma,
            order=1,
            mode="nearest",
        )
        * fs
    )

    # resampling target
    dt_target = t_sec / k
    fs_target = 1.0 / dt_target

    ratio = fs_target / fs
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    gf_rs = resample_poly(gf_smooth, up, down)
    dgf_rs = resample_poly(dgf, up, down)
    speech_rs = resample_poly(speech, up, down)

    fs_rs = fs * up / down

    # rebuild tau
    N_rs = len(dgf_rs)
    idx_rs = np.arange(N_rs) * (fs / fs_rs)
    tau_rs = np.interp(idx_rs, np.arange(len(tau)), tau)
    t_samples_rs = t_samples[0] + idx_rs

    s = {
        "t_samples": t_samples_rs,
        "tau": tau_rs,
        "gf": gf_rs,
        "dgf": dgf_rs,
        "speech": speech_rs,
        "fs": fs_rs,
    }

    return s


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
                v["mat"] = m["mat"]
                v["name"] = m["name"]
                v["f0_hz"] = m["f0_hz"]
                v["pressure_pa"] = m["pressure_pa"]
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

# %%
from prism.svi import pad_waveforms


def _shuffle_iterable(iterable, seed=42):
    """Shuffle a list generated by a generator function."""
    lst = list(iterable)
    rng = np.random.default_rng(seed)
    rng.shuffle(lst)
    return lst


def nanwhiten(s):
    mean = np.nanmean(s)
    std = np.nanstd(s)
    return (s - mean) / std


def get_data(
    n=None,
    offset=0,
    path_contains=None,
    width=None,
    dtype=np.float64,
    with_metadata=False,
    **smooth_dgf_kwargs,
):
    """Get data for learning test learning u'(t) from EGIFA

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
