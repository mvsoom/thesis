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
from gci.lgid import level_based_glottal_instant_detection


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