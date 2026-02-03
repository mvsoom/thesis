from pathlib import Path

import numpy as np
import scipy.io
import scipy.io.wavfile

from utils import __datadir__, __memory__, pyglottal
from utils.audio import frame_signal

EGIFA_DIR = __datadir__("EGIFA")
VOWEL_DIR = EGIFA_DIR / "vowel"

WINDOW_MS = 32.0
HOP_MS = 16.0
TOL_FRAC = 0.2
SEED = 0


def _list_vowel_stems():
    stems = []
    for wav_path in sorted(VOWEL_DIR.glob("*.wav")):
        mat_path = wav_path.with_suffix(".mat")
        if mat_path.exists():
            stems.append(wav_path.with_suffix(""))
    if not stems:
        raise FileNotFoundError(f"No EGIFA vowels found under {VOWEL_DIR}")
    return stems


def _parse_f0_from_stem(stem: Path) -> float:
    for token in stem.name.split("_"):
        if token.endswith("hz"):
            return float(token[:-2])
    raise ValueError(f"Could not parse f0 from filename: {stem.name}")


def _load_vowel(stem: Path):
    fs, s = scipy.io.wavfile.read(stem.with_suffix(".wav"))
    if s.ndim > 1:
        s = s[:, 0]
    s = s.astype(np.float32)
    s_max = np.max(np.abs(s))
    if s_max > 0:
        s = s / s_max

    mat = scipy.io.loadmat(stem.with_suffix(".mat"))
    gf = np.squeeze(mat["glottal_flow"]).astype(np.float32)
    gf_max = np.max(np.abs(gf))
    if gf_max > 0:
        gf = gf / gf_max

    return fs, s, gf


def _tau_from_gci(frame_len: int, gci_idx):
    gci = np.asarray(gci_idx, dtype=int)
    gci = gci[(gci >= 0) & (gci < frame_len)]
    gci = np.unique(gci)

    tau_full = np.full(frame_len, np.nan, dtype=np.float32)
    mask = np.zeros(frame_len, dtype=bool)

    if len(gci) < 2:
        return tau_full, mask

    for k in range(len(gci) - 1):
        a = gci[k]
        b = gci[k + 1]
        if b <= a:
            continue
        idx = np.arange(a, b, dtype=int)
        tau_full[idx] = k + (idx - a) / (b - a)
        mask[idx] = True

    tau_full[gci[-1]] = len(gci) - 1
    mask[gci[-1]] = True

    tau_masked = tau_full.copy()
    tau_masked[~mask] = np.nan
    return tau_masked, mask


def _apply_nan_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32).copy()
    out[~mask] = np.nan
    return out


def estimate_oq(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float32)
    valid = np.isfinite(u)
    if not np.any(valid):
        return np.nan
    u_max = np.nanmax(u)
    u_min = np.nanmin(u)
    height = u_min + 0.05 * (u_max - u_min)
    return float(np.mean(u[valid] > height))


@__memory__.cache
def _collect_frames():
    frames = []

    for stem in _list_vowel_stems():
        fs, s, gf = _load_vowel(stem)
        f0 = _parse_f0_from_stem(stem)

        period_samples = fs / f0
        frame_len = int(round(WINDOW_MS * fs / 1000.0))
        hop = int(round(HOP_MS * fs / 1000.0))

        frames_u = frame_signal(gf, frame_len, hop)

        if len(s) < frame_len:
            starts = np.array([0], dtype=int)
        else:
            starts = np.arange(0, len(s) - frame_len + 1, hop, dtype=int)
        ends = starts + frame_len

        gci_est = pyglottal.quick_gci(
            s,
            fs=fs,
            fmin=20,
            fmax=400,
            theta=-np.pi / 2,
            reps=2,
        )
        gci_est = np.asarray(gci_est, dtype=int)

        left = np.searchsorted(gci_est, starts, side="left")
        right = np.searchsorted(gci_est, ends, side="left")
        counts = right - left

        expected = int(round((WINDOW_MS / 1000.0) * f0))
        candidate_idx = np.where(counts == expected)[0]

        for idx in candidate_idx:
            gci_idx = gci_est[left[idx] : right[idx]]
            if expected > 1:
                if len(gci_idx) < 2:
                    continue
                diffs = np.diff(gci_idx)
                if not np.all(
                    np.abs(diffs - period_samples) <= TOL_FRAC * period_samples
                ):
                    continue

            start = starts[idx]
            est = gci_idx - start
            tau_masked, mask = _tau_from_gci(frame_len, est)
            if not np.any(mask):
                continue

            u_tau = _apply_nan_mask(frames_u[idx], mask)
            t_abs = 1e3 * (start + np.arange(frame_len)) / fs
            frames.append(
                dict(
                    stem=stem,
                    t=t_abs.astype(np.float32),
                    tau=tau_masked,
                    u=u_tau,
                    oq=estimate_oq(u_tau),
                )
            )

    if not frames:
        empty = np.empty((0, 0), dtype=np.float32)
        return dict(
            frames=[],
            train_idx=np.array([], dtype=int),
            test_idx=np.array([], dtype=int),
            train_tau=empty,
            train_u=empty,
            train_oq=np.array([], dtype=np.float32),
            test_tau=empty,
            test_u=empty,
            test_oq=np.array([], dtype=np.float32),
        )

    selected_tau = np.stack([f["tau"] for f in frames])
    selected_u = np.stack([f["u"] for f in frames])
    selected_oq = np.asarray([f["oq"] for f in frames], dtype=np.float32)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(frames))
    split = int(0.8 * len(frames))
    train_idx = perm[:split]
    test_idx = perm[split:]

    return dict(
        frames=frames,
        train_idx=train_idx,
        test_idx=test_idx,
        train_tau=selected_tau[train_idx],
        train_u=selected_u[train_idx],
        train_oq=selected_oq[train_idx],
        test_tau=selected_tau[test_idx],
        test_u=selected_u[test_idx],
        test_oq=selected_oq[test_idx],
    )


def get_train_data(n=None):
    """Return (tau, u, oq) for training; optionally take first n samples."""
    cache = _collect_frames()
    tau = cache["train_tau"]
    u = cache["train_u"]
    oq = cache["train_oq"]
    if n is None:
        return tau, u  # , oq FIXME
    return tau[:n], u[:n]  # , oq[:n] FIXME


def get_test_data(n=None):
    """Return (tau, u, oq) for testing; optionally take first n samples."""
    cache = _collect_frames()
    tau = cache["test_tau"]
    u = cache["test_u"]
    oq = cache["test_oq"]
    if n is None:
        return tau, u  # , oq # FIXME
    return tau[:n], u[:n]  # , oq[:n] FIXME


__all__ = ["get_train_data", "get_test_data"]
