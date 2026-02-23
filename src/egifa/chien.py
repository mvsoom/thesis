"""Run EGIFA MATLAB GIF methods on full files and export native frame outputs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from IPython.display import Audio, display
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.signal import resample_poly

from egifa.data import get_meta_grouped
from utils import __datadir__
from utils.audio import fit_affine_lag_nrmse, power_spectrum_db
from utils.matlab import (
    add_path_recursive,
    matlab_col,
    matlab_engine,
    matlab_row,
    numpy_vector,
)

MATLAB = matlab_engine()
add_path_recursive(__datadir__("EGIFA"))
MATLAB.eval("set(0,'DefaultFigureVisible','off');", nargout=0)
MATLAB.eval("warning('off','all');", nargout=0)

LPC_METHOD_TO_WPAR = {
    "cp": "cp",
    "wca1": "rgauss",
    "wca2": "ame",
}


def nan_dgf(length):
    return np.full(int(length), np.nan, dtype=np.float64)


def voiced_instants_from_meta(m):
    """Return voiced (gci, goi, group_id) arrays in source-file sample indices."""
    instants = np.asarray(m["instants"], dtype=np.int64)
    groups = np.asarray(m["groups"], dtype=np.int64)

    idx = np.flatnonzero(groups != -1)
    idx = idx[idx < len(instants)]

    gci = instants[idx, 0]
    goi = instants[idx, 1]
    gid = groups[idx]
    return gci, goi, gid


def to_fs_indices(indices, fs_src, fs_dst):
    indices = np.asarray(indices, dtype=np.float64).reshape(-1)
    t_sec = indices / float(fs_src)
    return np.rint(t_sec * float(fs_dst)).astype(np.int64)


def resample_file_to_egifa(m, fs_target=20000, speech_scale=1.0):
    """Resample file-level speech to EGIFA rate and map voiced GCIs/GOIs."""
    fs_src = float(m["fs"])
    speech_src = np.asarray(m["speech"], dtype=np.float64)

    # EGIFA original scripts scale by 1e6; keep configurable (default: no scaling)
    speech_dst = resample_poly(speech_src, int(fs_target), int(fs_src)) * float(
        speech_scale
    )

    gci_src, goi_src, gid = voiced_instants_from_meta(m)
    gci_dst = to_fs_indices(gci_src, fs_src, fs_target)
    goi_dst = to_fs_indices(goi_src, fs_src, fs_target)

    n = len(speech_dst)
    valid = (
        (0 <= gci_dst)
        & (gci_dst < n)
        & (0 <= goi_dst)
        & (goi_dst < n)
        & (gci_dst < goi_dst)
    )
    gci_dst = gci_dst[valid]
    goi_dst = goi_dst[valid]
    gid = gid[valid]

    return {
        "speech_20k": speech_dst,
        "fs_20k": float(fs_target),
        "gci_20k": gci_dst,
        "goi_20k": goi_dst,
        "group_id": gid,
        "gci_src": gci_src[valid],
        "goi_src": goi_src[valid],
        "fs_src": fs_src,
    }


def estimate_dgf_file(speech, fs, gci, goi, method="cp"):
    """Estimate full-file DGF with EGIFA MATLAB methods.

    Returns all-NaN on MATLAB failure.
    """
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    gci = np.asarray(gci, dtype=np.int64).reshape(-1)
    goi = np.asarray(goi, dtype=np.int64).reshape(-1)
    n = len(speech)

    if method == "null":
        rng = np.random.default_rng(0)
        return rng.standard_normal(n).astype(np.float64)

    if method == "iaif":
        try:
            uu = MATLAB.iaif(
                matlab_col(speech),
                float(fs),
                20.0,
                4.0,
                20.0,
                nargout=1,
            )
            uu = numpy_vector(uu).copy()
            if len(uu) != n:
                return nan_dgf(n)
            isnan_uu = np.isnan(uu)
            uu[isnan_uu] = 0.0
            return uu
        except Exception:
            return nan_dgf(n)

    try:
        par = MATLAB.projParam(LPC_METHOD_TO_WPAR[method], nargout=1)
        uu = MATLAB.weightedlpc3(
            matlab_col(speech),
            matlab_row(gci + 1),  # MATLAB 1-based
            matlab_row(goi + 1),  # MATLAB 1-based
            float(fs),
            par,
            nargout=1,
        )
        uu = numpy_vector(uu).copy()
        if len(uu) != n:
            return nan_dgf(n)
        isnan_uu = np.isnan(uu)
        uu[isnan_uu] = 0.0
        return uu
    except Exception:
        return nan_dgf(n)


def _sanitize_uu(uu, n):
    uu = numpy_vector(uu).copy()
    if len(uu) != n:
        return nan_dgf(n)
    uu[np.isnan(uu)] = 0.0
    return uu


def _analysis_window_bounds(method, n_samples, fs):
    n = int(n_samples)
    fs = float(fs)

    if method in ("iaif", "null"):
        win = int(np.floor(0.032 * fs))
        hop = int(np.floor(0.016 * fs))
        if n < win:
            return np.asarray([], dtype=np.int64), np.asarray(
                [], dtype=np.int64
            )
        nf = int(np.floor(1 + (n - win) / hop))
        starts = np.arange(nf, dtype=np.int64) * hop
        stops = starts + win
        return starts, stops

    nar = int(np.ceil(fs / 1000.0))
    wl = int(round(fs * 0.032))
    inc = int(round(fs * 0.016))
    start_1 = nar + 1
    stop_1 = n - wl - 1
    if start_1 > stop_1:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    tstart_1 = np.arange(start_1, stop_1 + 1, inc, dtype=np.int64)
    tend_1 = tstart_1 + wl
    starts = tstart_1 - 1
    stops = tend_1  # MATLAB inclusive end -> python exclusive end
    return starts, stops


def _starts_stops_from_matlab_intervals(T, n):
    T = np.asarray(T, dtype=np.float64)
    if T.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    T = np.atleast_2d(T)
    if T.shape[1] < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    starts = np.rint(T[:, 0]).astype(np.int64) - 1
    stops = np.rint(T[:, 1]).astype(
        np.int64
    )  # MATLAB inclusive -> python exclusive
    starts = np.clip(starts, 0, n)
    stops = np.clip(stops, 0, n)
    keep = starts < stops
    return starts[keep], stops[keep]


def _frames_from_analysis_windows(method, fs, starts, stops, uu_full):
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    stops = np.asarray(stops, dtype=np.int64).reshape(-1)
    uu_full = np.asarray(uu_full, dtype=np.float64).reshape(-1)
    n = len(uu_full)
    starts = np.clip(starts, 0, n)
    stops = np.clip(stops, 0, n)
    keep = starts < stops
    starts = starts[keep]
    stops = stops[keep]

    nf = len(starts)
    frames = []
    for i in range(nf):
        a = int(starts[i])
        b = int(stops[i])
        row = uu_full[a:b].copy()

        frames.append(
            {
                "frame_index": i,
                "method": method,
                "fs": float(fs),
                "start_20k": a,
                "stop_20k": b,
                "start_sec": float(a / fs),
                "stop_sec": float(b / fs),
                "t_samples_20k": np.arange(a, b, dtype=np.int64),
                "uu": row,
            }
        )
    return frames


def estimate_dgf_file_analysis_frames(speech, fs, gci, goi, method="cp"):
    """Estimate full-file DGF and analysis-window frame estimates.

    Frames correspond to the exact analysis windows used by MATLAB.
    """
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    gci = np.asarray(gci, dtype=np.int64).reshape(-1)
    goi = np.asarray(goi, dtype=np.int64).reshape(-1)
    n = len(speech)

    if method == "null":
        rng = np.random.default_rng(0)
        uu = rng.standard_normal(n).astype(np.float64)
        starts, stops = _analysis_window_bounds(method, n, fs)
        frames = _frames_from_analysis_windows(method, fs, starts, stops, uu)
        return uu, frames

    try:
        if method == "iaif":
            uu, _, Ts = MATLAB.iaif(
                matlab_col(speech),
                float(fs),
                20.0,
                4.0,
                20.0,
                nargout=3,
            )
            uu = _sanitize_uu(uu, n)
            starts, stops = _starts_stops_from_matlab_intervals(Ts, n)
            frames = _frames_from_analysis_windows(
                method, fs, starts, stops, uu
            )
            return uu, frames

        par = MATLAB.projParam(LPC_METHOD_TO_WPAR[method], nargout=1)
        uu, _, _, _, _, _, T = MATLAB.weightedlpc3(
            matlab_col(speech),
            matlab_row(gci + 1),  # MATLAB 1-based
            matlab_row(goi + 1),  # MATLAB 1-based
            float(fs),
            par,
            nargout=7,
        )
        uu = _sanitize_uu(uu, n)
        starts, stops = _starts_stops_from_matlab_intervals(T, n)
        frames = _frames_from_analysis_windows(method, fs, starts, stops, uu)
        return uu, frames
    except Exception:
        uu = nan_dgf(n)
        starts, stops = _analysis_window_bounds(method, n, fs)
        frames = _frames_from_analysis_windows(method, fs, starts, stops, uu)
        return uu, frames


def lpcifilt_piecewise_bounds(starts_1, n_samples):
    """Return disjoint lpcifilt fade=0 output ownership segments.

    `starts_1` is the 1-based start index used as `t` in MATLAB `lpcifilt`.
    Returns 0-based [start, stop) bounds.
    """
    starts_1 = np.asarray(starts_1, dtype=np.int64).reshape(-1)
    n = int(n_samples)
    if len(starts_1) == 0 or n <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    if len(starts_1) == 1:
        return np.asarray([0], dtype=np.int64), np.asarray([n], dtype=np.int64)

    starts = np.empty(len(starts_1), dtype=np.int64)
    stops = np.empty(len(starts_1), dtype=np.int64)

    starts[0] = 0
    stops[0] = int(np.ceil(starts_1[1] - 1))

    for i in range(1, len(starts_1) - 1):
        starts[i] = int(np.ceil(starts_1[i])) - 1
        stops[i] = int(np.ceil(starts_1[i + 1] - 1))

    starts[-1] = int(np.ceil(starts_1[-1])) - 1
    stops[-1] = n

    starts = np.clip(starts, 0, n)
    stops = np.clip(stops, 0, n)
    keep = (0 <= starts) & (starts < stops) & (stops <= n)
    return starts[keep], stops[keep]


def native_frame_bounds(method, n_samples, fs=20000):
    """Return method-native synthesis frame boundaries on a file signal.

    Returns (starts, stops) in 0-based [start, stop) convention.
    """
    n = int(n_samples)
    fs = float(fs)

    if method in ("iaif", "null"):
        win = int(np.floor(0.032 * fs))
        hop = int(np.floor(0.016 * fs))
        if n < win:
            return np.asarray([], dtype=np.int64), np.asarray(
                [], dtype=np.int64
            )
        nf = int(np.floor(1 + (n - win) / hop))
        starts_1 = 1 + np.arange(nf, dtype=np.int64) * hop
        return lpcifilt_piecewise_bounds(starts_1, n_samples=n)

    # weightedlpc3/CP/WCA: analysis starts from weightedlpc3(), synthesis from lpcifilt()
    nar = int(np.ceil(fs / 1000.0))
    wl = int(round(fs * 0.032))
    inc = int(round(fs * 0.016))
    start_1 = nar + 1
    stop_1 = n - wl - 1
    if start_1 > stop_1:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    starts_1 = np.arange(start_1, stop_1 + 1, inc, dtype=np.int64)
    return lpcifilt_piecewise_bounds(starts_1, n_samples=n)


def slice_dgf_to_native_frames(
    uu,
    method,
    fs=20000,
    fs_src=44100,
    gci_20k=None,
    group_id=None,
):
    """Slice full-file DGF into method-native frames with absolute indices."""
    uu = np.asarray(uu, dtype=np.float64).reshape(-1)
    starts, stops = native_frame_bounds(method, len(uu), fs=fs)

    frames = []
    for i, (a, b) in enumerate(zip(starts, stops)):
        t = np.arange(a, b, dtype=np.int64)
        frame_groups = np.asarray([], dtype=np.int64)
        if gci_20k is not None and group_id is not None:
            gci_20k = np.asarray(gci_20k, dtype=np.int64)
            group_id = np.asarray(group_id, dtype=np.int64)
            k = min(len(gci_20k), len(group_id))
            inside = (a <= gci_20k[:k]) & (gci_20k[:k] < b)
            frame_groups = np.unique(group_id[:k][inside])

        frames.append(
            {
                "frame_index": i,
                "method": method,
                "fs": float(fs),
                "start_20k": int(a),
                "stop_20k": int(b),
                "start_sec": float(a / fs),
                "stop_sec": float(b / fs),
                "t_samples_20k": t,
                "start_src": int(np.rint(a * fs_src / fs)),
                "stop_src": int(np.rint(b * fs_src / fs)),
                "group_ids": frame_groups,
                "uu": uu[a:b],
            }
        )
    return frames


def estimate_file_frames_from_meta(
    m, method="cp", fs_target=20000, speech_scale=1.0
):
    """Full pipeline for one file metadata dict from `get_meta_grouped()`.

    Output:
      - file-level DGF estimate (`uu`)
      - method-native frame list with absolute indices for syncing.
    """
    payload = resample_file_to_egifa(
        m, fs_target=fs_target, speech_scale=speech_scale
    )
    uu, frames = estimate_dgf_file_analysis_frames(
        payload["speech_20k"],
        payload["fs_20k"],
        payload["gci_20k"],
        payload["goi_20k"],
        method=method,
    )

    gci_20k = np.asarray(payload["gci_20k"], dtype=np.int64)
    group_id = np.asarray(payload["group_id"], dtype=np.int64)
    k = min(len(gci_20k), len(group_id))
    for f in frames:
        a = int(f["start_20k"])
        b = int(f["stop_20k"])
        f["start_src"] = int(np.rint(a * payload["fs_src"] / payload["fs_20k"]))
        f["stop_src"] = int(np.rint(b * payload["fs_src"] / payload["fs_20k"]))
        inside = (a <= gci_20k[:k]) & (gci_20k[:k] < b)
        f["group_ids"] = np.unique(group_id[:k][inside])

    return {
        "wav": m["wav"],
        "name": m["name"],
        "f0_hz": m["f0_hz"],
        "pressure_pa": m["pressure_pa"],
        "method": method,
        "fs_20k": payload["fs_20k"],
        "fs_src": payload["fs_src"],
        "uu": uu,
        "frames": frames,
    }


def estimate_collection_frames(
    path_contains=None,
    methods=("cp", "wca1", "wca2", "iaif"),
    speech_scale=1.0,
):
    for m in get_meta_grouped():
        if path_contains is not None and path_contains not in m["wav"]:
            continue
        for method in methods:
            yield estimate_file_frames_from_meta(
                m, method=method, speech_scale=speech_scale
            )


def _frame_t_samples_src(frame, fs_src, fs_frame):
    return np.asarray(frame["t_samples_20k"], dtype=np.float64) * (
        float(fs_src) / float(fs_frame)
    )


def _frame_inside_group(t_frame, group_t):
    eps = 1e-9
    return (t_frame[0] >= group_t[0] - eps) and (
        t_frame[-1] <= group_t[-1] + eps
    )


def _interp_group_field(v, key, t_frame, dtype):
    t_group = np.asarray(v["smooth"]["t_samples"], dtype=np.float64)
    y_group = np.asarray(v["smooth"][key], dtype=np.float64)
    return np.interp(t_frame, t_group, y_group).astype(dtype)


def _make_run_frame(v, mf, method, fs_src, fs_frame, dtype):
    t = _frame_t_samples_src(mf, fs_src=fs_src, fs_frame=fs_frame).astype(dtype)

    speech = _interp_group_field(v, "speech", t, dtype=dtype)
    gf = _interp_group_field(v, "gf", t, dtype=dtype)
    dgf = _interp_group_field(v, "dgf", t, dtype=dtype)
    tau = _interp_group_field(v, "tau", t, dtype=dtype)
    dgf_est = np.asarray(mf["uu"], dtype=dtype)

    t_ms = 1e3 * t / float(v["fs"])

    t_min, t_max = t[0], t[-1]
    loc = np.where((t_min <= v["gci"]) & (v["gci"] <= t_max))[0]

    gci = v["gci"][loc]
    goi = v["goi"][loc]
    oq = v["oq"][loc[:-1]]
    periods_ms = v["periods_ms"][loc[:-1]]

    return {
        "fs": float(fs_frame),
        "t_ms": t_ms,
        "t_samples": t,
        "tau": tau,
        "speech": speech,
        "gf": gf,
        "dgf": dgf,
        "dgf_est": dgf_est,
        "gci": gci,
        "goi": goi,
        "oq": oq,
        "periods_ms": periods_ms,
        "frame_index": int(mf["frame_index"]),
        "method": method,
    }


def get_voiced_runs_matlab(
    groups,
    method,  # one of "null", "iaif", "cp", "wca1", "wca2"
    fs_target=20000,
    speech_scale=1.0,
    dtype=np.float64,
):
    groups_by_wav = defaultdict(list)
    for v in groups:
        groups_by_wav[v["wav"]].append(v)

    meta = get_meta_grouped()
    for m in meta:
        if m["wav"] not in groups_by_wav:
            continue

        ret = estimate_file_frames_from_meta(
            m,
            method=method,
            fs_target=fs_target,
            speech_scale=speech_scale,
        )
        file_frames = ret["frames"]
        fs_src = float(ret["fs_src"])
        fs_frame = float(ret["fs_20k"])
        file_frames_src = [
            (mf, _frame_t_samples_src(mf, fs_src=fs_src, fs_frame=fs_frame))
            for mf in file_frames
        ]

        for v in groups_by_wav[m["wav"]]:
            t_group = np.asarray(v["smooth"]["t_samples"], dtype=np.float64)
            if len(t_group) == 0:
                continue

            for mf, t_frame in file_frames_src:
                if len(t_frame) == 0:
                    continue
                if not _frame_inside_group(t_frame, t_group):
                    continue

                f = _make_run_frame(
                    v,
                    mf,
                    method=method,
                    fs_src=fs_src,
                    fs_frame=fs_frame,
                    dtype=dtype,
                )
                yield {"group": v, "frame": f}


def _as_ms(sample_idx, fs):
    return 1e3 * np.asarray(sample_idx, dtype=np.float64) / float(fs)


def _load_full_file_context(group):
    import scipy.io
    import scipy.io.wavfile

    fs, speech = scipy.io.wavfile.read(group["wav"])
    if speech.ndim > 1:
        speech = speech[:, 0]
    speech = speech.astype(np.float64)
    peak = np.max(np.abs(speech))
    if peak > 0:
        speech /= peak

    mat = scipy.io.loadmat(group["mat"])
    gf = np.squeeze(mat["glottal_flow"]).astype(np.float64)
    return fs, speech, gf


def _infer_pitch_from_frame(frame):
    periods_ms = np.asarray(frame.get("periods_ms", []), dtype=np.float64)
    periods_ms = periods_ms[np.isfinite(periods_ms) & (periods_ms > 0)]
    if len(periods_ms) == 0:
        return np.nan
    return 1000.0 / np.mean(periods_ms)


def _align_true_to_inferred(true_dgf, inferred_excitation, fs, maxlag):
    true_dgf = np.asarray(true_dgf, dtype=np.float64)
    inferred_excitation = np.asarray(inferred_excitation, dtype=np.float64)
    n = int(min(len(true_dgf), len(inferred_excitation)))
    true_dgf = true_dgf[:n]
    inferred_excitation = inferred_excitation[:n]
    if n == 0:
        return np.asarray([], dtype=np.float64), np.nan

    try:
        best, _ = fit_affine_lag_nrmse(
            true_dgf, inferred_excitation, maxlag=maxlag
        )
        aligned_true_dgf = np.asarray(best["aligned"], dtype=np.float64)
        lag_est_ms = 1e3 * float(best["lag"]) / float(fs)
    except Exception:
        aligned_true_dgf = np.full_like(true_dgf, np.nan)
        lag_est_ms = np.nan

    return aligned_true_dgf, lag_est_ms


def _aligned_pair_for_spectrum(true_dgf, inferred_signal, maxlag):
    true_dgf = np.asarray(true_dgf, dtype=np.float64)
    inferred_signal = np.asarray(inferred_signal, dtype=np.float64)
    n = int(min(len(true_dgf), len(inferred_signal)))
    true_dgf = true_dgf[:n]
    inferred_signal = inferred_signal[:n]
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray(
            [], dtype=np.float64
        )

    try:
        # Keep inferred signal/noise on their native scale and transform truth.
        best, _ = fit_affine_lag_nrmse(true_dgf, inferred_signal, maxlag=maxlag)
        aligned_true_dgf = np.asarray(best["aligned"], dtype=np.float64)
        mask = np.isfinite(aligned_true_dgf) & np.isfinite(inferred_signal)
        if np.any(mask):
            return aligned_true_dgf[mask], inferred_signal[mask]
    except Exception:
        pass

    return true_dgf, inferred_signal


def post_process_run(run):
    group = run["group"]
    frame = run["frame"]

    pitch_true = _infer_pitch_from_frame(frame)
    oq = np.asarray(frame.get("oq", []), dtype=np.float64)
    oq_true = float(np.mean(oq)) if len(oq) else np.nan

    inferred_signal = np.asarray(frame["dgf_est"], dtype=np.float64)
    true_dgf = np.asarray(frame["dgf"], dtype=np.float64)

    n = int(min(len(inferred_signal), len(true_dgf)))
    inferred_signal = inferred_signal[:n]
    true_dgf = true_dgf[:n]

    fs = float(frame["fs"])
    dt = 1.0 / fs
    if np.isfinite(pitch_true) and pitch_true > 0:
        maxlag = max(1, int(0.5 / pitch_true / dt))  # half pitch period
    else:
        maxlag = max(1, int(0.002 / dt))

    try:
        best, _ = fit_affine_lag_nrmse(inferred_signal, true_dgf, maxlag=maxlag)
        excitation_aligned_nrmse = float(best["nrmse"])
    except Exception:
        excitation_aligned_nrmse = np.nan

    _, lag_est = _align_true_to_inferred(true_dgf, inferred_signal, fs, maxlag)

    return {
        "wav": group["wav"],
        "name": group["name"],
        "f0_hz_nominal": group["f0_hz"],
        "pressure_pa": group["pressure_pa"],
        "voiced_group": group["group"],
        "frame_index": frame["frame_index"],
        "oq_true": oq_true,
        "pitch_true": pitch_true,
        "excitation_aligned_nrmse": excitation_aligned_nrmse,
        "lag_est": lag_est,
        "affine_lag_a": best["a"],
        "affine_lag_b": best["b"],
    }


def plot_run(run):
    group = run["group"]
    frame = run["frame"]

    fs_model = float(frame["fs"])

    t_samples = np.asarray(frame["t_samples"], dtype=np.float64)
    speech = np.asarray(frame["speech"], dtype=np.float64)
    gf = np.asarray(frame["gf"], dtype=np.float64)
    dgf = np.asarray(frame["dgf"], dtype=np.float64)
    dgf_est = np.asarray(frame["dgf_est"], dtype=np.float64)

    n = int(min(len(t_samples), len(speech), len(gf), len(dgf), len(dgf_est)))
    t_samples = t_samples[:n]
    speech = speech[:n]
    gf = gf[:n]
    dgf = dgf[:n]
    inferred_signal = dgf_est[:n]
    frame_duration_ms = 1e3 * len(speech) / fs_model

    true_pitch = _infer_pitch_from_frame(frame)
    dt = 1.0 / fs_model
    if np.isfinite(true_pitch) and true_pitch > 0:
        maxlag = max(1, int(0.5 / true_pitch / dt))  # half pitch period
    else:
        maxlag = max(1, int(0.002 / dt))

    aligned_true_dgf, align_lag_ms = _align_true_to_inferred(
        dgf, inferred_signal, fs_model, maxlag
    )

    fs_file, speech_full, gf_full = _load_full_file_context(group)
    file_t_ms = _as_ms(np.arange(len(speech_full)), fs_file)
    fs_abs = float(group["fs"])
    if not np.isfinite(fs_abs) or fs_abs <= 0:
        fs_abs = float(fs_file)
    if not np.isclose(fs_abs, fs_file, rtol=0.0, atol=1e-9):
        fs_abs = float(fs_file)
    t_ms = _as_ms(t_samples, fs_abs)

    group_t_ms = _as_ms(group["smooth"]["t_samples"], fs_abs)
    group_start_ms = float(group_t_ms[0])
    group_end_ms = float(group_t_ms[-1])

    frame_start_ms = float(t_ms[0]) if len(t_ms) else np.nan
    frame_end_ms = float(t_ms[-1]) if len(t_ms) else np.nan

    frame_gci_ms = (
        _as_ms(frame["gci"], fs_abs) if len(frame["gci"]) else np.asarray([])
    )
    frame_goi_ms = (
        _as_ms(frame["goi"], fs_abs) if len(frame["goi"]) else np.asarray([])
    )

    dgf_spec, inferred_signal_spec = _aligned_pair_for_spectrum(
        dgf, inferred_signal, maxlag
    )

    f_signal, p_signal_db = power_spectrum_db(inferred_signal_spec, fs_model)
    mask_signal = (
        np.isfinite(f_signal) & np.isfinite(p_signal_db) & (f_signal > 0)
    )
    f_signal_plot = f_signal[mask_signal]
    p_signal_db_plot = p_signal_db[mask_signal]
    f_signal_true, p_signal_true_db = power_spectrum_db(dgf_spec, fs_model)
    mask_signal_true = (
        np.isfinite(f_signal_true)
        & np.isfinite(p_signal_true_db)
        & (f_signal_true > 0)
    )
    f_signal_true_plot = f_signal_true[mask_signal_true]
    p_signal_true_db_plot = p_signal_true_db[mask_signal_true]

    colors = qualitative.Plotly
    c_truth = colors[0]  # primary theme blue for ground truth
    c_inferred = colors[2]

    signal_xmax = 5000.0
    if len(f_signal_plot) and np.isfinite(np.max(f_signal_plot)):
        signal_xmax = max(signal_xmax, float(np.max(f_signal_plot)))
    if len(f_signal_true_plot) and np.isfinite(np.max(f_signal_true_plot)):
        signal_xmax = max(signal_xmax, float(np.max(f_signal_true_plot)))
    signal_xmax = max(50.5, signal_xmax)

    harmonics = np.asarray([], dtype=np.float64)
    if np.isfinite(true_pitch) and true_pitch > 0:
        n_harm = int(np.floor(5000.0 / true_pitch))
        if n_harm >= 1:
            harmonics = true_pitch * np.arange(1, n_harm + 1)
            harmonics = harmonics[
                np.isfinite(harmonics)
                & (harmonics >= 50.0)
                & (harmonics <= min(5000.0, signal_xmax))
            ]

    n_rows = 6

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        row_heights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vertical_spacing=0.05,
        subplot_titles=[
            r"Data $x(t)$ in file context",
            r"True $u(t)$ in file context",
            r"Data $x(t)$",
            r"True $u(t)$",
            r"Estimated signal $s(t)$ vs true $u'(t)$",
            r"Estimated signal spectral envelope $|S(f)|^2$ vs true $|U'(f)|$",
        ],
    )
    fig.update_annotations(yshift=6)

    fig.add_trace(
        go.Scatter(
            x=file_t_ms,
            y=speech_full,
            mode="lines",
            name="recording x(t)",
            line=dict(color=c_truth),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=file_t_ms,
            y=gf_full,
            mode="lines",
            name="recording u(t)",
            line=dict(color=c_truth),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=speech,
            mode="lines",
            name="frame x(t)",
            line=dict(color=c_truth),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=gf,
            mode="lines",
            name="frame u(t)",
            line=dict(color=c_truth),
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=aligned_true_dgf,
            mode="lines",
            name="aligned true u'(t)",
            line=dict(color=c_truth),
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=inferred_signal,
            mode="lines",
            name="inferred signal s(t)",
            line=dict(color=c_inferred),
        ),
        row=5,
        col=1,
    )

    def add_envelope_decorations(
        row, x_max, y_series, show_harmonic_legend=False
    ):
        finite_series = []
        for y in y_series:
            y_arr = np.asarray(y, dtype=np.float64)
            y_arr = y_arr[np.isfinite(y_arr)]
            if len(y_arr):
                finite_series.append(y_arr)
        if not finite_series:
            return
        y_peak = max(float(np.max(y_arr)) for y_arr in finite_series)
        y_floor = min(float(np.min(y_arr)) for y_arr in finite_series)
        y_span = y_peak - y_floor
        y_top = y_peak + 0.1 * (y_span if y_span > 0 else max(abs(y_peak), 1.0))

        if np.isfinite(true_pitch) and 50.0 <= true_pitch <= min(5000.0, x_max):
            fig.add_vline(
                x=float(true_pitch),
                line_color=c_truth,
                line_dash="dash",
                opacity=0.9,
                row=row,
                col=1,
            )

        if len(harmonics) > 0:
            fig.add_trace(
                go.Scatter(
                    x=harmonics,
                    y=np.full_like(harmonics, y_top),
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=6,
                        color=c_truth,
                    ),
                    name="harmonics kF0 (up to 5 kHz)",
                    cliponaxis=False,
                    opacity=0.75,
                    showlegend=show_harmonic_legend,
                ),
                row=row,
                col=1,
            )

        fig.add_vrect(
            x0=50.0,
            x1=200.0,
            fillcolor="rgba(120,120,120,0.20)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )
        if x_max > 5000.0:
            fig.add_vrect(
                x0=5000.0,
                x1=x_max,
                fillcolor="rgba(120,120,120,0.20)",
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=f_signal_true_plot,
            y=p_signal_true_db_plot,
            mode="lines",
            name="aligned true spectrum |U'(f)|^2",
            line=dict(color=c_truth),
            showlegend=False,
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=f_signal_plot,
            y=p_signal_db_plot,
            mode="lines",
            name="inferred signal spectrum |S(f)|^2",
            line=dict(color=c_inferred),
            showlegend=False,
        ),
        row=6,
        col=1,
    )
    add_envelope_decorations(
        row=6,
        x_max=signal_xmax,
        y_series=[p_signal_db_plot, p_signal_true_db_plot],
        show_harmonic_legend=True,
    )

    for row in (1, 2):
        fig.add_vrect(
            x0=group_start_ms,
            x1=group_end_ms,
            fillcolor="rgba(0,150,255,0.20)",
            line_width=1,
            line_color="rgba(0,150,255,0.5)",
            row=row,
            col=1,
        )

    if np.isfinite(frame_start_ms) and np.isfinite(frame_end_ms):
        for row in (1, 2):
            fig.add_vrect(
                x0=frame_start_ms,
                x1=frame_end_ms,
                fillcolor="rgba(255,0,0,0.20)",
                line_width=1,
                line_color="rgba(255,0,0,0.5)",
                row=row,
                col=1,
            )

    for x in frame_gci_ms:
        fig.add_vline(
            x=float(x),
            line_color="green",
            line_width=1,
            opacity=0.35,
            row=4,
            col=1,
        )
    for x in frame_goi_ms:
        fig.add_vline(
            x=float(x),
            line_color="purple",
            line_width=1,
            opacity=0.25,
            row=4,
            col=1,
        )

    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x3", row=4, col=1)
    fig.update_xaxes(matches="x3", row=5, col=1)

    if np.isfinite(frame_start_ms) and np.isfinite(frame_end_ms):
        for row in (3, 4, 5):
            fig.update_xaxes(
                range=[frame_start_ms, frame_end_ms], row=row, col=1
            )

    fig.update_xaxes(
        title_text=r"$t$ (ms)",
        title_standoff=2,
        automargin=True,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text=r"$t$ (ms)",
        title_standoff=2,
        automargin=True,
        row=5,
        col=1,
    )
    fig.update_xaxes(
        title_text=r"frequency $f$ (Hz)",
        type="log",
        range=[np.log10(50.0), np.log10(signal_xmax)],
        title_standoff=2,
        automargin=True,
        row=6,
        col=1,
    )

    fig.update_yaxes(title_text="amplitude", row=1, col=1)
    fig.update_yaxes(title_text="amplitude", row=2, col=1)
    fig.update_yaxes(title_text="amplitude", row=3, col=1)
    fig.update_yaxes(title_text="amplitude", row=4, col=1)
    fig.update_yaxes(title_text="amplitude", row=5, col=1)
    fig.update_yaxes(title_text="power (dB)", row=6, col=1)

    if np.isfinite(align_lag_ms):
        fig.add_annotation(
            text=rf"$\Delta_{{\mathrm{{est}}}}={align_lag_ms:+.2f}\,\mathrm{{ms}}$",
            x=1.0,
            y=0.0,
            xref="x4 domain",  # subplot 4 x-domain
            yref="y4 domain",  # subplot 4 y-domain
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.75)",
        )

    filename = group["wav"].split("/")[-1]
    main_title = (
        "EGIFA | "
        f"`{filename}` | "
        f"`group {group['group']}` | "
        f"`frame {frame['frame_index']}`"
    )

    fig.update_layout(
        height=240 * n_rows,
        hovermode="x unified",
        title=dict(
            text=main_title,
            pad=dict(b=14),
        ),
        margin=dict(r=60, b=70),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.035,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    fig.show()

    print(
        f"Audio preview: full file ({len(speech_full) / fs_file:.2f} s) and "
        f"frame ({frame_duration_ms:.1f} ms)"
    )
    display(Audio(speech_full, rate=int(round(fs_file))))
    display(Audio(speech, rate=int(round(fs_model))))
