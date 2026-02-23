# %%
import warnings

import numpy as np

from egifa.data import get_voiced_meta
from utils.audio import frame_signal


def get_voiced_runs(
    path_contains=None,
    frame_len_msec=128.0,
    hop_msec=64.0,
    num_vi_restarts=1,
    dtype=np.float64,
    **smooth_dgf_kwargs,
):
    for v in get_voiced_meta(path_contains=path_contains, **smooth_dgf_kwargs):
        t = v["smooth"]["t_samples"].astype(dtype)
        x = v["smooth"]["speech"].astype(dtype)
        u = v["smooth"]["gf"].astype(dtype)
        du = v["smooth"]["dgf"].astype(dtype)
        tau = v["smooth"]["tau"].astype(dtype)
        assert len(x) == len(u) == len(du) == len(t) == len(tau)

        fs = float(v["fs"])
        fs_smooth = float(v["smooth"]["fs"])
        frame_len = int(frame_len_msec / 1000 * fs_smooth)
        hop = int(hop_msec / 1000 * fs_smooth)

        if len(t) < frame_len:  # ditch voiced groups shorter than one frame
            continue

        t_frames = frame_signal(t, frame_len, hop)
        x_frames = frame_signal(x, frame_len, hop)
        u_frames = frame_signal(u, frame_len, hop)
        du_frames = frame_signal(du, frame_len, hop)
        tau_frames = frame_signal(tau, frame_len, hop)

        for frame_index, (t, x, u, du, tau) in enumerate(
            zip(t_frames, x_frames, u_frames, du_frames, tau_frames)
        ):
            t_ms = 1e3 * t / fs

            t_min, t_max = t[0], t[-1]
            loc = np.where((t_min <= v["gci"]) & (v["gci"] <= t_max))[0]

            gci = v["gci"][loc]
            goi = v["goi"][loc]
            oq = v["oq"][loc[:-1]]
            periods_ms = v["periods_ms"][loc[:-1]]

            for restart_index in range(num_vi_restarts):
                f = {
                    "fs": fs_smooth,
                    "t_ms": t_ms,
                    "t_samples": t,
                    "tau": tau,
                    "speech": x,
                    "gf": u,
                    "dgf": du,
                    "gci": gci,
                    "goi": goi,
                    "oq": oq,
                    "periods_ms": periods_ms,
                    "frame_index": frame_index,
                    "restart_index": restart_index,
                }

                yield {"group": v, "frame": f}


if __name__ == "__main__":
    runs = list(get_voiced_runs())
    print("Total runs:", len(runs))

    x = np.vstack([r["frame"]["speech"] for r in runs])
    print("Data shape:", x.shape)


# %%


import numpy as np
import plotly.graph_objects as go
import scipy.io
import scipy.io.wavfile
from IPython.display import Audio, display
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from ar.spectrum import (
    ar_gain_energy,
    ar_power_spectrum,
    ar_stat_score,
    band_ratio_db,
    estimate_formants,
)
from iklp.hyperparams import active_components
from prism.pack import NormalizedPACK
from utils.audio import fit_affine_lag_nrmse, power_spectrum_db
from utils.stats import weighted_pitch_error


def get_standard_pack(d, period):
    """Return PACK with params estimated from experiments/lf/pack"""
    sigma_bc_by_d = {
        0: {"sigma_b": 2.0, "sigma_c": 0.5},
        1: {"sigma_b": 1.0, "sigma_c": 3.0},
        2: {"sigma_b": 0.4, "sigma_c": 7.0},
        3: {"sigma_b": 0.5, "sigma_c": 5.0},
    }

    sigma_a = 1.0
    sigma_b = sigma_bc_by_d[d]["sigma_b"]
    sigma_c = sigma_bc_by_d[d]["sigma_c"]

    pack = NormalizedPACK(
        d=d,
        J=1,
        period=period,
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
    )

    return pack


def unpack4(x):
    return (list(x) + [np.nan] * 4)[:4]


def _as_ms(sample_idx, fs):
    return 1e3 * np.asarray(sample_idx, dtype=np.float64) / float(fs)


def _load_full_file_context(group):
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


def post_process_run(run, metrics, f0):
    group = run["group"]
    frame = run["frame"]

    posterior_pi = metrics.E.nu_w / (metrics.E.nu_w + metrics.E.nu_e)
    sum_theta = (metrics.E.theta).sum()

    a = np.asarray(metrics.a)
    filter_gain_energy = ar_gain_energy(a)
    filter_stationary_score = ar_stat_score(a)

    pitch_true = 1000.0 / np.mean(frame["periods_ms"])
    oq_true = np.mean(frame["oq"])

    I_eff = active_components(metrics.E.theta)
    pitch_est, pitch_wrmse, pitch_wmae = weighted_pitch_error(
        f0, metrics.E.theta, pitch_true
    )

    # FIXME: there could be multiple samples
    if metrics.signals.shape[0] > 1:
        warnings.warn(
            f"Multiple samples (shape={metrics.signals.shape}) not supported in post-processing yet, using first sample only"
        )

    inferred_signal = metrics.signals[0]
    inferred_noise = metrics.noise[0]
    inferred_excitation = inferred_signal + inferred_noise

    SNR_db = 10 * np.log10(np.var(inferred_signal) / np.var(inferred_noise))

    dt = 1.0 / float(frame["fs"])
    maxlag = int(0.5 / pitch_true / dt)  # half a pitch period

    # calculate NRMSE for signal only
    best_signal, original_signal = fit_affine_lag_nrmse(
        frame["dgf"], inferred_signal, maxlag=maxlag
    )
    signal_nrmse = original_signal["nrmse"]
    signal_aligned_nrmse = best_signal["nrmse"]

    # calculate NRMSE for excitation e(t)=s(t)+n(t)
    best_excitation, original_excitation = fit_affine_lag_nrmse(
        frame["dgf"], inferred_excitation, maxlag=maxlag
    )
    excitation_nrmse = original_excitation["nrmse"]
    excitation_aligned_nrmse = best_excitation["nrmse"]

    lag_est = 1e3 * best_excitation["lag"] / frame["fs"]

    # estimate formants
    f, P = ar_power_spectrum(metrics.a, frame["fs"], db=False)
    filter_mid_low_db, filter_mid_high_db = band_ratio_db(f, P)

    f, power_db = ar_power_spectrum(metrics.a, frame["fs"], db=True)
    centers, bandwidths = estimate_formants(f, power_db, peak_prominence=1.0)

    f1_est, f2_est, f3_est, f4_est = unpack4(centers)
    b1_est, b2_est, b3_est, b4_est = unpack4(bandwidths)

    return {
        # frame metadata
        "wav": group["wav"],
        "name": group["name"],
        "f0_hz_nominal": group["f0_hz"],
        "pressure_pa": group["pressure_pa"],
        "voiced_group": group["group"],
        "frame_index": frame["frame_index"],
        "restart_index": frame["restart_index"],
        # vi metadata
        "elbo": metrics.elbo,
        "num_iterations": metrics.i,
        "E_nu_w": metrics.E.nu_w,
        "E_nu_e": metrics.E.nu_e,
        "sum_theta": sum_theta,
        "posterior_pi": posterior_pi,
        "I_eff": I_eff,
        # oq and pitch
        "oq_true": oq_true,
        "pitch_true": pitch_true,
        "pitch_est": pitch_est,
        "pitch_wrmse": pitch_wrmse,
        "pitch_wmae": pitch_wmae,
        # fit
        "SNR_db": SNR_db,  # if small, noise is doing excitation sculpting
        "signal_nrmse": signal_nrmse,
        "signal_aligned_nrmse": signal_aligned_nrmse,
        "excitation_nrmse": excitation_nrmse,
        "excitation_aligned_nrmse": excitation_aligned_nrmse,
        "lag_est": lag_est,
        "affine_lag_a": best_excitation["a"],
        "affine_lag_b": best_excitation["b"],
        "affine_lag": best_excitation["lag"],
        # filter
        "filter_mid_low_db": filter_mid_low_db,  # if small, AR is doing excitation sculpting
        "filter_mid_high_db": filter_mid_high_db,
        "filter_gain_energy": filter_gain_energy,
        "filter_stationary_score": filter_stationary_score,
        # formants
        "f1_est": f1_est,
        "f2_est": f2_est,
        "f3_est": f3_est,
        "f4_est": f4_est,
        "b1_est": b1_est,
        "b2_est": b2_est,
        "b3_est": b3_est,
        "b4_est": b4_est,
    }


def plot_run(result, run, metrics, f0):
    group = run["group"]
    frame = run["frame"]

    fs_model = float(frame["fs"])

    t_samples = np.asarray(frame["t_samples"], dtype=np.float64)
    speech = np.asarray(frame["speech"], dtype=np.float64)
    gf = np.asarray(frame["gf"], dtype=np.float64)
    dgf = np.asarray(frame["dgf"], dtype=np.float64)

    n = int(
        min(
            len(t_samples),
            len(speech),
            len(gf),
            len(dgf),
            len(metrics.signals[0]),
            len(metrics.noise[0]),
        )
    )
    t_samples = t_samples[:n]
    speech = speech[:n]
    gf = gf[:n]
    dgf = dgf[:n]

    inferred_signal = np.asarray(metrics.signals[0], dtype=np.float64)[:n]
    inferred_noise = np.asarray(metrics.noise[0], dtype=np.float64)[:n]
    inferred_excitation = inferred_signal + inferred_noise
    frame_duration_ms = 1e3 * len(speech) / fs_model

    true_pitch = float(result["pitch_true"])
    lag_samples = int(result["affine_lag"])
    affine_a = float(result["affine_lag_a"])
    affine_b = float(result["affine_lag_b"])
    align_lag_ms = float(result.get("lag_est", np.nan))
    if lag_samples < 0:
        warnings.warn(
            f"Expected nonnegative affine_lag, got {lag_samples}; clamping to 0 for plotting."
        )
        lag_samples = 0
    if not np.isfinite(align_lag_ms):
        align_lag_ms = 1e3 * lag_samples / fs_model

    # Align ground truth to the estimated excitation using stored affine+lag fit.
    # Convention for plotting here: shift ground truth right by lag_samples.
    aligned_true_dgf = np.full(n, np.nan, dtype=np.float64)
    if np.isfinite(affine_a) and abs(affine_a) > 1e-12 and lag_samples < n:
        aligned_true_dgf[lag_samples:] = (1 / affine_a) * (
            dgf[: n - lag_samples] - affine_b
        )
    elif lag_samples < n:
        warnings.warn(
            f"Cannot align ground truth: affine_lag_a={affine_a} is not usable."
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

    f_ar, p_ar_db = ar_power_spectrum(metrics.a, fs_model, db=True)
    mask_ar = np.isfinite(f_ar) & np.isfinite(p_ar_db) & (f_ar > 0)
    f_ar_plot = f_ar[mask_ar]
    p_ar_db_plot = p_ar_db[mask_ar]
    centers = np.asarray(
        [
            result["f1_est"],
            result["f2_est"],
            result["f3_est"],
            result["f4_est"],
        ],
        dtype=np.float64,
    )
    bandwidths = np.asarray(
        [
            result["b1_est"],
            result["b2_est"],
            result["b3_est"],
            result["b4_est"],
        ],
        dtype=np.float64,
    )

    # Spectra: estimated signal/noise are shown as-is.
    # Ground truth uses the aligned (shifted+affine) DGF, without NaN-padded prefix.
    dgf_spec = aligned_true_dgf[np.isfinite(aligned_true_dgf)]

    f_signal, p_signal_db = power_spectrum_db(inferred_signal, fs_model)
    mask_signal = (
        np.isfinite(f_signal) & np.isfinite(p_signal_db) & (f_signal > 0)
    )
    f_signal_plot = f_signal[mask_signal]
    p_signal_db_plot = p_signal_db[mask_signal]
    if len(dgf_spec) > 0:
        f_signal_true, p_signal_true_db = power_spectrum_db(dgf_spec, fs_model)
        mask_signal_true = (
            np.isfinite(f_signal_true)
            & np.isfinite(p_signal_true_db)
            & (f_signal_true > 0)
        )
        f_signal_true_plot = f_signal_true[mask_signal_true]
        p_signal_true_db_plot = p_signal_true_db[mask_signal_true]
    else:
        f_signal_true_plot = np.asarray([], dtype=np.float64)
        p_signal_true_db_plot = np.asarray([], dtype=np.float64)

    f_n, p_n_db = power_spectrum_db(inferred_noise, fs_model)
    mask_n = np.isfinite(f_n) & np.isfinite(p_n_db) & (f_n > 0)
    f_n_plot = f_n[mask_n]
    p_n_db_plot = p_n_db[mask_n]
    theta = np.asarray(metrics.E.theta, dtype=np.float64)
    theta = theta / (theta.sum() + 1e-16)
    f0 = np.asarray(f0, dtype=np.float64)
    use_f0_axis = len(f0) == len(theta)
    pitch_x = f0 if use_f0_axis else np.arange(len(theta))

    colors = qualitative.Plotly
    c_truth = colors[0]  # primary theme blue for ground truth
    c_inferred = colors[2]
    c_noise = "rgba(120,120,120,0.75)"
    c_black_tempered = "rgba(45,45,45,0.72)"
    c_pitch_posterior = c_black_tempered
    c_ar = c_black_tempered
    ar_xmax = (
        float(np.max(f_ar_plot))
        if len(f_ar_plot) and np.isfinite(np.max(f_ar_plot))
        else 5000.0
    )
    ar_xmax = max(50.0, ar_xmax)
    if ar_xmax <= 50.0:
        ar_xmax = 50.5
    signal_xmax = ar_xmax
    if len(f_signal_plot) and np.isfinite(np.max(f_signal_plot)):
        signal_xmax = max(signal_xmax, float(np.max(f_signal_plot)))
    if len(f_signal_true_plot) and np.isfinite(np.max(f_signal_true_plot)):
        signal_xmax = max(signal_xmax, float(np.max(f_signal_true_plot)))
    signal_xmax = max(50.5, signal_xmax)
    noise_xmax = (
        float(np.max(f_n_plot))
        if len(f_n_plot) and np.isfinite(np.max(f_n_plot))
        else ar_xmax
    )
    noise_xmax = max(50.5, noise_xmax)
    env_xmax = max(ar_xmax, signal_xmax, noise_xmax)

    harmonics = np.asarray([], dtype=np.float64)
    if np.isfinite(true_pitch) and true_pitch > 0:
        n_harm = int(np.floor(5000.0 / true_pitch))
        if n_harm >= 1:
            harmonics = true_pitch * np.arange(1, n_harm + 1)
            harmonics = harmonics[
                np.isfinite(harmonics)
                & (harmonics >= 50.0)
                & (harmonics <= min(5000.0, env_xmax))
            ]

    fig = make_subplots(
        rows=9,
        cols=1,
        shared_xaxes=False,
        row_heights=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        vertical_spacing=0.05,
        subplot_titles=[
            r"Data $x(t)$ in file context",
            r"True $u(t)$ in file context",
            r"Data $x(t)$",
            r"True $u(t)$",
            r"Estimated signal $s(t)$ and excitation $e(t)$ vs true $u'(t)$",
            r"Estimated signal spectral envelope $|S(f)|^2$ vs true $|U'(f)|$",
            r"Estimated noise spectral envelope $|N(f)|^2$",
            r"Estimated spectral envelope $|A(f)|^2$",
            r"Estimated $F_0$ from $\mathbb{E}[\theta_i]$ vs true $F_0$",
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
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=inferred_excitation,
            mode="lines",
            name="inferred excitation e(t)=s(t)+n(t)",
            line=dict(color=c_noise),
            opacity=0.8,
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

    fig.add_trace(
        go.Scatter(
            x=f_n_plot,
            y=p_n_db_plot,
            mode="lines",
            name="inferred noise spectrum |N(f)|^2",
            line=dict(color=c_noise),
            opacity=0.8,
            showlegend=False,
        ),
        row=7,
        col=1,
    )
    add_envelope_decorations(
        row=7,
        x_max=noise_xmax,
        y_series=[p_n_db_plot],
        show_harmonic_legend=False,
    )

    fig.add_trace(
        go.Scatter(
            x=f_ar_plot,
            y=p_ar_db_plot,
            mode="lines",
            name="inferred AR spectrum |A(f)|^2",
            line=dict(color=c_ar),
            showlegend=False,
        ),
        row=8,
        col=1,
    )
    for x in centers:
        if not (np.isfinite(x) and x > 0 and x <= 5000.0):
            continue
        fig.add_vline(
            x=float(x),
            line_color=c_black_tempered,
            line_dash="dot",
            opacity=0.8,
            row=8,
            col=1,
        )
    for c, bw in zip(np.asarray(centers), np.asarray(bandwidths)):
        if not (
            np.isfinite(c)
            and np.isfinite(bw)
            and c > 0
            and c <= 5000.0
            and bw > 0
        ):
            continue
        x0 = float(c - bw / 2.0)
        x1 = float(c + bw / 2.0)
        if x1 <= 0:
            continue
        x0 = max(x0, 1e-6)
        y_peak = float(np.interp(c, f_ar, p_ar_db))
        y_bw = y_peak - 3.0  # canonical -3 dB bandwidth line

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y_bw, y_bw],
                mode="lines",
                line=dict(color=c_black_tempered, width=2),
                name="-3 dB bandwidth",
                showlegend=False,
            ),
            row=8,
            col=1,
        )
    add_envelope_decorations(
        row=8,
        x_max=ar_xmax,
        y_series=[p_ar_db_plot],
        show_harmonic_legend=False,
    )

    fig.add_trace(
        go.Scatter(
            x=pitch_x,
            y=theta,
            mode="lines",
            name="pitch posterior p(F0)",
            line=dict(color=c_pitch_posterior),
            showlegend=False,
        ),
        row=9,
        col=1,
    )
    if np.isfinite(true_pitch) and (
        (use_f0_axis and true_pitch > 0) or (not use_f0_axis)
    ):
        fig.add_vline(
            x=float(true_pitch),
            line_color=c_truth,
            line_dash="dash",
            opacity=0.8,
            annotation_text=r"$F_0$ true",
            annotation_position="top right",
            row=9,
            col=1,
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
    fig.update_xaxes(matches="x6", row=7, col=1)
    fig.update_xaxes(matches="x6", row=8, col=1)

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
        type="log",
        range=[np.log10(50.0), np.log10(env_xmax)],
        title_standoff=2,
        automargin=True,
        row=6,
        col=1,
    )
    fig.update_xaxes(
        type="log",
        range=[np.log10(50.0), np.log10(env_xmax)],
        title_standoff=2,
        automargin=True,
        row=7,
        col=1,
    )
    fig.update_xaxes(
        title_text=r"frequency $f$ (Hz)",
        type="log",
        range=[np.log10(50.0), np.log10(env_xmax)],
        title_standoff=2,
        automargin=True,
        row=8,
        col=1,
    )
    row9_axis_kwargs = {
        "title_text": (
            "fundamental frequency $F_0$ (Hz)" if use_f0_axis else "pitch index"
        )
    }
    if use_f0_axis:
        valid_f0 = pitch_x[np.isfinite(pitch_x) & (pitch_x > 0)]
        if len(valid_f0) > 0:
            x0 = float(np.min(valid_f0))
            x1 = float(np.max(valid_f0))
            if x1 > x0 > 0:
                row9_axis_kwargs.update(
                    {
                        "type": "log",
                        "range": [np.log10(x0), np.log10(x1)],
                    }
                )
    fig.update_xaxes(row=9, col=1, **row9_axis_kwargs)

    fig.update_yaxes(title_text="amplitude", row=1, col=1)
    fig.update_yaxes(title_text="amplitude", row=2, col=1)
    fig.update_yaxes(title_text="amplitude", row=3, col=1)
    fig.update_yaxes(title_text="amplitude", row=4, col=1)
    fig.update_yaxes(title_text="amplitude", row=5, col=1)
    fig.update_yaxes(title_text="power (dB)", row=6, col=1)
    fig.update_yaxes(title_text="power (dB)", row=7, col=1)
    fig.update_yaxes(title_text="power (dB)", row=8, col=1)
    fig.update_yaxes(title_text="probability", row=9, col=1)

    if np.isfinite(align_lag_ms):
        fig.add_annotation(
            text=rf"$\mathrm{{lag}}\approx{align_lag_ms:.2f}\,\mathrm{{ms}}$",
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

    filename = result["wav"].split("/")[-1]
    main_title = (
        "EGIFA | "
        f"`{filename}` | "
        f"`group {result['voiced_group']}` | "
        f"`frame {result['frame_index']}`"
    )

    fig.update_layout(
        height=240 * 9,
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
