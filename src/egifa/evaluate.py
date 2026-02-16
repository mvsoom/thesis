# %%
import numpy as np

from egifa.data import get_voiced_meta
from utils.audio import frame_signal


def get_voiced_runs(
    path_contains=None,
    frame_len_msec=32.0,
    hop_msec=16.0,
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

        fs_model = float(v["smooth"]["fs"])
        fs_abs = float(v["fs"])
        frame_len = int(frame_len_msec / 1000 * fs_model)
        hop = int(hop_msec / 1000 * fs_model)

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
            t_ms = 1e3 * t / fs_abs

            t_min, t_max = t[0], t[-1]
            loc = np.where((t_min <= v["gci"]) & (v["gci"] <= t_max))[0]

            gci = v["gci"][loc]
            goi = v["goi"][loc]
            oq = v["oq"][loc[:-1]]
            periods_ms = v["periods_ms"][loc[:-1]]

            for restart_index in range(num_vi_restarts):
                f = {
                    "fs": fs_model,
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
from plotly.subplots import make_subplots

from ar.spectrum import (
    ar_gain_energy,
    ar_power_spectrum,
    ar_stat_score,
    estimate_formants,
)
from iklp.hyperparams import active_components
from utils.audio import fit_affine_lag_nrmse, power_spectrum_db
from utils.stats import weighted_pitch_error


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


def _infer_pitch_from_frame(frame):
    periods_ms = np.asarray(frame.get("periods_ms", []), dtype=np.float64)
    periods_ms = periods_ms[np.isfinite(periods_ms) & (periods_ms > 0)]
    if len(periods_ms) == 0:
        return np.nan
    return 1000.0 / np.mean(periods_ms)


def post_process_run(run, metrics, f0):
    group = run["group"]
    frame = run["frame"]

    group_type = "vowel" if "vowel" in group["wav"] else "speech"

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
    assert metrics.signals.shape[0] == 1, (
        "Multiple samples not supported in post-processing yet"
    )

    inferred_dgf = metrics.signals[0]
    inferred_noise = metrics.noise[0]

    voicedness_db = 10 * np.log10(np.var(inferred_dgf) / np.var(inferred_noise))

    dt = 1.0 / float(frame["fs"])
    maxlag = int(0.5 / pitch_true / dt)  # half a pitch period

    # calculate NRMSE for signal only
    best, original = fit_affine_lag_nrmse(
        inferred_dgf, frame["dgf"], maxlag=maxlag
    )
    dgf_nrmse = original["nrmse"]
    dgf_aligned_nrmse = best["nrmse"]

    # caculate NRMSE for signal + noise
    best_both, original_both = fit_affine_lag_nrmse(
        inferred_dgf + inferred_noise, frame["dgf"], maxlag=maxlag
    )
    dgf_both_nrmse = original_both["nrmse"]
    dgf_both_aligned_nrmse = best_both["nrmse"]

    # estimate formants
    f, power_db = ar_power_spectrum(metrics.a, frame["fs"], db=True)
    centers, bandwidths = estimate_formants(f, power_db, peak_prominence=1.0)

    f1_est, f2_est, f3_est, f4_est = unpack4(centers)
    b1_est, b2_est, b3_est, b4_est = unpack4(bandwidths)

    return {
        # frame metadata
        "wav": group["wav"],
        "type": group_type,
        "name": group["name"],
        "f0_hz_nominal": group["f0_hz"],
        "pressure_pa": group["pressure_pa"],
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
        "voicedness_db": voicedness_db,
        "dgf_nrmse": dgf_nrmse,
        "dgf_aligned_nrmse": dgf_aligned_nrmse,
        "dgf_both_nrmse": dgf_both_nrmse,
        "dgf_both_aligned_nrmse": dgf_both_aligned_nrmse,
        # filter
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


def plot_run(run, metrics, f0):
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
    inferred_signal_plus_noise = inferred_signal + inferred_noise
    frame_duration_ms = 1e3 * len(speech) / fs_model

    true_pitch = _infer_pitch_from_frame(frame)
    dt = 1.0 / fs_model
    if np.isfinite(true_pitch) and true_pitch > 0:
        maxlag = max(1, int(0.5 / true_pitch / dt))  # half pitch period
    else:
        maxlag = max(1, int(0.002 / dt))

    try:
        best_true_to_inferred, _ = fit_affine_lag_nrmse(
            dgf, inferred_signal_plus_noise, maxlag=maxlag
        )
        aligned_true_dgf = np.asarray(best_true_to_inferred["aligned"])
    except Exception:
        aligned_true_dgf = np.full_like(dgf, np.nan)

    fs_file, speech_full, gf_full = _load_full_file_context(group)
    file_t_ms = _as_ms(np.arange(len(speech_full)), fs_file)
    fs_abs = float(group["fs"])
    if not np.isfinite(fs_abs) or fs_abs <= 0:
        fs_abs = float(fs_file)
    if not np.isclose(fs_abs, fs_file, rtol=0.0, atol=1e-9):
        fs_abs = float(fs_file)
    t_ms = _as_ms(t_samples, fs_abs)

    group_t_ms = _as_ms(group["smooth"]["t_samples"], fs_abs)
    group_speech = np.asarray(group["smooth"]["speech"], dtype=np.float64)
    group_gf = np.asarray(group["smooth"]["gf"], dtype=np.float64)
    group_dgf = np.asarray(group["smooth"]["dgf"], dtype=np.float64)
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
    centers, bandwidths = estimate_formants(f_ar, p_ar_db, peak_prominence=1.0)

    f_n, p_n_db = power_spectrum_db(inferred_noise, fs_model)
    theta = np.asarray(metrics.E.theta, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    use_f0_axis = len(f0) == len(theta)
    pitch_x = f0 if use_f0_axis else np.arange(len(theta))
    frame_duration_text = (
        f"{frame_duration_ms:.1f} ms"
        if np.isfinite(frame_duration_ms)
        else "n/a"
    )

    fig = make_subplots(
        rows=9,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[
            "File context: speech",
            "File context: glottal flow",
            "Selected voiced group: speech / gf / dgf (smoothed)",
            f"Frame detail ({frame_duration_text}): speech",
            "Frame detail: gf",
            "Frame detail: dgf (aligned true / inferred / inferred+noise)",
            "AR spectral envelope with estimated formants",
            "Inferred noise power spectrum",
            "Pitch posterior (index domain)",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=file_t_ms, y=speech_full, mode="lines", name="file speech"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=file_t_ms, y=gf_full, mode="lines", name="file gf"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=group_t_ms,
            y=group_speech,
            mode="lines",
            name="group speech (smooth)",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=group_t_ms, y=group_gf, mode="lines", name="group gf (smooth)"
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=group_t_ms, y=group_dgf, mode="lines", name="group dgf (smooth)"
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=t_ms, y=speech, mode="lines", name="frame speech"),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_ms, y=gf, mode="lines", name="frame gf"),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=aligned_true_dgf,
            mode="lines",
            name="true dgf (aligned affine+lag)",
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ms, y=inferred_signal, mode="lines", name="inferred signal"
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=inferred_signal_plus_noise,
            mode="lines",
            name="inferred signal + noise",
        ),
        row=6,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=f_ar_plot, y=p_ar_db_plot, mode="lines", name="AR spectrum"
        ),
        row=7,
        col=1,
    )
    for x in centers:
        if not (np.isfinite(x) and x > 0):
            continue
        fig.add_vline(
            x=float(x),
            line_color="green",
            line_dash="dot",
            opacity=0.8,
            row=7,
            col=1,
        )
    for c, bw in zip(np.asarray(centers), np.asarray(bandwidths)):
        if not (np.isfinite(c) and np.isfinite(bw) and c > 0 and bw > 0):
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
                line=dict(color="green", width=2),
                name="-3 dB bandwidth",
                showlegend=False,
            ),
            row=7,
            col=1,
        )

    if len(f_ar_plot) > 0:
        low_start = max(float(np.min(f_ar_plot)), 1e-6)
        low_end = 100.0
        high_start = 5000.0
        high_end = float(np.max(f_ar_plot))

        if low_end > low_start:
            fig.add_vrect(
                x0=low_start,
                x1=low_end,
                fillcolor="rgba(120,120,120,0.20)",
                line_width=0,
                row=7,
                col=1,
            )
        if high_end > high_start:
            fig.add_vrect(
                x0=high_start,
                x1=high_end,
                fillcolor="rgba(120,120,120,0.20)",
                line_width=0,
                row=7,
                col=1,
            )

    fig.add_trace(
        go.Scatter(x=f_n, y=p_n_db, mode="lines", name="noise spectrum"),
        row=8,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=pitch_x, y=theta, mode="lines", name="pitch posterior"),
        row=9,
        col=1,
    )
    if np.isfinite(true_pitch):
        fig.add_vline(
            x=float(true_pitch),
            line_color="red",
            line_dash="dash",
            opacity=0.8,
            annotation_text="pitch_true",
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
        for row in (1, 2, 3):
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
            row=5,
            col=1,
        )
    for x in frame_goi_ms:
        fig.add_vline(
            x=float(x),
            line_color="purple",
            line_width=1,
            opacity=0.25,
            row=5,
            col=1,
        )

    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_xaxes(matches="x4", row=5, col=1)
    fig.update_xaxes(matches="x4", row=6, col=1)

    if np.isfinite(frame_start_ms) and np.isfinite(frame_end_ms):
        for row in (4, 5, 6):
            fig.update_xaxes(
                range=[frame_start_ms, frame_end_ms], row=row, col=1
            )

    fig.update_xaxes(
        title_text="absolute time (ms): file/group context", row=3, col=1
    )
    fig.update_xaxes(
        title_text="absolute time (ms): frame detail", row=6, col=1
    )
    fig.update_xaxes(title_text="frequency (Hz)", type="log", row=7, col=1)
    fig.update_xaxes(title_text="frequency (Hz)", row=8, col=1)
    fig.update_xaxes(
        title_text=(
            "fundamental frequency $F_0$ (Hz)" if use_f0_axis else "pitch index"
        ),
        row=9,
        col=1,
    )

    fig.update_yaxes(title_text="speech", row=1, col=1)
    fig.update_yaxes(title_text="gf", row=2, col=1)
    fig.update_yaxes(title_text="amplitude", row=3, col=1)
    fig.update_yaxes(title_text="speech", row=4, col=1)
    fig.update_yaxes(title_text="gf", row=5, col=1)
    fig.update_yaxes(title_text="dgf", row=6, col=1)
    fig.update_yaxes(title_text="power (dB)", row=7, col=1)
    fig.update_yaxes(title_text="power (dB)", row=8, col=1)
    fig.update_yaxes(title_text="power", row=9, col=1)

    fig.update_layout(
        height=1900,
        hovermode="x unified",
        title=(
            "EGIFA run diagnostics | "
            f"{group['name']} ({group['f0_hz']} Hz) | "
            f"group={group['group']} frame={frame['frame_index']} "
            f"restart={frame['restart_index']} | "
            f"elbo={metrics.elbo:.3f}"
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.03,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.show()

    print(
        f"Audio preview: full file ({len(speech_full) / fs_file:.2f} s) and "
        f"frame ({frame_duration_ms:.1f} ms)"
    )
    display(Audio(speech_full, rate=int(round(fs_file))))
    display(Audio(speech, rate=int(round(fs_model))))

    return [fig]
