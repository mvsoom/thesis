import os
from glob import glob

import jax
import numpy as np
import soundfile as sf
from IPython.display import display

from ar.spectrum import (
    ar_gain_energy,
    ar_power_spectrum,
    ar_stat_score,
    estimate_formants,
    match_formants,
)
from iklp.hyperparams import active_components
from iklp.psi import psi_matvec
from utils.audio import fit_affine_lag_nrmse, power_spectrum_db, resample
from utils.plots import plt, retain
from utils.stats import weighted_pitch_error


class OpenGlotI:
    # From OPENGLOT paper Table 1
    true_resonance_frequencies = {
        "a": [730, 1090, 2440, 3500],
        "e": [530, 1840, 2480, 3500],
        "i": [390, 1990, 2550, 3500],
        "o": [570, 840, 2410, 3500],
        "u": [440, 1020, 2240, 3500],
        "ae": [660, 1720, 2410, 3500],
    }

    # From data/OPENGLOT/RepositoryI/Generation_codes/makeVT.m
    true_formant_bandwidths = [60, 100, 120, 175]

    @staticmethod
    def wav_files():
        PROJECT_DATA_PATH = os.environ["PROJECT_DATA_PATH"]
        root_path = f"{PROJECT_DATA_PATH}/OPENGLOT/RepositoryI"
        paths = glob(f"{root_path}/**/*.wav", recursive=True)
        return paths

    @staticmethod
    def parse_wav(wav_file):
        vowel, modality, true_pitch = (
            wav_file.split("/")[-1].split(".")[-2].split("_")[:3]
        )
        vowel = vowel.lower()
        modality = modality.lower()
        true_pitch = int(true_pitch.lower()[:-2])  # strip 'Hz'
        return vowel, modality, true_pitch

    @staticmethod
    def read_wav(wav_file, target_fs, verbose=True):
        audio, original_fs = sf.read(
            wav_file,
            always_2d=False,
            dtype="float64" if jax.config.jax_enable_x64 else "float32",
        )

        # Split channels
        x = audio[:, 0]
        gf = audio[:, 1]

        x = resample(x, original_fs, target_fs)
        gf = resample(gf, original_fs, target_fs)

        dt = 1.0 / target_fs
        dgf = np.gradient(gf, dt, edge_order=2)  # This has very large spikes

        # Normalize data to unit data x power
        # NOTE: normalizing wrt DGF confuses model compeletely
        scale = 1 / np.sqrt(np.mean(x**2))
        x = x * scale
        dgf = dgf * scale

        if verbose:
            print(
                f"Loaded {wav_file} ({len(x)} samples, {original_fs} Hz resampled to {target_fs} Hz); rescaled to unit power by {scale:.3f}"
            )

        return x, dgf, original_fs

    @staticmethod
    def post_process_run(run, metrics, f0):
        posterior_pi = metrics.E.nu_w / (metrics.E.nu_w + metrics.E.nu_e)
        sum_theta = (metrics.E.theta).sum()

        a = np.asarray(metrics.a)
        filter_gain_energy = ar_gain_energy(a)
        filter_stationary_score = ar_stat_score(a)

        I_eff = active_components(metrics.E.theta)
        pitch_wrmse, pitch_wmae = weighted_pitch_error(
            f0, metrics.E.theta, run["true_pitch"]
        )

        inferred_dgf = metrics.signals[0]

        dt = 1.0 / float(run["target_fs"])
        maxlag = int(1 / run["true_pitch"] / dt)  # one pitch period
        best, original = fit_affine_lag_nrmse(
            inferred_dgf, run["dgf"], maxlag=maxlag
        )
        dgf_nrmse = original["nrmse"]
        dgf_aligned_nrmse = best["nrmse"]

        f, power_db = ar_power_spectrum(metrics.a, run["target_fs"], db=True)
        centers, bandwidths = estimate_formants(
            f, power_db, peak_prominence=1.0
        )

        pairing = match_formants(
            centers, run["true_formants"], est_bw=bandwidths
        )
        estimated_formants = pairing["matched_freqs"]

        formant_rmse = np.sqrt(
            np.mean((run["true_formants"] - estimated_formants) ** 2)
        )

        formant_mae = np.mean(np.abs(run["true_formants"] - estimated_formants))

        f1_true, f2_true, f3_true, f4_true = run["true_formants"]
        f1_est, f2_est, f3_est, f4_est = estimated_formants

        return {
            "wav_file": run["wav_file"],
            "vowel": run["vowel"],
            "modality": run["modality"],
            "true_pitch": run["true_pitch"],
            "frame_index": run["frame_index"],
            "restart_index": run["restart_index"],
            "elbo": metrics.elbo,
            "num_iterations": metrics.i,
            "E_nu_w": metrics.E.nu_w,
            "E_nu_e": metrics.E.nu_e,
            "sum_theta": sum_theta,
            "posterior_pi": posterior_pi,
            "filter_gain_energy": filter_gain_energy,
            "filter_stationary_score": filter_stationary_score,
            "I_eff": I_eff,
            "pitch_wrmse": pitch_wrmse,
            "pitch_wmae": pitch_wmae,
            "formant_rmse": formant_rmse,
            "formant_mae": formant_mae,
            "dgf_nrmse": dgf_nrmse,
            "dgf_aligned_nrmse": dgf_aligned_nrmse,
            "f1_true": f1_true,
            "f2_true": f2_true,
            "f3_true": f3_true,
            "f4_true": f4_true,
            "f1_est": f1_est,
            "f2_est": f2_est,
            "f3_est": f3_est,
            "f4_est": f4_est,
        }

    @staticmethod
    def plot_run(run, metrics, f0, retain_plots=False):
        x = run["x"]
        dgf = run["dgf"]
        true_formants = run["true_formants"]
        true_pitch = run["true_pitch"]
        target_fs = run["target_fs"]

        dt = 1.0 / float(target_fs)
        t_ms = np.arange(x.shape[0]) * (1000.0 * dt)

        inferred_dgf = metrics.signals[0]
        e = psi_matvec(metrics.a, x)
        noise = e - inferred_dgf

        maxlag = int(1 / run["true_pitch"] / dt)  # one pitch period
        best, _ = fit_affine_lag_nrmse(inferred_dgf, run["dgf"], maxlag=maxlag)
        a = np.abs(best["a"])

        figs = []

        # 1) true vs inferred glottal flow + noise (time domain)
        fig1, ax = plt.subplots()

        ax.plot(t_ms, (1 / a) * dgf, label="True $u'(t)$ (rescaled)")
        ax.plot(
            t_ms,
            inferred_dgf,
            label="Inferred $u'(t)$",
        )
        ax.plot(
            t_ms,
            (1 / a) * best["aligned"],
            ls="--",
            label="Inferred $u'(t)$ (aligned)",
        )
        ax.plot(t_ms, noise, label="Inferred noise")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude")
        ax.legend()
        mid = 0.5 * (len(x) * dt * 1000.0)
        sides = 2000.0 / float(true_pitch)
        ax.set_xlim(mid - sides, mid + sides)
        figs.append(fig1)
        (retain(fig1) if retain_plots else display(fig1))

        # 2) raw frame view (x and dgf with offsets)
        fig2, ax = plt.subplots()
        ax.plot(t_ms, a * x + 3.0, label="Data $x(t)$ (rescaled)")
        ax.plot(t_ms, dgf - 3.0, label="True $u(t)$ (to scale)")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude")
        ax.legend()
        figs.append(fig2)
        (retain(fig2) if retain_plots else display(fig2))

        # 3) AR power spectrum with formants
        f_ar, p_ar_db = ar_power_spectrum(metrics.a, target_fs, db=True)
        centers, bws = estimate_formants(f_ar, p_ar_db, peak_prominence=1.0)

        fig3, ax = plt.subplots()
        ax.plot(f_ar, p_ar_db, lw=2)
        for f in true_formants:
            ax.axvline(f, color="C1", ls="--", alpha=0.8, label=None)
        for c in centers:
            ax.axvline(c, color="C2", ls=":", alpha=0.8, label=None)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power (dB)")
        ax.set_title("Inferred AR power spectrum")
        figs.append(fig3)
        (retain(fig3) if retain_plots else display(fig3))

        # 4) noise spectrum
        f_n, p_n_db = power_spectrum_db(noise, target_fs)

        fig4, ax = plt.subplots()
        ax.plot(f_n, p_n_db)
        ax.set_title("Inferred noise power spectrum")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power (dB)")
        figs.append(fig4)
        (retain(fig4) if retain_plots else display(fig4))

        # 5) pitch posterior (if available)
        fig5, ax = plt.subplots()
        theta = metrics.E.theta
        ax.plot(f0, theta)
        ax.set_xlabel("fundamental frequency $F_0$ (Hz)")
        ax.set_ylabel("power")
        ax.set_title("Pitch posterior")
        figs.append(fig5)
        (retain(fig5) if retain_plots else display(fig5))

        return figs


class OpenGlotII:
    true_resonance_frequencies = {
        "male": {
            "a": [752, 1095, 2616, 3169],
            "i": [340, 2237, 2439, 3668],
            "u": [367, 1180, 2395, 3945],
            "ae": [693, 1521, 2435, 3252],
        },
        "female": {
            "a": [848, 1210, 2923, 3637],
            "i": [379, 2634, 4256, 5395],
            "u": [420, 1264, 2714, 4532],
            "ae": [795, 1700, 2692, 3740],
        },
    }

    @staticmethod
    def wav_files():
        PROJECT_DATA_PATH = os.environ["PROJECT_DATA_PATH"]
        root_path = f"{PROJECT_DATA_PATH}/OPENGLOT/RepositoryII_*"
        paths = glob.glob(f"{root_path}/**/*.wav", recursive=True)
        return paths

    @staticmethod
    def parse_wav(wav_file):
        base = os.path.splitext(os.path.basename(wav_file))[0]

        gender, vowel, p, adduction = base.split("_")
        gender, vowel, true_pitch, adduction = (
            gender.lower(),
            vowel.lower(),
            int(p.rstrip("Hz").lower()),
            adduction.lower(),
        )

        return gender, vowel, true_pitch, adduction

    @staticmethod
    def read_wav(wav_file, target_fs, verbose=True):
        return OpenGlotI.read_wav(wav_file, target_fs, verbose)  # same