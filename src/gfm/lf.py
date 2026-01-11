# %%
import numpy as np

from utils import constants, lfmodel

DEFAULT_PERIOD_MS = 10.0  # 100 Hz
DEFAULT_SAMPLES_PER_PERIOD = 1024

# Values copied from data/OPENGLOT/.../synthFrame.m
LF_MODALITIES = {
    "modal": {"Ee": 1.0, "Ra": 0.01, "Rg": 1.17, "Rk": 0.34},
    "breathy": {"Ee": 10 ** (0.7 / 20), "Ra": 0.025, "Rg": 0.88, "Rk": 0.41},
    "whispery": {"Ee": 10 ** (-4.6 / 20), "Ra": 0.07, "Rg": 0.94, "Rk": 0.32},
    "creaky": {"Ee": 10 ** (-1.8 / 20), "Ra": 0.008, "Rg": 1.13, "Rk": 0.2},
}


def lf_times_from_ratios(params, period_ms):
    """Convert LF ratios [Ra, Rg, Rk] to timings expected by lfmodel.dgf()."""

    period_s = period_ms / 1000.0
    ratios = {**params, "T0": period_s}
    timings = lfmodel.convert_lf_params(ratios, "R -> T")
    return {key: float(timings[key]) for key in ("T0", "Te", "Tp", "Ta")}


def synthesize_lf_period(
    params,
    *,
    period_ms=DEFAULT_PERIOD_MS,
    samples_per_period=DEFAULT_SAMPLES_PER_PERIOD,
    normalize_power=False,
    add_noise=False,
    cacheid=654561,
):
    """Synthesize glottal flow derivative and flow for one LF period."""

    period_s = period_ms / 1000.0
    time_axis = np.linspace(0.0, period_s, samples_per_period)
    dt = time_axis[1] - time_axis[0]

    lf_params = lf_times_from_ratios(params, period_ms)
    d_flow = params["Ee"] * np.asarray(lfmodel.dgf(time_axis, lf_params))

    flow = np.cumsum(d_flow) * dt

    if normalize_power:
        avg_power = (np.sum(d_flow**2) * dt) / period_s
        scale = 1.0 / np.sqrt(avg_power) if avg_power > 0.0 else 1.0
        d_flow = d_flow * scale
        flow = flow * scale

    if add_noise:
        rng = np.random.default_rng(cacheid)
        noise_amp = np.sqrt(constants.NOISE_FLOOR_POWER)
        noise = noise_amp * rng.normal(size=d_flow.shape)
        d_flow = d_flow + noise
        flow = flow + np.cumsum(noise) * dt

    return {
        "t": time_axis * 1e3,  # milliseconds
        "du": d_flow,
        "u": flow,
        "timings": lf_params,
    }


def lf_modality_waveforms(
    *,
    period_ms=DEFAULT_PERIOD_MS,
    samples_per_period=DEFAULT_SAMPLES_PER_PERIOD,
    normalize_power=False,
    add_noise=False,
):
    """Return synthesized LF waveforms for the four OPENGLOT modalities."""

    data = {}
    for name, params in LF_MODALITIES.items():
        data[name] = synthesize_lf_period(
            params,
            period_ms=period_ms,
            samples_per_period=samples_per_period,
            normalize_power=normalize_power,
            add_noise=add_noise,
        )
    return data


def lf_relaxation_open_phase(Rd, tc, N, open_phase_only=False, eps=1e-5):
    """Generate normalized LF exemplar with given relaxation coefficient Rd on the open phase [0, tc] with N samples

    Here Rd in [0.3, 2.7] according to Fant (1995) indicates tense to lax phonation.

    If open_phase_only is True, only the open phase [0, tc] of the LF model is generated.
    The LF waveform is thus cut off at the point where the cumulative flow U reaches 1 - eps, because for small Rd the effective GCI is way lower than tc. eps was chosen for a good balance across the Rd range.
    The waveform is then stretched to span [0, tc] by definition and resampled to N points.
    The closure constraint and unit power is then imposed.
    """
    p = lfmodel.convert_lf_params({"T0": tc, "Rd": Rd}, "Rd -> T")

    N_dense = 10000
    t = np.linspace(0, tc, N_dense)
    dt = t[1] - t[0]

    du = np.array(lfmodel.dgf(t, p))
    u = np.cumsum(du) * dt

    if open_phase_only:
        # find cutoff point where flow is total but eps
        U = np.cumsum(u) * dt
        U = U / U[-1]

        idx_cutoff = np.where(U >= 1 - eps)[0][0]
        t_cutoff = t[idx_cutoff]

        # cutoff at t_cutoff
        mask = t <= t_cutoff
        t = t[mask]
        du = du[mask]

    # stretch time axis to tc and
    # resample t, du, u to N points in open phase [0, tc]
    t *= tc / t[-1]
    t_data = np.linspace(0, tc, N + 1)[:-1]  # exclude endpoint
    du = np.interp(t_data, t, du)
    dt = t_data[1] - t_data[0]

    # impose closure constraint
    I = np.sum(du) * dt
    du -= I / tc
    u = np.cumsum(du) * dt

    # normalize energy to 1
    rescale = (du**2).sum() * dt / tc
    du /= np.sqrt(rescale)
    u /= np.sqrt(rescale)

    return {
        "t": t_data,
        "u": u,
        "du": du,
    }


if __name__ == "__main__":
    import pandas as pd
    from matplotlib import pyplot as plt

    # Visualize the four LF modalities
    waves = lf_modality_waveforms(
        period_ms=DEFAULT_PERIOD_MS, normalize_power=False
    )

    records = []
    for name, wave in waves.items():
        timings = wave["timings"]
        records.append(
            {
                "phonation": name,
                "Te (ms)": timings["Te"] * 1e3,
                "Tp (ms)": timings["Tp"] * 1e3,
                "Ta (ms)": timings["Ta"] * 1e3,
            }
        )

    timing_table = pd.DataFrame.from_records(records).set_index("phonation")
    print(timing_table)

    fig, (ax_du, ax_u) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name in ["modal", "breathy", "whispery", "creaky"]:
        wave = waves[name]
        ax_du.plot(wave["t"], wave["du"], label=name)
        ax_u.plot(wave["t"], wave["u"], label=name)

    ax_du.set_ylabel("dU/dt (a.u.)")
    ax_du.set_title("LF glottal flow derivatives")
    ax_du.legend(loc="upper right")

    ax_u.set_xlabel("Time (ms)")
    ax_u.set_ylabel("U (a.u.)")
    ax_u.set_title("LF glottal flow")
    ax_u.legend(loc="upper right")

    fig.tight_layout()

    # Original relaxation open-phase smoke test
    tc = 6.0
    Rd = 1.0  # modal
    N = 265

    d = lf_relaxation_open_phase(Rd, tc, N, open_phase_only=False)

    plt.figure()
    plt.plot(d["t"], d["du"], label="du (relaxation)")
    plt.plot(d["t"], d["u"], label="u (relaxation)")
    plt.legend()
    plt.title("lf_relaxation_open_phase check")
    plt.show()
