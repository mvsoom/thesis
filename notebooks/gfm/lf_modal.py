# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

import gfm.lf as lf


# %%
MODALITY_ORDER = ["modal", "breathy", "whispery", "creaky"]
N_SPECTRAL_PERIODS = 5
MAX_HARMONIC = 64

waves = lf.lf_modality_waveforms(
    period_ms=lf.DEFAULT_PERIOD_MS,
    samples_per_period=lf.DEFAULT_SAMPLES_PER_PERIOD,
    normalize_power=False,
    add_noise=False,
)

records = []
for name in MODALITY_ORDER:
    timings = waves[name]["timings"]
    records.append(
        {
            "phonation": name,
            "Te (ms)": timings["Te"] * 1e3,
            "Tp (ms)": timings["Tp"] * 1e3,
            "Ta (ms)": timings["Ta"] * 1e3,
        }
    )

timing_table = pd.DataFrame.from_records(records).set_index("phonation")
timing_table


# %%
def harmonic_spectral_density_and_phase(
    x_one_period, dt_s, period_s, n_periods=5, max_harmonic=None
):
    # One LF period includes the endpoint; drop it before tiling periods.
    x = np.asarray(x_one_period, dtype=float)[:-1]
    x_long = np.tile(x, n_periods)
    n_total = len(x_long)
    fs = 1.0 / dt_s

    X = np.fft.rfft(x_long)
    n_harmonic_max = int(np.floor((0.5 * fs) * period_s))
    if max_harmonic is not None:
        n_harmonic_max = min(n_harmonic_max, int(max_harmonic))

    harmonic = np.arange(1, n_harmonic_max + 1)
    bins = harmonic * n_periods
    valid = bins < len(X)
    harmonic = harmonic[valid]
    bins = bins[valid]

    X_h = X[bins]
    psd_h = (np.abs(X_h) ** 2) / (fs * n_total)

    if n_total % 2 == 0:
        nyq_bin = n_total // 2
        psd_h = np.where(bins == nyq_bin, psd_h, 2.0 * psd_h)
    else:
        psd_h = 2.0 * psd_h

    power_db = 10.0 * np.log10(psd_h + 1e-20)

    phase = np.angle(X_h)
    if len(phase) > 0:
        phase = np.unwrap(phase)
    phase = phase - phase[0]

    freqs_h = harmonic / period_s
    return harmonic, freqs_h, power_db, phase


# %%
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=[
        "LF dU/dt waveform (one period)",
        "Harmonic power spectral density",
        "Harmonic unwrapped phase",
    ],
    vertical_spacing=0.08,
    row_heights=[0.30, 0.35, 0.35],
)

colors = {
    name: qualitative.Plotly[i % len(qualitative.Plotly)]
    for i, name in enumerate(MODALITY_ORDER)
}
power_in_view = []
phase_in_view = []

for name in MODALITY_ORDER:
    wave = waves[name]
    t_ms = wave["t"][:-1]
    du = wave["du"][:-1]
    dt_s = (wave["t"][1] - wave["t"][0]) * 1e-3
    period_s = float(wave["timings"]["T0"])
    harmonic, freqs_h, power_db, phase = harmonic_spectral_density_and_phase(
        wave["du"],
        dt_s=dt_s,
        period_s=period_s,
        n_periods=N_SPECTRAL_PERIODS,
        max_harmonic=MAX_HARMONIC,
    )
    octaves = np.log2(harmonic.astype(float))
    power_in_view.append(power_db)
    phase_in_view.append(phase)
    customdata = np.column_stack([harmonic, freqs_h])

    fig.add_trace(
        go.Scattergl(
            x=t_ms,
            y=du,
            mode="lines",
            name=name,
            legendgroup=name,
            line=dict(color=colors[name]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=octaves,
            y=power_db,
            mode="lines+markers",
            name=name,
            legendgroup=name,
            line=dict(color=colors[name]),
            marker=dict(size=4),
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "k=%{customdata[0]:.0f}<br>oct=%{x:.2f}"
                "<br>f=%{customdata[1]:.1f} Hz"
                "<br>PSD=%{y:.2f} dB/Hz<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=octaves,
            y=phase,
            mode="lines+markers",
            name=name,
            legendgroup=name,
            line=dict(color=colors[name]),
            marker=dict(size=4),
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "k=%{customdata[0]:.0f}<br>oct=%{x:.2f}"
                "<br>f=%{customdata[1]:.1f} Hz"
                "<br>phase=%{y:.2f} rad<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

power_vals = np.concatenate(power_in_view)
power_span = float(np.max(power_vals) - np.min(power_vals))
if power_span <= 0.0:
    power_span = 1.0
power_pad = 0.05 * power_span
power_range = [
    float(np.min(power_vals) - power_pad),
    float(np.max(power_vals) + power_pad),
]

phase_vals = np.concatenate(phase_in_view)
phase_span = float(np.max(phase_vals) - np.min(phase_vals))
if phase_span <= 0.0:
    phase_span = 1.0
phase_pad = 0.05 * phase_span
phase_range = [
    float(np.min(phase_vals) - phase_pad),
    float(np.max(phase_vals) + phase_pad),
]

octave_max = float(np.log2(MAX_HARMONIC))
tickvals = np.arange(0, int(octave_max) + 1, dtype=float)
ticktext = [str(2 ** int(v)) for v in tickvals]

fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
fig.update_xaxes(
    title_text="Harmonic index k",
    range=[0.0, octave_max],
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    row=2,
    col=1,
)
fig.update_xaxes(
    title_text="Harmonic index k",
    range=[0.0, octave_max],
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    row=3,
    col=1,
)
fig.update_yaxes(title_text="dU/dt (a.u.)", row=1, col=1)
fig.update_yaxes(title_text="Power (dB/Hz)", range=power_range, row=2, col=1)
fig.update_yaxes(title_text="Phase (rad)", range=phase_range, row=3, col=1)

fig.update_layout(
    height=980,
    width=760,
    template="plotly_white",
    legend=dict(groupclick="togglegroup"),
    title=(
        "LF modality harmonic spectra "
        f"(k/T0), estimated over {N_SPECTRAL_PERIODS} periods"
    ),
)
fig.show()
