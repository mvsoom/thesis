# %%
import numpy as np

from utils import lfmodel


def lf_relaxation_open_phase(Rd, tc, N, open_phase_only, eps=1e-5):
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
    from matplotlib import pyplot as plt

    tc = 6.0
    Rd = 1.0  # modal
    N = 265

    d = lf_relaxation_open_phase(Rd, tc, N)

    plt.plot(d["t"], d["du"], label="du")
    plt.plot(d["t"], d["u"], label="u")
    plt.legend()
