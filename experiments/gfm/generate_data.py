# %%
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import lfmodel

os.makedirs("data/hard_gci", exist_ok=True)
os.makedirs("data/soft_gci", exist_ok=True)

# %%
# Generate LF exemplars (hard GCI and soft GCI)
N = 256  # number of samples per waveform
tc = 6.0  # open phase from [0, 7] msec
t_data = np.linspace(0, tc, N)

lf_exemplars = {
    "hard_gci": {
        "Rd": 0.22916666666733365,
        "t_cutoff": 1.821608534385752,  # point of maximum du excitation
        "T0": 6.0,
    },  # induces minimum allowed Ta = 0.005
    "soft_gci": {
        "Rd": 1.5,
        "t_cutoff": tc,
        "T0": 6.0,
    },
}


def generate_lf_examplar(Rd, t_cutoff, T0):
    p = lfmodel.convert_lf_params({"T0": T0, "Rd": Rd}, "Rd -> T")

    print(p)

    N_dense = 5000
    t = np.linspace(0, T0, N_dense)
    du = np.array(lfmodel.dgf(t, p))

    # cutoff at t_cutoff
    mask = t <= t_cutoff
    t = t[mask]
    du = du[mask]

    # stretch time axis to tc and
    # resample t, du, u to N points in open phase [0, tc]
    t *= tc / t[-1]
    du = np.interp(t_data, t, du)
    dt = t_data[1] - t_data[0]

    # impose closure constraint
    I = np.sum(du) * dt
    du -= I / tc
    u = np.cumsum(du) * dt

    print("closure constraint:", np.sum(du) * dt)

    # normalize energy to 1
    rescale = (du**2).sum() * dt / tc
    du /= np.sqrt(rescale)
    u /= np.sqrt(rescale)

    return {
        "t": t_data,
        "u": u,
        "du": du,
    }


examplar = lf_exemplars["hard_gci"]

q = generate_lf_examplar(**examplar)
examplar["data"] = q

plt.plot(t_data, q["du"], label="du (hard gci)")
plt.plot(t_data, q["u"], label="u (hard gci)")
plt.legend()

dt = t_data[1] - t_data[0]
q["du"].sum(), (q["du"] ** 2).sum() * dt / tc

np.savetxt(
    "data/hard_gci/lf.dat",
    np.column_stack([t_data, q["du"], q["u"]]),
    header="# t du u",
    comments="",
)

# %%
examplar = lf_exemplars["soft_gci"]

q = generate_lf_examplar(**examplar)
examplar["data"] = q

plt.plot(t_data, q["du"], label="du (soft gci)")
plt.plot(t_data, q["u"], label="u (soft gci)")
plt.legend()

dt = t_data[1] - t_data[0]
q["du"].sum(), (q["du"] ** 2).sum() * dt / tc

np.savetxt(
    "data/soft_gci/lf.dat",
    np.column_stack([t_data, q["du"], q["u"]]),
    header="# t du u",
    comments="",
)


# %%
# Sample a piecewise polynomial from the prior with closure constraint
from gfm.poly import build_phi, sample_poly

d = 1
H = 20

sigma_a = 1.0
b = np.random.uniform(0.0, tc, size=H)
b[0] = 0.0

f = sample_poly(d, H, t_data, tc, sigma_a=sigma_a, b=b, closure=True)

plt.plot(t_data, f)
plt.plot(t_data, np.cumsum(f) * (t_data[1] - t_data[0]))
# mark the b locations
for bi in b:
    plt.axvline(bi, color="gray", linestyle="--")

# %%
# Generate the parametric piecewise polynomial approximations to hard and soft GCI LF waveforms
# Number of basisfunctions H is selected by maximizing the model evidence p(u' | H, closure, b) given fixed sigma_a, sigma_noise as in Bretthorst (1988)
import numpy as np

from gfm.poly import log_evidence_fixed_sigma, map_a

sigma_noise = 0.5  # -6 dB
sigma_a = 1.0

for name, examplar in lf_exemplars.items():
    for d in [0, 1, 2, 3]:
        t = examplar["data"]["t"]
        du = examplar["data"]["du"]

        scores = []
        for H in tqdm(range(0, N), f"p(u' | H, {name}, d={d})"):
            b = np.linspace(0, tc, H)
            L = log_evidence_fixed_sigma(
                t, du, H, d, tc, sigma_a, b, sigma_noise, closure=True
            )
            scores.append(L)

        best_H, best_score = np.argmax(scores), np.max(scores)
        print(
            f"=> Best H for {name}: d={d}: {best_H} (log Z=score={best_score})"
        )

        # rebuild grid b for best_H
        b_best = np.linspace(0, tc, best_H)

        # posterior MAP amplitudes
        a_best = map_a(t, du, b_best, d, tc, sigma_a, sigma_noise, closure=True)

        # reconstruct fit
        phi_best = build_phi(t, b_best, d)
        du_fit = phi_best @ a_best

        u_fit = np.cumsum(du_fit) * (t[1] - t[0])

        # dump
        np.savetxt(
            f"data/{name}/d={d}.dat",
            np.column_stack([t, du_fit, u_fit]),
            header=f"# t du u (d={d} H={best_H})",
            comments="",
        )

        # plot in two panels (left du, right u)
        plt.plot(t, du, label=f"data {name}")
        plt.plot(t, du_fit, label=f"fit d={d} H={best_H}")
        plt.legend()
        plt.show()