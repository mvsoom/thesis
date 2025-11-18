# %%
import jax
from matplotlib import pyplot as plt

from utils import lfmodel

jax.config.update("jax_log_compiles", False)
import os

import numpy as np

os.makedirs("data/hard_gci", exist_ok=True)
os.makedirs("data/soft_gci", exist_ok=True)

# %%
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
def repu(x, d):
    return np.heaviside(x, 1) * x**d


d = 1
H = 4

t = t_data

sigma_a = 1.0
b = np.random.uniform(0.0, tc, size=H)

Phi = repu(t[:, None] - b[None, :], d)

r = repu(tc - b, d + 1) / (d + 1)
q = r / np.linalg.norm(r)
I = np.eye(H)
Sigma = sigma_a**2 * (I - np.outer(q, q))  # rank H - 1

# sample from a ~ N(0, Sigma) which is not full rank
U, S, Vt = np.linalg.svd(Sigma)
z = np.random.normal(0, 1.0, size=H)
a = U @ np.diag(np.sqrt(S)) @ z

f = Phi @ a

plt.plot(t_data, f)
plt.plot(t_data, np.cumsum(f) * (t_data[1] - t_data[0]))
# mark the b locations
for bi in b:
    plt.axvline(bi, color="gray", linestyle="--")

# %%
import numpy as np


def repu(x, d):
    return np.heaviside(x, 1.0) * x**d


def build_phi(t, b, d):
    return repu(t[:, None] - b[None, :], d)


def build_q(b, tc, d):
    r = (tc - b) ** (d + 1) / float(d + 1)
    nr = np.linalg.norm(r)
    if nr == 0.0:
        nr = np.nan
    return r / nr


def log_evidence_fixed_sigma(t, u, H, d, tc, sigma_a, sigma):
    tmin, tmax = t[0], t[-1]
    b = np.linspace(tmin, tmax, H)

    phi = build_phi(t, b, d)
    q = build_q(b, tc, d)

    I = np.eye(H)
    P = I - np.outer(q, q)

    K = sigma_a**2 * (phi @ P @ phi.T)
    K.flat[:: K.shape[0] + 1] += sigma**2

    U, S, Vt = np.linalg.svd(K)
    if np.any(S <= 0.0):
        return -1e300

    quad = u @ ((U * (1.0 / S)) @ (U.T @ u))
    ldet = np.sum(np.log(S))
    N = len(u)
    return -0.5 * (quad + ldet + N * np.log(2.0 * np.pi))


sigma = 0.5  # -6 dB
sigma_a = 1.0
d = 0

du_data = lf_exemplars["hard_gci"]["data"]["du"]

scores = []
for H in range(0, N):
    L = log_evidence_fixed_sigma(t_data, du_data, H, d, tc, sigma_a, sigma)
    scores.append(L)

best_H, best_score = np.argmax(scores), np.max(scores)
print(best_H, best_score)


# %%
def map_a(t, u, b, d, tc, sigma_a, sigma):
    H = len(b)
    phi = build_phi(t, b, d)
    q = build_q(b, tc, d)
    I = np.eye(H)
    P = I - np.outer(q, q)
    Sigma_a = sigma_a**2 * P

    K = phi @ Sigma_a @ phi.T
    K.flat[:: K.shape[0] + 1] += sigma**2

    U, S, Vt = np.linalg.svd(K)
    Sinv = 1.0 / S
    Kinv_u = (U * Sinv) @ (U.T @ u)
    return Sigma_a @ phi.T @ Kinv_u


# build grid b for best_H
b_best = np.linspace(t_data[0], t_data[-1], best_H)

# posterior MAP amplitudes
a_best = map_a(
    t_data, du_data, b_best, d=d, tc=tc, sigma_a=sigma_a, sigma=sigma
)

# reconstruct fit
phi_best = repu(t_data[:, None] - b_best[None, :], d)
u_fit = phi_best @ a_best

# plot
plt.plot(t_data, du_data, label="data")
plt.plot(t_data, u_fit, label="fit")
plt.legend()
plt.show()

# %%
plt.plot(t_data, np.cumsum(du_data), label="data")
plt.plot(t_data, np.cumsum(u_fit), label="fit")
plt.legend()
