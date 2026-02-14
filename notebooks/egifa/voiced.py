# %%
import numpy as np
from matplotlib import pyplot as plt

from egifa.data import get_voiced_meta, smooth_dgf

# meta = list(get_voiced_meta("speech"))

fs = 44100

window_ms = 32
max_samples = int(fs * window_ms / 1000)

# do windowing at 90% quantile on speech = 48 cycles ~ 3 ells ~ comparable to what pitch track model learnt, so the taus are not off bounds for the inducing points
# max_samples = 2**14

meta = list(get_voiced_meta("speech", max_samples=max_samples))

v = meta[np.random.randint(len(meta))]

t_ms = v["t_samples"] / v["fs"] * 1000
t_ms = np.arange(len(v["gf"])) / v["fs"] * 1000

# plt.plot(t_ms, v["gf"])


# For u'(t) training data, we know the extent of tau because t_samples start and end exactly at GCIs
# We also know due to synthetic data AND conservative t-PRISM that the error in locating the GCIs is small
# Together with this we also know that for long lengthscales ell~15 a taylor expansion says locally we are linearly interpolating GCI events
# so we do this for simplicity

gci = v["gci"]
x, y = gci, np.arange(len(gci))

tau_samples = np.interp(v["t_samples"], x, y)

dgf = smooth_dgf(v["gf"], v["fs"])

plt.plot(tau_samples, dgf)

# %%
from gci.meta import pitch_track_model

model = pitch_track_model()

qsvi, whiten, unwhiten = model["qsvi"], model["whiten"], model["unwhiten"]

n = len(v["periods_ms"])
tau_g = np.arange(n) + 0.5
g = whiten(np.log10(v["periods_ms"]))

plt.plot(tau_g, g)

# %%
from prism.svi import gp_posterior_mean_from_eps, infer_eps_posterior_single

# qsvi.posterior.likelihood.obs_stddev = 1e-3

mu_eps, Sigma_eps = infer_eps_posterior_single(qsvi, tau_g, g)

tau, dtau = np.linspace(0, n, 1024, retstep=True)
mean = gp_posterior_mean_from_eps(qsvi, tau, mu_eps)

plt.plot(tau, mean)
plt.plot(tau_g, g, "x")

# %%
t0 = v["gci"][0]

y = unwhiten(mean)
T = 10.0**y
plt.plot(tau, 1000 / T)

# %%
t = t0 + np.cumsum(y) * dtau
plt.plot(tau, t)

# %%
# Estimate tau range to query via linear interpolation
mean_period = np.mean(v["period_samples"])

gci = v["gci"]
x, y = gci, np.arange(len(gci)) + 0.5

tau_samples = np.interp(v["t_samples"], x, y)
plt.plot(tau_samples, v["gf"])

# %%
Z = qsvi.inducing_inputs.squeeze()

offset = np.median(Z) - np.median(tau_samples)
tau_samples_offset = tau_samples  # + offset
plt.plot(tau_samples_offset - 0.5, v["gf"])

# rug plot of where Z lie
mask = (Z >= tau_samples_offset.min()) & (Z <= tau_samples_offset.max())
plt.plot(Z[mask], np.zeros_like(Z[mask]), "|", markersize=20)

# %%
# TODO: normalization

# TODO: take derivative

# %%

from egifa.data import get_voiced_meta

meta_vowel = list(get_voiced_meta("vowel"))
meta_speech = list(get_voiced_meta("speech"))

# %%
# Compute stats on how many samples
subsets = ["vowel", "speech"]
metas = [list(get_voiced_meta(s)) for s in subsets]

nsamples = {}
times_ms = {}
nsamples_eff = {}
nperiods_eff = {}

for meta in metas:
    # nsamples = np.array([len(m["speech"]) for m in meta])
    nsamples = np.array([len(m["gci"]) for m in meta])
    times_ms = np.array([len(m["speech"]) / m["fs"] * 1000 for m in meta])

    nsamples_eff = np.mean(nsamples)
    nperiods_eff = np.mean(
        times_ms / 1000 * np.array([m["f0_hz"] for m in meta])
    )

    plt.hist(nsamples, bins=20)
    plt.show()


# %%
# count for each element

# %%
# plot random voiced groups
from matplotlib import pyplot as plt

m = meta[np.random.randint(len(meta))]

t = np.arange(len(m["speech"])) / m["fs"]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, m["speech"])
plt.title(f"Speech: {m['name']} ({m['f0_hz']} Hz, {m['pressure_pa']} Pa)")
plt.subplot(2, 1, 2)
plt.plot(t, m["gf"])
plt.title("Glottal Flow")
plt.tight_layout()
plt.show()


# %%
from egifa.data import get_data

X, y, meta = get_data(with_metadata=True, width=16384)

# %%
# plot random element
i = np.random.randint(len(X))
plt.plot(X[i], y[i])
