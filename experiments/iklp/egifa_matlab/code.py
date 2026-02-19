# %% tags=["parameters", "export"]
from matplotlib import pyplot as plt

from egifa.data import get_meta_grouped

meta = get_meta_grouped()

# %%

from egifa.chien import *

# resample_file_to_egifa(m, fs_target=20_000)

ret = estimate_file_frames_from_meta(m=meta[0], method="iaif", fs_target=20_000)

f = ret["frames"][33]
plt.plot(f["uu"])

# %%
from egifa.chien import *

it = get_voiced_runs_matlab(path_contains="speech")


# %%
run = next(it)

group = run["group"]
frame = run["frame"]

dgf_est = frame["dgf_est"]
dgf_est /= np.max(np.abs(dgf_est))

plt.plot(frame["t_ms"], dgf_est)
plt.plot(frame["t_ms"], frame["dgf"])
# %%
