# %% tags=["parameters", "export"]
collection = "vowel"
egifa_f0 = 120
method = "iaif"

# %%
import traceback

import numpy as np
from tqdm import tqdm

from egifa.chien import get_voiced_runs_matlab, plot_run, post_process_run
from egifa.data import get_voiced_meta
from utils import time_this

# %%
groups = [
    g
    for g in get_voiced_meta(path_contains=collection)
    if g["f0_hz"] == egifa_f0
]

# %%
with time_this() as elapsed:
    runs = get_voiced_runs_matlab(groups, method=method)
    runs = list(runs)

# %% tags=["export"]
time_per_iter = elapsed.walltime / len(runs)

results = [post_process_run(run) for run in tqdm(runs)]

# %%
# Plot 5 quartiles from best to worst
scores = np.array([r["excitation_aligned_nrmse"] for r in results])

valid = np.isfinite(scores)
idx_valid = np.where(valid)[0]

if len(idx_valid) == 0:
    raise RuntimeError("No valid scores.")

order = idx_valid[np.argsort(scores[valid])]
qpos = [0.0, 0.25, 0.5, 0.75, 1.0]
qidx = [order[int(q * (len(order) - 1))] for q in qpos]

for qi, i in zip(qpos, qidx):
    print(f"\nQuartile {1 - qi:.2f} -> index {i}")

    try:
        plot_run(runs[i])
    except Exception:
        traceback.print_exc()
