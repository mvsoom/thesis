# %%
import numpy as np

from egifa.chien import get_voiced_runs_matlab, plot_run, post_process_run

runs = get_voiced_runs_matlab(method="iaif", path_contains="vowel")
runs = list(runs)

# %%
run = runs[np.random.randint(len(runs))]

summary = post_process_run(run)

plot_run(run)

summary
