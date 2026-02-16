# pitch = 100
# pressure = 150
# type = "vowel"  # etc

# TODO: specialize on Z

# %%
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from egifa.data import _shuffle_iterable
from egifa.evaluate import get_voiced_runs, post_process_run
from gp.periodic import PeriodicSE
from iklp.hyperparams import (
    ARPrior,
    pi_kappa_hyperparameters,
    solve_for_alpha,
)
from iklp.mercer_op import backend
from iklp.run import vi_run_criterion_batched
from utils import time_this
from utils.jax import maybe32

runs = get_voiced_runs(frame_len_msec=128.0, hop_msec=128.0)  # FIXME

# %%
N_TEST = 100

runs = _shuffle_iterable(list(runs))
runs = runs[:N_TEST]

fs = runs[0]["frame"]["fs"]
print("Sample rate:", fs)

master_key = jax.random.PRNGKey(0)  # FIXME

# %%
x = jnp.vstack([run["frame"]["speech"] for run in runs])
t = jnp.arange(x.shape[1]) / fs * 1000.0  # shared cos stationary kernels
# t = jnp.vstack([run["frame"]["t_ms"] for run in runs])  # absolute time in ms

print("Data shape:", x.shape)
print("Data dtype:", x.dtype)


# %%
from iklp.periodic import f0_series

P = 24  # FIXME
f0 = f0_series(60, 320, 200)
I = len(f0)
r = 16

max_vi_iter = 50


# %%


ell = jnp.array(1.0)


def compute_phi(f0):
    T_ms = 1000.0 / f0

    k = PeriodicSE(ell=ell, period=T_ms, J=r // 2)

    Phi = jax.vmap(k.compute_phi)(t)
    L = k.compute_weights_root()
    Phi = Phi @ L

    return Phi


Phi = jax.vmap(compute_phi)(f0)
Phi.shape

# %%
from matplotlib import pyplot as plt

z = np.random.normal(size=Phi.shape[-1])

plt.plot(t, Phi[0, :, :] @ z)
plt.plot(t, Phi[10, :, :] @ z)

# %%

arprior = ARPrior.yoshii_lambda(P)

beta = 0.0
alpha_scale = 1.0
kappa = 1.0
prior_pi = 0.5

alpha = solve_for_alpha(I) * alpha_scale

h = pi_kappa_hyperparameters(
    maybe32(Phi),
    pi=maybe32(prior_pi),
    kappa=maybe32(kappa),
    alpha=maybe32(alpha),
    arprior=arprior,
    num_metrics_samples=1,
    num_vi_iters=max_vi_iter,
    beta=maybe32(beta),
)

print("Phi shape:", h.Phi.shape)  # (I, M, r)
print("Phi dtype:", h.Phi.dtype)
print("Mercer operator backend:", backend(h))

# %%
batch_size = 8

with time_this() as elapsed:
    metrics_tree, unpack = vi_run_criterion_batched(
        master_key, x, h, batch_size=batch_size, verbose=True
    )

metrics_list = list(unpack(metrics_tree))

# batch_size = 16 takes 2:30 and vram = 7400 MB
# batch_size = 8 takes 1:40 and vram = unknown

# %%
m = metrics_list[0]

# %%


# This includes compilation for the shapes of the first and last batch, which are O(1) min, and waiting for all runs in batch to finish
time_per_iter = elapsed.walltime / metrics_tree.i.sum()

results = [
    post_process_run(run, metrics, f0)
    for run, metrics in tqdm(zip(runs, metrics_list))
]

# %%
import traceback

from egifa.evaluate import plot_run

# Plot best u'(t) fit
print("Best u'(t) fit:", end=" ")

try:
    i = int(np.nanargmin([r["dgf_both_aligned_nrmse"] for r in results]))
    print(i)

    print(f"Voicedness db {results[i]['voicedness_db']:.2f}")

    print(f"I_eff: {results[i]['I_eff']:.2f}")

    figs = plot_run(runs[i], metrics_list[i], f0)
except Exception:
    traceback.print_exc()


# %%
# Plot worst u'(t) fit
print("Worst u'(t) fit:", end=" ")

try:
    i = int(np.nanargmax([r["dgf_both_aligned_nrmse"] for r in results]))
    print(i)

    print(f"Voicedness db {results[i]['voicedness_db']:.2f}")

    plot_run(runs[i], metrics_list[i], f0)
except Exception:
    traceback.print_exc()

# %%
