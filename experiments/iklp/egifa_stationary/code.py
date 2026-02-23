# %% tags=["parameters", "export"]
collection = "vowel"
kernel = "periodickernel"
seed = 54512703
egifa_f0 = 90

# %%
import os
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from egifa.evaluate import get_voiced_runs, plot_run, post_process_run
from gp.periodic import PeriodicSE
from iklp.hyperparams import (
    ARPrior,
    pi_kappa_hyperparameters,
    solve_for_alpha,
)
from iklp.mercer_op import backend
from iklp.periodic import f0_series
from iklp.run import vi_run_criterion_batched
from utils import dump_egg, time_this
from utils.jax import maybe32

# %%
runs = [
    r
    for r in get_voiced_runs(path_contains=collection)
    if r["group"]["f0_hz"] == egifa_f0
]

print("Number of runs:", len(runs))
fs = runs[0]["frame"]["fs"]
print("Sample rate:", fs)

# %%
x = jnp.vstack([run["frame"]["speech"] for run in runs])
t = jnp.arange(x.shape[1]) / fs * 1000.0  # shared cos stationary kernels
# t = jnp.vstack([run["frame"]["t_ms"] for run in runs])  # absolute time in ms

print("Data shape:", x.shape)
print("Data dtype:", x.dtype)

# %%
f0 = f0_series(60, 320, 100)
I = len(f0)
r = 16

# %%
from egifa.evaluate import get_standard_pack
from iklp.mercer import psd_eigh_fixed

if kernel == "periodickernel":
    ell = jnp.array(1.0)

    def compute_phi(f0):
        T_ms = 1000.0 / f0

        k = PeriodicSE(ell=ell, period=T_ms, J=r // 2)

        Phi = jax.vmap(k.compute_phi)(t)
        L = k.compute_weights_root()
        Phi = Phi @ L

        return Phi

    Phi = jax.vmap(compute_phi)(f0)
elif "pack" in kernel:
    d = int(kernel.split(":")[1])

    def compute_gram(f0):
        T_ms = 1000.0 / f0
        pack = get_standard_pack(d, T_ms)
        K = pack.gram(t[:, None]).to_dense()
        return K

    with jax.default_device(jax.devices("cpu")[0]):
        K = jax.vmap(compute_gram)(f0)  # (I, M, M)

    with time_this():  # ~2 min on CPU, 20 gb RAM
        Phi, energy = psd_eigh_fixed(K, rank=r)

    del K
    print("Captured energy:", np.mean(energy))
elif kernel == "whitenoise":
    M = len(t)
    Phi = jnp.empty((1, M, 0))
else:
    raise ValueError(f"Unknown kernel: {kernel}")

# %%
P = 20
arprior = ARPrior.yoshii_lambda(P)

beta = 0.0
alpha_scale = 1.0
kappa = 1.0
prior_pi = 0.5

alpha = solve_for_alpha(I) * alpha_scale

max_vi_iter = 50

h = pi_kappa_hyperparameters(
    maybe32(Phi),
    pi=maybe32(prior_pi),
    kappa=maybe32(kappa),
    alpha=maybe32(alpha),
    arprior=arprior,
    num_metrics_samples=-1,  # sample mean
    num_vi_iters=max_vi_iter,
    beta=maybe32(beta),
    mercer_backend="woodbury",
)

print("Phi shape:", h.Phi.shape)  # (I, M, r)
print("Phi dtype:", h.Phi.dtype)
print("Mercer operator backend:", backend(h))

# %%
batch_size = 32  # VRAM used is ~4 GB
master_key = jax.random.PRNGKey(seed)

with time_this() as elapsed:
    metrics_tree, unpack = vi_run_criterion_batched(
        master_key, x, h, batch_size=batch_size, verbose=True
    )

metrics_list = list(unpack(metrics_tree))

# %% tags=["export"]
time_per_iter = elapsed.walltime / metrics_tree.i.sum()

results = [
    post_process_run(run, metrics, f0)
    for run, metrics in tqdm(zip(runs, metrics_list))
]

# %%
payload = {
    "metrics_list": metrics_list,
    "runs": runs,
    # calculate `results` from these to save in disk space
}

dump_egg(payload, os.getenv("EXPERIMENT_NOTEBOOK_REL"))

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
    print(f"SNR dB   : {results[i]['SNR_db']:.2f}")
    print(f"I_eff    : {results[i]['I_eff']:.2f}")
    print(f"VI iters : {metrics_list[i].i}")

    try:
        plot_run(results[i], runs[i], metrics_list[i], f0)
    except Exception:
        traceback.print_exc()