# %%
# parameters, export
seed = 0

pitch = 220
kernel = "periodickernel"
refine = False
gauge = False

# %%
import jax

jax.config.update("jax_log_compiles", False)
jax.config.update("jax_enable_x64", True)

master_key = jax.random.PRNGKey(seed)

# %%
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from iklp.hyperparams import (
    ARPrior,
    pi_kappa_hyperparameters,
)
from iklp.mercer_op import backend
from iklp.run import vi_run_criterion_batched
from utils import time_this
from utils.audio import frame_signal
from utils.jax import maybe32
from utils.openglot import OpenGlotI

# %%
P = 8

target_fs = 8000
M = 1024  # frame length
hop = 80

max_vi_iter = 50
batch_size = 256

dt = 1.0 / target_fs
t = np.arange(M) * dt  # sec

T = 1 / pitch  # Hz

beta = 0.0
alpha_scale = 1.0
kappa = 1.0
prior_pi = 0.5

# %%
if kernel == "periodickernel":
    from gp.periodic import PeriodicSE

    ell = 0.5
    T = 1.0 / pitch
    r = 20

    k = PeriodicSE(ell=jnp.array(ell), period=T, J=r)

    Phi = jax.vmap(k.compute_phi)(t)
    L = k.compute_weights_root()
    Phi = Phi @ L
elif kernel == "spack:1":
    pass


Phi = Phi.reshape((1, *Phi.shape))

# %%
arprior = ARPrior.yoshii_lambda(P)

I = Phi.shape[0]
alpha = 1.0  # solve_for_alpha(I) * alpha_scale # doesnt work for I <= 2

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

del Phi

print("Phi shape:", h.Phi.shape)  # (I, M, r)
print("Phi dtype:", h.Phi.dtype)
print("Mercer operator backend:", backend(h))


# %%
def all_runs(verbose=False):
    for wav_file in tqdm(OpenGlotI.wav_files()):
        vowel, modality, true_pitch = OpenGlotI.parse_wav(wav_file)
        if true_pitch != pitch:
            continue

        true_formants = OpenGlotI.true_resonance_frequencies[vowel]
        x_full, dgf_full, original_fs = OpenGlotI.read_wav(
            wav_file, target_fs, verbose=verbose
        )

        x_frames = frame_signal(x_full, M, hop)  # ((n_frames, frame_len)
        dgf_frames = frame_signal(dgf_full, M, hop)

        x_frames = maybe32(x_frames)
        dgf_frames = maybe32(dgf_frames)

        for frame_index, (x, dgf) in enumerate(zip(x_frames, dgf_frames)):
            for restart_index in range(h.num_vi_restarts):
                yield {
                    "wav_file": wav_file,
                    "original_fs": original_fs,
                    "target_fs": target_fs,
                    "vowel": vowel,
                    "modality": modality,
                    "true_pitch": true_pitch,
                    "true_formants": true_formants,
                    "frame_index": frame_index,
                    "num_frames": x_frames.shape[0],
                    "restart_index": restart_index,
                    "x": x,
                    "dgf": dgf,
                    "x_frames": x_frames,
                    "dgf_frames": dgf_frames,
                }


runs = list(all_runs())

print("Total runs:", len(runs))


# %%
x = jnp.vstack([run["x"] for run in runs])

print("Data shape:", x.shape)
print("Data dtype:", x.dtype)

# %%
with time_this() as elapsed:
    metrics_tree, unpack = vi_run_criterion_batched(
        master_key, x, h, batch_size=batch_size, verbose=True
    )

metrics_list = list(unpack(metrics_tree))

# %%
f0 = np.array([pitch])

# %%
# export

# This includes compilation for the shapes of the first and last batch, which are O(1) min
time_per_iter = elapsed.walltime / metrics_tree.i.sum()

results = [
    OpenGlotI.post_process_run(run, metrics, f0)
    for run, metrics in tqdm(zip(runs, metrics_list))
]

# %%
from utils.openglot import OpenGlotI

# Plot best u'(t) fit
i = int(np.nanargmin([r["dgf_aligned_nrmse"] for r in results]))
print(i)

OpenGlotI.plot_run(runs[i], metrics_list[i], f0, retain_plots=True)

# %%
# Plot best formant fit
i = int(np.nanargmin([r["formant_rmse"] for r in results]))
print(i)

OpenGlotI.plot_run(runs[i], metrics_list[i], f0, retain_plots=True)
