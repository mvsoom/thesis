# %%
# parameters, export
seed = 0

pitch = 100
kernel = "periodickernel"
gauge = True
scale_dgf_to_unit_power = True
beta = 1.0
refine = False


# %%
import jax

jax.config.update("jax_log_compiles", False)
jax.config.update("jax_enable_x64", True)

master_key = jax.random.PRNGKey(seed)


# %%
import traceback

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from iklp.hyperparams import ARPrior, pi_kappa_hyperparameters
from iklp.mercer_op import backend
from iklp.run import vi_run_criterion_batched
from utils import time_this
from utils.jax import maybe32
from utils.openglot import OpenGlotI

# %%
target_fs = 8000  # OpenGlotI is 8000 Hz

fs_factor = target_fs / 16000

M = int(512 * fs_factor)  # 32 msec frame length
hop = int(256 * fs_factor)  # 16 msec hop size

max_vi_iter = 50
batch_size = 128  # 2048

dt = 1.0 / target_fs
t = np.arange(M) * dt  # sec

T = 1 / pitch  # Hz
num_harmonics = int(np.floor((target_fs / pitch) / 2))

alpha_scale = 1.0
kappa = 1.0
prior_pi = 0.5
P = 8


# %%
T = 1.0 / pitch
J = num_harmonics


if kernel == "periodickernel":
    from gp.periodic import PeriodicSE

    # Use learned best params from earlier nested sampling experiments
    ell = 0.2384  # 0.5

    k = PeriodicSE(ell=jnp.array(ell), period=T, J=J)

    Phi = jax.vmap(k.compute_phi)(t)
    L = k.compute_weights_root()
    Phi = Phi @ L
elif kernel == "whitenoise":
    Phi = jnp.empty((M, 0))
    k = None

Phi = Phi.reshape((1, *Phi.shape))

print("Using kernel or GaussianProcess:", k)

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

print("Phi shape:", h.Phi.shape)  # (I, M, r)
print("Phi dtype:", h.Phi.dtype)
print("Mercer operator backend:", backend(h))


# %%
from utils.audio import frame_signal
from utils.reskew import polarity_reskew


def all_runs(verbose=False):
    for wav_file in tqdm(OpenGlotI.wav_files()):
        vowel, modality, true_pitch = OpenGlotI.parse_wav(wav_file)
        if true_pitch != pitch:
            continue

        true_formants = OpenGlotI.true_resonance_frequencies[vowel]
        x_full, dgf_full, original_fs = OpenGlotI.read_wav(
            wav_file, target_fs, verbose=verbose
        )

        polarity = polarity_reskew(x_full, target_fs)

        x_frames = frame_signal(x_full, M, hop)
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
                    "polarity": polarity,
                }


runs = list(all_runs())

print("Total runs:", len(runs))


# %%
fails = sum(r["polarity"] < 0 for r in runs)

print(f"Wrong polarities estimated by RESKEW: {fails}/{len(runs)}")
print("Note: polarities are NOT applied")


# %%

run_i = np.random.randint(len(runs))

z = np.random.randn(Phi.shape[-1])
plt.plot(t, Phi[0] @ z, label="prior sample")

dgf = runs[run_i]["dgf"]

plt.plot(t, dgf, label="true dgf for run %d" % run_i)
# plt.xlim(8 * T, 14 * T)
plt.legend()
plt.show()

runs[run_i]["polarity"]


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
# export

# This includes compilation for the shapes of the first and last batch, which are O(1) min
time_per_iter = elapsed.walltime / metrics_tree.i.sum()

results = [
    OpenGlotI.post_process_run(run, metrics, pitch)
    for run, metrics in tqdm(zip(runs, metrics_list))
]


# %%
# Plot best u'(t) fit
print("Best u'(t) fit:")

try:
    i = int(np.nanargmin([r["dgf_both_aligned_nrmse"] for r in results]))
    print(i)

    figs = OpenGlotI.plot_run(
        runs[i], metrics_list[i], pitch, retain_plots=False
    )
except Exception:
    traceback.print_exc()


# %%
# Plot best formant fit
print("Best formant fit:")

try:
    i = int(np.nanargmin([r["formant_rmse"] for r in results]))
    print(i)

    OpenGlotI.plot_run(runs[i], metrics_list[i], pitch, retain_plots=False)
except Exception:
    traceback.print_exc()


# %%
# Plot worst u'(t) fit
print("Worst u'(t) fit:")

try:
    i = int(np.nanargmax([r["dgf_both_aligned_nrmse"] for r in results]))
    print(i)

    OpenGlotI.plot_run(runs[i], metrics_list[i], pitch, retain_plots=False)
except Exception:
    traceback.print_exc()


# %%
# Plot worst formant fit
print("Worst formant fit:")
try:
    i = int(np.nanargmax([r["formant_rmse"] for r in results]))
    print(i)

    OpenGlotI.plot_run(runs[i], metrics_list[i], pitch, retain_plots=False)
except Exception:
    traceback.print_exc()
