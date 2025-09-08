from itertools import product
from pathlib import Path
from random import Random

rng = Random(55497863)


def wav_files():
    probes = [
        "/home/marnix/thesis/data/OPENGLOT/RepositoryI/Vowel_U/U_breathy_100Hz.wav",  # [0]: run 159: top left
        "/home/marnix/thesis/data/OPENGLOT/RepositoryI/Vowel_U/U_normal_180Hz.wav",  # [1]: run 119: top middle
        "/home/marnix/thesis/data/OPENGLOT/RepositoryI/Vowel_E/E_creaky_100Hz.wav",  # [2]: run 105: top right
        "/home/marnix/thesis/data/OPENGLOT/RepositoryI/Vowel_O/O_normal_320Hz.wav",  # [3]: run 245: bottom right
    ]
    return probes  # [probes[3]]


"""
Need to vary still:
- P
- lambda
- lengthscale of kernel r
- frame_length
- x32 vs x64
- noise_floor_db
"""


def configurations():
    noise_floor_dbs = [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0]

    jax_enable_x64s = [True, False]
    jax_platform_names = ["cpu", "gpu"]

    batch_sizes = [1, 2, 4]

    num_metrics_sampless = [1, 5]

    for (
        wav_file,
        noise_floor_db,
        jax_enable_x64,
        jax_platform_name,
        batch_size,
        num_metrics_samples,
    ) in product(
        wav_files(),
        noise_floor_dbs,
        jax_enable_x64s,
        jax_platform_names,
        batch_sizes,
        num_metrics_sampless,
    ):
        stem = Path(wav_file).stem
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "name": f"{stem}_{noise_floor_db}_{jax_enable_x64}_{jax_platform_name}_{batch_size}-{num_metrics_samples}",
            "wav_file": wav_file,
            "noise_floor_db": noise_floor_db,
            "jax_enable_x64": jax_enable_x64,
            "jax_platform_name": jax_platform_name,
            "batch_size": batch_size,
            "num_metrics_samples": num_metrics_samples,
        }
