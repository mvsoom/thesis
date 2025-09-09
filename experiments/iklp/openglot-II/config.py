import os
from glob import glob
from itertools import product
from pathlib import Path
from random import Random

rng = Random(112879403)


def wav_files():
    PROJECT_DATA_PATH = os.environ["PROJECT_DATA_PATH"]
    root_path = f"{PROJECT_DATA_PATH}/OPENGLOT/RepositoryII_*"
    paths = glob.glob(f"{root_path}/**/*.wav", recursive=True)
    return paths


def configurations():
    for wav_file, jax_platform_name, jax_enable_64, prior_pi, ell in product(
        wav_files(),
        ["gpu"],
        [False],
        [0.05, 0.5, 0.95],
        [0.5, 1.0, 2.0],
    ):
        stem = Path(wav_file).stem
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "wav_file": wav_file,
            "jax_platform_name": jax_platform_name,
            "jax_enable_64": jax_enable_64,
            "prior_pi": prior_pi,
            "ell": ell,
            "name": f"{stem}_{jax_enable_64}_{prior_pi}_{ell}",
        }
