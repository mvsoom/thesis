# %%
import glob
import os
from itertools import product
from random import Random

rng = Random(3082025)


def wav_files():
    PROJECT_DATA_PATH = os.environ["PROJECT_DATA_PATH"]
    root_path = f"{PROJECT_DATA_PATH}/OPENGLOT/RepositoryI"
    paths = glob.glob(f"{root_path}/**/*.wav", recursive=True)
    return paths


def configurations():
    for wav_file in product(wav_files()):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "wav_file": wav_file,
        }
