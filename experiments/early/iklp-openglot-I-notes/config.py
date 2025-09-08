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
    initial_pitchedness = 0.99
    noise_floor_dbs = [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0]
    # Ps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]

    for wav_file, noise_floor_db in product(wav_files(), noise_floor_dbs):
        stem = Path(wav_file).stem
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "name": f"{stem}_{noise_floor_db}",
            "wav_file": wav_file,
            "initial_pitchedness": initial_pitchedness,
            "noise_floor_db": noise_floor_db,
        }
