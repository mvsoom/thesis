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
"""


def configurations():
    initial_pitchednesses = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    # Ps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]

    for wav_file, initial_pitchedness in product(
        wav_files(), initial_pitchednesses
    ):
        stem = Path(wav_file).stem
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "name": f"{stem}_{initial_pitchedness}",
            "wav_file": wav_file,
            "initial_pitchedness": initial_pitchedness,
        }
