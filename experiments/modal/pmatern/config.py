from itertools import product
from random import Random

rng = Random(33204897)


def configurations():
    for (
        modality,
        nu,
        M,
        iteration,
    ) in product(
        ["modal", "breathy", "whispery", "creaky"],
        [1 / 2, 3 / 2, 5 / 2],
        [8, 16, 32, 64, 128, 256, 512],
        [1, 2, 3, 4, 5],
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "modality": modality,
            "nu": nu,
            "M": M,
            "iteration": iteration,
            "name": f"modality={modality}_nu={nu}_M={M}_iter={iteration}",
        }
