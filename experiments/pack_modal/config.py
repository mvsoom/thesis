from itertools import product
from random import Random

rng = Random(33978946)


def configurations():
    pack_kernels = ["pack:0", "pack:1", "pack:2", "pack:3"]

    for (
        modality,
        kernel,
        normalized,
        single_sigma_c,
        J,
        iteration,
    ) in product(
        ["modal", "breathy", "whispery", "creaky"],
        pack_kernels,
        [True, False],
        [True, False],
        [1, 2, 4, 8, 16, 32],
        [1],  # [1, 2, 3, 4, 5],
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "modality": modality,
            "kernel": kernel,
            "normalized": normalized,
            "single_sigma_c": single_sigma_c,
            "J": J,
            "iteration": iteration,
            "name": f"modality={modality}_kernel={kernel}_normalized={normalized}_single_sigma_c={single_sigma_c}_J={J}_iter={iteration}",
        }
