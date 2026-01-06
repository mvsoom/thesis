from itertools import product
from random import Random

rng = Random(12456423)


def configurations():
    pack_kernels = ["pack:0", "pack:1", "pack:2", "pack:3"]

    for (
        modality,
        kernel,
        normalized,
        effective_num_harmonics,
        iteration,
    ) in product(
        ["modal", "breathy", "whispery", "creaky"],
        pack_kernels,
        [True, False],
        [0.5, 0.75, 0.95, 1.25, 1.5],
        [1],  # [1, 2, 3, 4, 5],
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "modality": modality,
            "kernel": kernel,
            "normalized": normalized,
            "effective_num_harmonics": effective_num_harmonics,
            "iteration": iteration,
            "name": f"modality={modality}_kernel={kernel}_normalized={normalized}_effective_num_harmonics={effective_num_harmonics}_iter={iteration}",
        }
