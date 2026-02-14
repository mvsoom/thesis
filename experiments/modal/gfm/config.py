from itertools import product
from random import Random

rng = Random(69317545)


def configurations():
    tack_kernels = ["tack:0", "tack:1", "tack:2", "tack:3"]
    stack_kernels = ["stack:0", "stack:1", "stack:2", "stack:3"]
    matern_kernels = ["matern:12", "matern:32", "matern:52", "matern:inf"]
    periodic_kernel = ["periodickernel"]
    kernels = tack_kernels + stack_kernels + matern_kernels + periodic_kernel

    for (
        modality,
        kernel,
        centered,
        normalized,
        iteration,
    ) in product(
        ["modal", "breathy", "whispery", "creaky"],
        kernels,
        [True, False],
        [True, False],
        [1, 2, 3, 4, 5],
    ):
        if ("tack" not in kernel) and (centered or normalized):
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "modality": modality,
            "kernel": kernel,
            "centered": centered,
            "normalized": normalized,
            "iteration": iteration,
            "name": f"modality={modality}_kernel={kernel}_centered={centered}_normalized={normalized}_iter={iteration}",
        }
