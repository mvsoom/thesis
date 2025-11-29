from itertools import product
from random import Random

rng = Random(457899911)


def configurations():
    tack_kernels = ["tack:0", "tack:1", "tack:2", "tack:3"]
    stack_kernels = ["stack:0", "stack:1", "stack:2", "stack:3"]
    matern_kernels = ["matern:12", "matern:32", "matern:52", "matern:inf"]
    periodic_kernel = ["periodickernel"]
    kernels = tack_kernels + stack_kernels + matern_kernels + periodic_kernel

    for (
        Rd,
        open_phase_only,
        kernel,
        centered,
        normalized,
        iteration,
    ) in product(
        [0.3, 1.0, 1.5, 2.0, 2.7],
        [True, False],
        kernels,
        [True, False],
        [True, False],
        [1, 2, 3, 4, 5],
    ):
        if ("tack" not in kernel) and (centered or normalized):
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "Rd": Rd,
            "open_phase_only": open_phase_only,
            "kernel": kernel,
            "centered": centered,
            "normalized": normalized,
            "iteration": iteration,
            "name": f"Rd={Rd}_open_phase_only={open_phase_only}_kernel={kernel}_centered={centered}_normalized={normalized}_iter={iteration}",
        }
