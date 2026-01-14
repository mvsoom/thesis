from itertools import product
from random import Random

rng = Random(87623178)


def configurations():
    pack_kernels = ["pack:0", "pack:1", "pack:2", "pack:3"]
    sample_idxs = range(100)

    for (
        kernel,
        J,
        sample_idx,
    ) in product(
        pack_kernels,
        [1, 2, 4, 8, 16],
        sample_idxs,
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "kernel": kernel,
            "J": J,
            "sample_idx": sample_idx,
            "name": f"kernel={kernel}_J={J}_sample_idx={sample_idx}",
        }
