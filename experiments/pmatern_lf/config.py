from itertools import product
from random import Random

rng = Random(33365478)


def configurations():
    nus = [1 / 2, 3 / 2, 5 / 2, 100]
    sample_idxs = range(100)

    for (
        nu,
        M,
        sample_idx,
    ) in product(
        nus,
        [16, 32, 64, 128, 256, 512],
        sample_idxs,
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "nu": nu,
            "M": M,
            "sample_idx": sample_idx,
            "name": f"nu={nu}_M={M}_sample_idx={sample_idx}",
        }
