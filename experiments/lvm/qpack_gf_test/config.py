from itertools import product
from random import Random

rng = Random(3368974)


def configurations():
    for (
        M,  # number of PRISM basis functions
        Q,  # latent dimension
        iteration,
        d,
    ) in product(
        [16, 32, 64, 128, 256],
        [1, 3, 6, 9, 12, 15],
        [1],
        [0, 1, 2],
    ):
        if M < Q:  # require dimensionality reduction
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "M": M,
            "Q": Q,
            "iteration": iteration,
            "d": d,
            "name": f"M={M}_Q={Q}_iter={iteration}_d={d}",
        }
