from itertools import product
from random import Random

rng = Random(1145930)


def configurations():
    for (
        M,  # number of PRISM basis functions
        Q,  # latent dimension
        iteration,
    ) in product(
        [8, 16, 32, 64, 128],
        [1, 3, 6, 9, 12],
        [1],
    ):
        if M < Q:  # require dimensionality reduction
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "M": M,
            "Q": Q,
            "iteration": iteration,
            "name": f"M={M}_Q={Q}_iter={iteration}",
        }