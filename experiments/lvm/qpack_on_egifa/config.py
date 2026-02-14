from itertools import product
from random import Random

rng = Random(456789132)


def configurations():
    for (
        M,  # number of PRISM basis functions
        Q,  # latent dimension
        iteration,
        d,
        am,
    ) in product(
        [16, 32, 64, 128],
        [1, 3, 6, 9, 12],
        [1],
        [1],
        ["rbf"],  # ["rbf", "rationalquadratic"],
    ):
        if M < Q:  # require dimensionality reduction
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "M": M,
            "Q": Q,
            "iteration": iteration,
            "d": d,
            "am": am,
            "name": f"M={M}_Q={Q}_iter={iteration}_d={d}_am={am}",
        }
