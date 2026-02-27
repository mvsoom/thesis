from itertools import product
from random import Random

rng = Random(65479472)


def configurations():
    iterations = range(16)

    for (
        M,  # number of PRISM basis functions
        iteration,
    ) in product(
        [4, 8, 16, 32, 64],
        iterations,
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "M": M,
            "iteration": iteration,
            "name": f"M={M}_iter={iteration}",
        }
