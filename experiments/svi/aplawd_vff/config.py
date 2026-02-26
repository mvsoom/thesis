from itertools import product
from random import Random

rng = Random(3789145)


def configurations():
    iterations = range(16)

    for (
        M,  # number of PRISM basis functions
        iteration,
        kernelname,
    ) in product(
        [4, 8, 16, 32, 64],
        iterations,
        ["rationalquadratic"],
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "M": M,
            "iteration": iteration,
            "kernelname": kernelname,
            "name": f"M={M}_iter={iteration}_kernelname={kernelname}",
        }
