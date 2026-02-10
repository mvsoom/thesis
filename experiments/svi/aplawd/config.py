from itertools import product
from random import Random

rng = Random(65933849)


def configurations():
    for (
        M,  # number of PRISM basis functions
        iteration,
        kernelname,
    ) in product(
        [4, 8, 16, 32, 64, 128],
        [1],
        ["matern:12", "matern:32", "matern:52", "rbf"],
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "M": M,
            "iteration": iteration,
            "kernelname": kernelname,
            "name": f"M={M}_iter={iteration}_kernelname={kernelname}",
        }