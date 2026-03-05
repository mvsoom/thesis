from itertools import product
from random import Random

rng = Random(85647219)


def configurations():
    grid = {
        "iteration": range(16),
        "prism": ["iteration=0_M=64_J=8_kernelname=pack:0"],
        "Q": [1, 3, 6, 9],
        "K": [1, 2, 3, 4, 5, 6],
    }

    keys = list(grid)

    for values in product(*(grid[k] for k in keys)):
        d = dict(zip(keys, values))
        d["seed"] = rng.randint(0, 2**32 - 1)
        d["name"] = "_".join(f"{k}={d[k]}" for k in keys)
        yield d
