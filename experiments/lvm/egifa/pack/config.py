from itertools import product
from random import Random

rng = Random(68423785)


def configurations():
    grid = {
        "d": [1],
        "Q": [1, 3, 6, 9],
        "M": [16, 32],
        "J": [1, 2, 4, 8],
    }

    keys = list(grid)

    for values in product(*(grid[k] for k in keys)):
        d = dict(zip(keys, values))
        d["seed"] = rng.randint(0, 2**32 - 1)
        d["name"] = "_".join(f"{k}={d[k]}" for k in keys)
        yield d
