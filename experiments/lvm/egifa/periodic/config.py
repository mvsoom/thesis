from itertools import product
from random import Random

rng = Random(12346578)


def configurations():
    grid = {
        "Q": [1, 3, 6, 9],
        "M": [4, 8, 16, 32, 64],
    }

    keys = list(grid)

    for values in product(*(grid[k] for k in keys)):
        d = dict(zip(keys, values))
        d["seed"] = rng.randint(0, 2**32 - 1)
        d["name"] = "_".join(f"{k}={d[k]}" for k in keys)
        yield d
