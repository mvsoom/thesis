from itertools import product
from random import Random

rng = Random(9497315)


def configurations():
    grid = {
        "iteration": range(16),
        "M": [64, 32, 16, 8, 4],
        "J": [16, 8, 4, 2, 1],
        "kernelname": ["periodic", "pack:0", "pack:1", "pack:2", "pack:3"],
    }

    keys = list(grid)

    for values in product(*(grid[k] for k in keys)):
        d = dict(zip(keys, values))
        d["seed"] = rng.randint(0, 2**32 - 1)
        d["name"] = "_".join(f"{k}={d[k]}" for k in keys)

        if d["kernelname"] == "periodic":
            if d["J"] == 1:
                d["J"] = 0
            else:
                continue

        yield d
