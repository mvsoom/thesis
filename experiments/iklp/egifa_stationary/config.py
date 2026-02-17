from itertools import product
from random import Random

rng = Random(96374851)


def configurations():
    grid = {
        "collection": ["vowel", "speech"],
        "kernel": [
            "whitenoise",
            "periodickernel",
            "pack:0",
            "pack:1",
            "pack:2",
            "pack:3",
        ],
    }

    keys = list(grid)

    for values in product(*(grid[k] for k in keys)):
        d = dict(zip(keys, values))
        d["seed"] = rng.randint(0, 2**32 - 1)
        d["name"] = "_".join(f"{k}={d[k]}" for k in keys)
        yield d
