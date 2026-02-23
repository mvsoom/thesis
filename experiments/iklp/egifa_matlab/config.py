from itertools import product
from random import Random

rng = Random(5798432)


def configurations():
    grid = {
        "collection": ["vowel", "speech"],
        "egifa_f0": [90, 120, 150, 180, 210],
        "method": ["null", "iaif", "cp", "wca1", "wca2"],
    }

    keys = list(grid)

    for values in product(*(grid[k] for k in keys)):
        d = dict(zip(keys, values))
        d["name"] = "_".join(f"{k}={d[k]}" for k in keys)
        yield d
