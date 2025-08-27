# %%
from itertools import product
from random import Random

rng = Random(1560050)


def configurations():
    for (i,) in product(range(0, 2)):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "i": i,
        }
