import os
from itertools import product
from random import Random

rng = Random(23789546)


HERE = os.path.dirname(os.path.abspath(__file__))


def configurations():
    tack_kernels = ["tack:0", "tack:1", "tack:2", "tack:3"]
    matern_kernels = ["matern:12", "matern:32", "matern:52", "matern:inf"]
    sqexp_kernels = ["periodickernel"]
    kernels = tack_kernels + matern_kernels + sqexp_kernels

    for (
        examplar_name,
        d,
        kernel,
        centered,
        normalized,
        iteration,
    ) in product(
        ["hard_gci", "soft_gci"],
        [0, 1, 2, 3, 100],
        kernels,
        [True, False],
        [True, False],
        [1],  # TODO: [1, 2, 3],
    ):
        if ("tack" not in kernel) and (centered or normalized):
            continue

        data_filename = "lf.dat" if d == 100 else f"d={d}.dat"
        data_file = f"data/{examplar_name}/{data_filename}"
        data_file_fullpath = os.path.join(HERE, data_file)

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "examplar_name": examplar_name,
            "d": d,
            "kernel": kernel,
            "centered": centered,
            "normalized": normalized,
            "data_file": data_file_fullpath,
            "iteration": iteration,
            "name": f"{examplar_name}_d={d}_kernel={kernel}_centered={centered}_normalized={normalized}_iter={iteration}",
        }
