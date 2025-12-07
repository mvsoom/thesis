from itertools import product
from random import Random

rng = Random(7983211)


def configurations():
    spacks = ["spack:0", "spack:1", "spack:2", "spack:3"]
    kernels = ["whitenoise", "periodickernel"] + spacks

    for (
        pitch,
        kernel,
        gauge,
        scale_dgf_to_unit_power,
        beta,
        refine,
    ) in product(
        [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360],
        kernels,
        [True, False],
        [True, False],
        [0.0, 1.0],
        [True, False],
    ):
        if ("pack" not in kernel) and (gauge or scale_dgf_to_unit_power):
            continue

        if (not gauge) and scale_dgf_to_unit_power:
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "pitch": pitch,
            "kernel": kernel,
            "gauge": gauge,
            "scale_dgf_to_unit_power": scale_dgf_to_unit_power,
            "beta": beta,
            "refine": refine,
            "name": f"pitch={pitch}_kernel={kernel}_gauge={gauge}_scale_dgf_to_unit_power={scale_dgf_to_unit_power}_beta={beta}_refine={refine}",
        }
