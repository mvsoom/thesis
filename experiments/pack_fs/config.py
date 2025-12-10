from itertools import product
from random import Random

rng = Random(57624383)


def configurations():
    spacks = ["spack:0", "spack:1", "spack:2", "spack:3"]
    kernels = ["periodickernel"] + spacks

    for (
        pitch,
        kernel,
        gauge,
        scale_dgf_to_unit_power,
        window_type,
        P,
    ) in product(
        [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360],
        kernels,
        [True, False],
        [True, False],
        ["iklp", "iaif", "adaptive"],
        [8, 10],
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
            "window_type": window_type,
            "P": P,
            "name": f"pitch={pitch}_kernel={kernel}_gauge={gauge}_scale_dgf_to_unit_power={scale_dgf_to_unit_power}_window_type={window_type}_P={P}",
        }
