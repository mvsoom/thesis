from itertools import product
from random import Random

rng = Random(334979152)


def configurations():
    for pitch, kernel, refine, gauge in product(
        [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360],
        ["periodickernel", "spack:1"],
        [True, False],
        [True, False],
    ):
        if ("pack" not in kernel) and (refine or gauge):
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "pitch": pitch,
            "kernel": kernel,
            "refine": refine,
            "gauge": gauge,
            "name": f"pitch={pitch}_kernel={kernel}_refine={refine}_gauge={gauge}",
        }
