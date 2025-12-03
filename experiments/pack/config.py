from itertools import product
from random import Random

rng = Random(334979152)


def configurations():
    spacks = ["spack:0", "spack:1", "spack:2", "spack:3"]
    kernels = ["periodickernel"] + spacks

    for (
        pitch,
        kernel,
        prior_pi,
        P,
        refine,
        gauge,
        scale_dgf_to_unit_power,
    ) in product(
        [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360],
        kernels,
        [0.5, 0.95],  # neutral and confident expectation of `is_voiced`
        [8, 9],
        [False],
        [True, False],
        [True, False],
    ):
        if ("pack" not in kernel) and (
            refine or gauge or scale_dgf_to_unit_power
        ):
            continue

        if (not gauge) and scale_dgf_to_unit_power:
            continue

        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "pitch": pitch,
            "kernel": kernel,
            "prior_pi": prior_pi,
            "P": P,
            "refine": refine,
            "gauge": gauge,
            "name": f"pitch={pitch}_kernel={kernel}_prior_pi={prior_pi}_P={P}_refine={refine}_gauge={gauge}_scale_dgf_to_unit_power={scale_dgf_to_unit_power}",
        }
