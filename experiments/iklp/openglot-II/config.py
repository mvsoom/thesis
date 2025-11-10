from itertools import product
from random import Random

rng = Random(112879403)


def configurations():
    for (
        jax_enable_x64,
        r,
        beta,
        alpha_scale,
        kappa,
        prior_pi,
        ell,
        target_gender,
    ) in product(
        [True, False],
        [5, 10, 15],
        [0.0, 1.0],
        [0.1, 1.0],
        [0.1, 1.0],
        [0.05, 0.5, 0.95],
        [0.5, 1.0, 2.0],
        ["male", "female"],
    ):
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "jax_enable_x64": jax_enable_x64,
            "r": r,
            "beta": beta,
            "alpha_scale": alpha_scale,
            "kappa": kappa,
            "prior_pi": prior_pi,
            "ell": ell,
            "target_gender": target_gender,
            "name": f"{jax_enable_x64}_{r}_{beta}_{alpha_scale}_{kappa}_{prior_pi}_{ell}_{target_gender}",
        }
