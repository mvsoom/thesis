# %%
from random import Random

rng = Random(687891)


def configurations():
    configs = [
        {"N_kernels": 1, "N_ell": 5, "N_data": 1024},
        {"N_kernels": 1, "N_ell": 20, "N_data": 1024},
        {"N_kernels": 1, "N_ell": 50, "N_data": 1024},
        {"N_kernels": 4, "N_ell": 1, "N_data": 1024},
        {"N_kernels": 4, "N_ell": 10, "N_data": 1024},
        {"N_kernels": 4, "N_ell": 20, "N_data": 1024},
        {"N_kernels": 4, "N_ell": 50, "N_data": 512},
        {"N_kernels": 4, "N_ell": 100, "N_data": 512},
    ]

    for config in configs:
        yield {
            "seed": rng.randint(0, 2**32 - 1),
            "name": f"{config['N_kernels']}-{config['N_ell']}-{config['N_data']}",
            **config,
        }