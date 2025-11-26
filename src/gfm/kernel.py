# %%
import jax.numpy as jnp
from tinygp.kernels import Exp, ExpSineSquared, ExpSquared, Matern32, Matern52

from gfm.ack import STACK, TACK


def instantiate_kernel(kernel, theta, hyper={}):
    match kernel:
        case "matern:12":
            k = theta["sigma_a"] * Exp(scale=theta["ell"])
        case "matern:32":
            k = theta["sigma_a"] * Matern32(scale=theta["ell"])
        case "matern:52":
            k = theta["sigma_a"] * Matern52(scale=theta["ell"])
        case "matern:inf":
            k = theta["sigma_a"] * ExpSquared(scale=theta["ell"])
        case "periodickernel":
            # Parametrization (r, T) agrees with src.iklp.periodic.periodic_kernel_generator() [but the latter calculates the time indices t differently; we have PERIOD inclusive and the latter exclusive]
            r = theta["r"]
            T = hyper["T"]
            gamma = 1.0 / (2.0 * r**2)
            k = theta["sigma_a"] * ExpSineSquared(scale=T, gamma=gamma)
        case _ if "tack" in kernel:
            d = int(kernel[-1])
            normalized = hyper.get("normalized", False)
            centered = hyper.get("centered", False)

            center = hyper["center"] if centered else 0.0

            if "stack" in kernel:
                k = STACK(d=d, normalized=normalized, center=center)
            else:
                LSigma = jnp.diag(
                    jnp.array([theta["sigma_b"], theta["sigma_c"]])
                )
                k = theta["sigma_a"] * TACK(
                    d=d, normalized=normalized, center=center, LSigma=LSigma
                )
        case _:
            raise NotImplementedError(f"Kernel {kernel} not implemented")

    return k


def build_theta(x, kernel):
    match kernel:
        case _ if "matern" in kernel:
            return {
                "sigma_noise": x[0],
                "sigma_a": x[1],
                "ell": x[2],
            }
        case "periodickernel":
            return {
                "sigma_noise": x[0],
                "sigma_a": x[1],
                "r": x[2],
            }
        case _ if "stack" in kernel:
            return {
                "sigma_noise": x[0],
            }
        case _ if "tack" in kernel:
            return {
                "sigma_noise": x[0],
                "sigma_a": x[1],
                "sigma_b": x[2],
                "sigma_c": x[3],
            }
        case _:
            raise NotImplementedError(f"Kernel {kernel} not implemented")


if __name__ == "__main__":
    kernel = "tack:2"

    hyper = {
        "normalized": True,
    }

    theta = {
        "sigma_a": 5.0,
        "ell": 1.789,
        "r": 0.78113212,
        "sigma_b": 3.0,
        "sigma_c": 0.5,
    }

    k = instantiate_kernel(kernel, theta, hyper)
