# %%
import jax.numpy as jnp
from tinygp.kernels import Exp, ExpSineSquared, ExpSquared, Matern32, Matern52

from gfm.ack import STACK, TACK


def instantiate_kernel(kernel, theta, hyper={}):
    match kernel:
        case "matern:12":
            k = theta["sigma_a"] ** 2 * Exp(scale=theta["ell"])
        case "matern:32":
            k = theta["sigma_a"] ** 2 * Matern32(scale=theta["ell"])
        case "matern:52":
            k = theta["sigma_a"] ** 2 * Matern52(scale=theta["ell"])
        case "matern:inf":
            k = theta["sigma_a"] ** 2 * ExpSquared(scale=theta["ell"])
        case "periodickernel":
            # Uses Yoshii+ (2013) parametrization (like we do in IKLP experiments):
            #   K = np.exp(-2 * (np.sin(np.pi * tau / T)) ** 2 / (ell**2))
            scale = hyper["T"]
            gamma = 2.0 / theta["ell"] ** 2
            k = theta["sigma_a"] ** 2 * ExpSineSquared(scale=scale, gamma=gamma)
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
                k = theta["sigma_a"] ** 2 * TACK(
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
                "ell": x[2],
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
        "T": 3.2,
        "sigma_b": 3.0,
        "sigma_c": 0.5,
    }

    k = instantiate_kernel(kernel, theta, hyper)

# %%
