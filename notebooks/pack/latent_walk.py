# %%
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gfm.ack import DiagonalTACK
from gp.blr import blr_from_mercer
from pack import PACK
from utils.jax import vk


def sample_latent(blr, key):
    """z ~ N(0, I_R), f = Phi @ (cov_root @ z)."""
    R = blr.cov_root.shape[1]
    z = jax.random.normal(key, (R,))
    return z


def sample(blr, z):
    Phi = blr.state.Phi
    f = Phi @ (blr.cov_root @ z) + Phi @ gp.mu
    return f


def make_gif(gp, t, key, n_frames=100, step=1e-2, out="latent_walk.gif"):
    z0 = sample_latent(gp, key)

    frames = []
    z = z0

    fig, ax = plt.subplots()

    for k in tqdm(range(n_frames)):
        key, subkey = jax.random.split(key)
        z_noise = sample_latent(gp, subkey)

        # latent update
        z = z * (1.0 - step) + z_noise * step

        f = sample(gp, z)
        u = np.cumsum(f) * (t[1] - t[0])

        ax.clear()
        ax.plot(t, np.asarray(u))
        ax.set_ylim(-1, 1)
        ax.set_title(f"step {k}")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[..., :3].copy()
        frames.append(frame)

    plt.close(fig)

    imageio.mimsave(out, frames, fps=20)
    return out


t = jnp.linspace(0, 20, 1024)

k = DiagonalTACK(
    d=1,
    normalized=True,
    sigma_b=0.1,
    sigma_c=1.0,
    center=5.0,
)

pack = PACK(
    k,
    period=10.0,
    t1=0.0,
    t2=7.5,
    J=128,
)

gp = blr_from_mercer(pack, t)
f = gp.sample(vk())
plt.plot(t, f)

key = vk()
gif_path = make_gif(gp, t, key, step=0.025, n_frames=500)
print("saved to", gif_path)
