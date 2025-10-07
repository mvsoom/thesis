# %%
import contextlib

import gnuplotlib as gp
import jax.numpy as jnp
from matplotlib import pyplot as plt

from utils.lfmodel import dgf

to = 1
tm = 4.5
te = 6.5
tc = 7

p = {
    "T0": tc,
    "Te": te - to,
    "Tp": tm - to,
    "Ta": 0.2,
}

t = jnp.linspace(0, tc + 1, 1000)

du = dgf(t, p, offset=to)
u = jnp.cumsum(du) * (t[1] - t[0])

plt.plot(t, du)
plt.plot(t, u)


curves = [
    (t, du, {"legend": "du", "with": "lines"}),
    (t, u, {"legend": "u", "with": "lines"}),
]

with open("lf.gp", "w") as f, contextlib.redirect_stdout(f):
    gp.plot(*curves, ascii=True, dump=True, notest=True)
