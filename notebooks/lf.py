# %%
import contextlib

import gnuplotlib as gp
import jax.numpy as jnp
from matplotlib import pyplot as plt

from utils.lfmodel import dgf

to = 0
tm = 4
te = 5.5
tc = 7

T0 = tc + 1

p = {
    "T0": tc,
    "Te": te,
    "Tp": tm,
    "Ta": 0.15,
}

t = jnp.linspace(0, T0, 1000)

du = dgf(t, p)
u = jnp.cumsum(du) * (t[1] - t[0])

plt.plot(t, du)
plt.plot(t, u)


curves = [
    (t, du, {"legend": "du", "with": "lines"}),
    (t, u, {"legend": "u", "with": "lines"}),
]

with open("lf.gp", "w") as f, contextlib.redirect_stdout(f):
    gp.plot(*curves, ascii=True, dump=True, notest=True)