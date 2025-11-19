# %%
import contextlib

import gnuplotlib as gp
import matplotlib.pyplot as plt
import numpy as np

# %%

to = 0
tp = te = 5.5
tc = 7

T0 = tc + 1

K = 2
t = np.array([to, tp])

x = np.linspace(-0.01, T0, 500)
u = np.piecewise(
    x,
    [x <= to, (x >= to) & (x < tp), (x >= tp) & (x <= tc), x > tc],
    [0, lambda x: (x - to) / (tp - to), lambda x: (tc - x) / (tc - tp), 0],
)

du = np.piecewise(
    x,
    [x <= to, (x >= to) & (x < tp), (x >= tp) & (x <= tc), x > tc],
    [0, 1 / (tp - to), -1 / (tc - tp), 0],
)

plt.plot(x, u * 0)
plt.plot(x, u)
plt.plot(x, du)


curves = [
    (x, du, {"legend": "du", "with": "lines"}),
    (x, u, {"legend": "u", "with": "lines"}),
]

with open("du.gp", "w") as f, contextlib.redirect_stdout(f):
    gp.plot(*curves, ascii=True, dump=True, notest=True)
# %%


# %%
def H(x, n):
    return np.heaviside(x, 1) * x**n


tp = np.random.uniform(te, to)
t = np.array([to, tp])

t = np.random.uniform(te, to, size=500)

n = 3  # higher n, more look like LF
sigma_a = 1.0
b = 1 / (n + 1) * (T - t) ** (n + 1)
q = b / np.linalg.norm(b)
I = np.eye(len(t))
Sigma = sigma_a**2 * (I - np.outer(q, q))

x = np.linspace(0, T + 1, 1000)

# Sigma not full rank, so use SVD to sample from it
U, S, Vt = np.linalg.svd(Sigma)
z = np.random.normal(0, sigma_a, size=len(t))
a = U @ np.diag(np.sqrt(S)) @ z

du = (a * H(x[:, None] - t[None, :], n)).sum(axis=1) * ((to <= x) & (x <= te))
u = np.cumsum(du) * (x[1] - x[0])

plt.plot(x, du)
plt.plot(x, u)

# %%
