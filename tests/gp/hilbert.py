# %%

import jax
import matplotlib.pyplot as plt
from tinygp import kernels

from gp.hilbert import *
from gp.mercer import *
from gp.spectral import *
from utils.jax import vk

scale = 1 + 1 / jnp.pi**2

k = Matern52(scale=scale)
kk = kernels.Matern52(scale)

# Test instantations

M = 256
L = 5.0
D = 1


def inputs(N):
    c = 0.8**D
    X = jax.random.uniform(vk(), (N, D), minval=-L * c, maxval=L * c)
    r2 = jnp.sum(X * X, axis=1)
    idx = jnp.argsort(r2)
    return X[idx]


x = inputs(1).squeeze()  # (D,)
h = Hilbert(k, M, L, D=D)
phi = h.compute_phi(x)
weights = h.compute_weights()


# %%
X = inputs(1)

k(X), kk(X), h(X)

# %%
N = 1000

X = inputs(N)

K = h(X, X)

plt.matshow(K)
# %%
plt.matshow(k(X, X))
plt.matshow(kk(X, X))
plt.matshow(h(X, X))

# %%
X = inputs(N)

d = jax.vmap(h.evaluate_diag)(X)

plt.plot(d)
plt.show()

d = jax.vmap(kk.evaluate_diag)(X)

plt.plot(d)
# %%
r = k(X, X) / h(X, X)

plt.matshow(jnp.log10(r))
plt.colorbar()

# %%
plt.hist(r.flatten(), bins=50)
