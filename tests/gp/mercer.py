# %%

import jax

from gp.hilbert import *
from gp.mercer import *
from gp.spectral import *
from utils.jax import vk

scale = 1 + 1 / jnp.pi**2

kernel = Matern52(scale=scale)
l = Matern32(scale=1 / scale)

# Test instantations

M = 64
L = 5.0
D = 1


def inputs(N):
    c = 0.8**D
    X = jax.random.uniform(vk(), (N, D), minval=-L * c, maxval=L * c)
    r2 = jnp.sum(X * X, axis=1)
    idx = jnp.argsort(r2)
    return X[idx]


N = 100

X = inputs(N).squeeze()  # (N, D)

h = Hilbert(kernel, M, L, D=D)
g = Hilbert(l, M, L, D=D)

f = g * 5 + 2 * h

ff = g * h

K = f(X)

K

# %%
Y = inputs(1)

ff(Y)
# %%
kx = Hilbert(Matern32(), M, L, D=2)
ky = Hilbert(Matern32(), M, L, D=2)
kxy = Hilbert(Matern32(), M, L, D=2)

# Hilbert addition
kx + ky

# Hilbert multiplication
kx * ky

# Mercer stack of 1D kernels via Subspace.__add__
Subspace(0, kx) + Subspace(1, ky)

# Mercer stack via Hilbert.__add__ and/or Subspace.__add__
Subspace(0, kx) + kxy

# Mercer product
Subspace(0, kx) * kxy
