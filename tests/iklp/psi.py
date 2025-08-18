# %%
import jax
from jax import random

from iklp.psi import (
    build_Psi,
    psi_matvec,
    psi_matvec_fft,
    psi_matvec_shift,
    solve_Psi,
)

key = random.PRNGKey(4581)
P = 30
M = 2048
lam = 0.1

key, sub = random.split(key)
x = random.normal(sub, (M,))

key, sub = random.split(key)
a = jax.numpy.sqrt(lam) * random.normal(sub, (P,))

v_true = build_Psi(M, a) @ x

v_shift = psi_matvec_shift(a, x)
v_fft = psi_matvec_fft(a, x)
v_naive = psi_matvec(a, x, mode="naive")

print("shift max-diff:", jax.numpy.max(jax.numpy.abs(v_shift - v_true)))
print("fft   max-diff:", jax.numpy.max(jax.numpy.abs(v_fft - v_true)))
print("naive max-diff:", jax.numpy.max(jax.numpy.abs(v_naive - v_true)))

# Check solve
eps = random.normal(key, (M,))

x_true = jax.numpy.linalg.solve(build_Psi(M, a), eps)
x_fast = solve_Psi(a, eps)
print("solve max-diff:", jax.numpy.max(jax.numpy.abs(x_fast - x_true)))
