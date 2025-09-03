# %%
import jax

from iklp.hyperparams import Hyperparams, KrylovParams
from iklp.mercer_op import build_X, build_data

# Turn off compilation logging
jax.config.update("jax_log_compiles", False)

import jax.numpy as jnp

from iklp.mercer_op.krylov import *

master_key = jax.random.PRNGKey(0)


def vk():
    global master_key
    master_key, subkey = jax.random.split(master_key)
    return subkey


# Mock data and hyperparameters
I = 20
M = 1048
r = 8
P = 30
nu = 1.56
lam = 0.1

hyper_kwargs = {
    "P": P,
    "lam": lam,
}

# Generate I random posdef matrices
Phi = jax.random.normal(vk(), (I, M, r))
K = Phi @ jnp.swapaxes(Phi, -1, -2)  # (I,M,M)

h = Hyperparams(Phi, **hyper_kwargs)

print("K shape:", K.shape)

x = jax.random.normal(vk(), (M,))
coeff = jax.random.normal(vk(), (I,)) ** 2
v = jax.random.normal(vk(), (M,))

# Compute ground truth
X = build_X(x, P)
S = nu * jnp.eye(M)
for i in range(I):
    S += coeff[i] * K[i]

Sinv_explicit = jla.inv(S)
logdet_exp = 2 * jnp.sum(jnp.log(jnp.diag(jla.cholesky(S, lower=True))))
trinv_exp = jnp.trace(Sinv_explicit)
trinv_Ki_exp = jnp.trace(Sinv_explicit @ K, axis1=1, axis2=2)


def compute_naieve_a(S_inv, X, P, lam):
    """Uses Sigma^(-1) == S^(-1) -- see _compute_sigma_inv_with_upsilon()"""
    A = X.T.dot(S_inv).dot(X) + lam * jnp.eye(P)
    b = X.T.dot(S_inv).dot(x)
    a = jnp.linalg.solve(A, b)
    return a


a_exp = compute_naieve_a(Sinv_explicit, X, P, lam)

# %%
build_operator = jax.jit(build_operator)
solve = jax.jit(solve)
logdet = jax.jit(logdet)
trinv = jax.jit(trinv)
trinv_Ki = jax.jit(trinv_Ki)
solve_normal_eq = jax.jit(solve_normal_eq)

# %%
print("Phi shape:", Phi.shape)

h = h.replace(krylov=h.krylov.replace(key=vk()))

data = build_data(x, h)

op = build_operator(nu, coeff, data)

err = [
    jnp.max(jnp.abs(solve(op, v) - Sinv_explicit @ v)),
    jnp.abs(logdet(op) - logdet_exp),
    jnp.abs(trinv(op) - trinv_exp),
    jnp.max(jnp.abs(trinv_Ki(op) - trinv_Ki_exp)),
    jnp.max(jnp.abs(solve_normal_eq(op, lam) - a_exp)),
]

print("\tmax |S⁻¹v - exact|:", err[0])
print("\tlogdet diff:", err[1])
print("\ttrinv   diff:", err[2])
print("\ttrinv_Ki max diff:", err[3])
print("\ta max diff:", err[4])

# %%

# %%
%timeit build_operator(nu, coeff, data).sketch.Z.block_until_ready()
# %%
%timeit trinv_Ki(op).block_until_ready()
# %%
%timeit solve_normal_eq(op, lam).block_until_ready()

#%%

## latest

	max |S⁻¹v - exact|: 3.60208371377041e-11
	logdet diff: 15.389114149628767
	trinv   diff: 0.020679429636852653
	trinv_Ki max diff: 24.099367365230364
	a max diff: 5.357124802899023e-10

## default op

	max |S⁻¹v - exact|: 3.60208371377041e-11
	logdet diff: 3.6236310242120453
	trinv   diff: 0.008811100214131562
	trinv_Ki max diff: 43.68383465613579
	a max diff: 5.357124802899023e-10

## p = 64, m = 256

max |S⁻¹v - exact|: 3.60208371377041e-11
	logdet diff: 3.676238342738543
	trinv   diff: 0.004758805568240154
	trinv_Ki max diff: 9.539404530139564
	a max diff: 5.357124802899023e-10


## p = 128, m = 96

	max |S⁻¹v - exact|: 3.60208371377041e-11
	logdet diff: 0.6411150439344055
	trinv   diff: 0.0003794705239485996
	trinv_Ki max diff: 22.33945765773933
	a max diff: 5.357124802899023e-10


fallbacks:
	logdet SLQ diff: 2.847686300348869
	trinv SLQ   diff: 0.012258553174772091
	trinv_Ki Hutch max diff: 30.37643268118586

## +preconditioner

max |S⁻¹v - exact|: 3.60208371377041e-11
	logdet diff: 1.9018406250579574
	trinv   diff: 0.0018062686416380203
	trinv_Ki max diff: 9.495849500129452
	a max diff: 5.357124802899023e-10


fallbacks:
	logdet SLQ diff: 3.789614495322894
	trinv SLQ   diff: 0.02112541232290499
	trinv_Ki Hutch max diff: 11.306138940150959