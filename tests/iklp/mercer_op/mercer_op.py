# %%
from iklp import mercer
from iklp.hyperparams import random_periodic_kernel_hyperparams
from iklp.mercer_op import *

master_key = jax.random.PRNGKey(10)


def vk():
    global master_key
    master_key, subkey = jax.random.split(master_key)
    return subkey


# Mock data and hyperparameters
I = 50
M = 500
P = 30
nu = 1.56
lam = 0.1

kernel_kwargs = {
    "noise_floor_db": -jnp.inf,
}

hyper_kwargs = {
    "P": P,
    "lam": lam,
}

h, K = random_periodic_kernel_hyperparams(
    vk(), I, M, kernel_kwargs, hyper_kwargs, return_K=True
)

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
# As noise_floor_db increases, calculations become more exact while Woodbury becomes less efficient
errs = []
db = 0.0

for _ in range(4):
    db -= 20.0

    print()
    print(f"Noise floor: {db:.0f} dB")

    Phi = mercer.psd_svd(K, noise_floor_db=db)
    h = h.replace(Phi=Phi)
    print("Phi shape:", Phi.shape)

    op = build_operator(nu, coeff, data)

    err = [
        jnp.max(jnp.abs(solve(op, v) - Sinv_explicit @ v)),
        jnp.abs(logdet(op) - logdet_exp),
        jnp.abs(trinv(op) - trinv_exp),
        jnp.max(jnp.abs(trinv_Ki(op) - trinv_Ki_exp)),
        jnp.max(jnp.abs(solve_normal_eq(op, lam) - a_exp)),
    ]
    errs.append(err)

    print("\tmax |S⁻¹v - exact|:", err[0])
    print("\tlogdet diff:", err[1])
    print("\ttrinv   diff:", err[2])
    print("\ttrinv_Ki max diff:", err[3])
    print("\ta max diff:", err[4])

# %%
