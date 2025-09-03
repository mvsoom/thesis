# %%
import jax

from iklp.mercer_op import build_data, build_X

# Turn off compilation logging
jax.config.update("jax_log_compiles", False)

from iklp.hyperparams import Hyperparams
from iklp.mercer_op import *

master_key = jax.random.PRNGKey(0)


def vk():
    global master_key
    master_key, subkey = jax.random.split(master_key)
    return subkey


# Mock data and hyperparameters
I = 400
M = 1048
r = 10
P = 30
nu = 0.5
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
mercer_backends = ["krylov"]  # ["cholesky", "woodbury", "krylov"]

for backend in mercer_backends:
    h = h.replace(mercer_backend=backend, krylov=h.krylov.replace(key=vk()))
    data = build_data(x, h)
    op = build_operator(nu, coeff, data)

    abs_errs = [
        jnp.max(jnp.abs(solve(op, v) - Sinv_explicit @ v)),
        jnp.abs(logdet(op) - logdet_exp),
        jnp.abs(trinv(op) - trinv_exp),
        jnp.max(jnp.abs(trinv_Ki(op) - trinv_Ki_exp)),
        jnp.max(jnp.abs(solve_normal_eq(op, lam) - a_exp)),
    ]

    ref_vals = [
        jnp.max(jnp.abs(Sinv_explicit @ v)),
        jnp.abs(logdet_exp),
        jnp.abs(trinv_exp),
        jnp.max(jnp.abs(trinv_Ki_exp)),
        jnp.max(jnp.abs(a_exp)),
    ]

    rel_errs = [
        (ae / rv * 100.0) if rv != 0 else jnp.nan
        for ae, rv in zip(abs_errs, ref_vals)
    ]

    labels = [
        "max |S⁻¹v - exact|",
        "logdet diff",
        "trinv diff",
        "trinv_Ki max diff",
        "a max diff",
    ]

    print(f"\nBackend: {backend}")
    for lbl, ae, re in zip(labels, abs_errs, rel_errs):
        print(f"  {lbl:<22s} abs: {float(ae):.3e}   rel: {float(re):6.3f}%")

# %%
