# %%
import jax

from iklp.hyperparams import ARPrior, Hyperparams
from iklp.mercer_op import *
from iklp.mercer_op import build_data
from utils.jax import vk

# Mock data and hyperparameters
I = 16
M = 128
r = 8
P = 15
nu = 0.5
lam = 0.5

arprior = ARPrior.yoshii_lambda(P=P, lam=lam)

# Generate I random posdef matrices
Phi = jax.random.normal(vk(), (I, M, r))
K = Phi @ jnp.swapaxes(Phi, -1, -2)  # (I,M,M)

h = Hyperparams(Phi, arprior=arprior)

x = jax.random.normal(vk(), (M,))
coeff = jax.random.normal(vk(), (I,)) ** 2
v = jax.random.normal(vk(), (M,))
data_ref = build_data(x, h)

print("K shape:", K.shape)

# Compute ground truth
S = nu * jnp.eye(M)
for i in range(I):
    S += coeff[i] * K[i]

Sinv_explicit = jla.inv(S)
logdet_exp = 2 * jnp.sum(jnp.log(jnp.diag(jla.cholesky(S, lower=True))))
trinv_exp = jnp.trace(Sinv_explicit)
trinv_Ki_exp = jnp.trace(Sinv_explicit @ K, axis1=1, axis2=2)


def compute_naive_a(S_inv, x, X, arprior):
    """Match solve_normal_eq() convention: (XᵀS⁻¹X + Q)a = XᵀS⁻¹x + Qμ."""
    G = X.T @ S_inv @ X
    r = X.T @ S_inv @ x
    Q = arprior.precision
    mu = arprior.mean
    return jnp.linalg.solve(G + Q, r + Q @ mu)


a_exp = compute_naive_a(Sinv_explicit, data_ref.x, data_ref.X, arprior)

# %%
mercer_backends = ["cholesky", "woodbury"]

print(
    "IMPORTANT: beefing up the diagonal in safe_cholesky() will cause numerical discrepancies! This is set by Hyperparams.beta"
)

for backend in mercer_backends:
    h_backend = h.replace(mercer_backend=backend)
    data = build_data(x, h_backend)
    op = build_operator(nu, coeff, data)

    abs_errs = [
        jnp.max(jnp.abs(solve(op, v) - Sinv_explicit @ v)),
        jnp.abs(logdet(op) - logdet_exp),
        jnp.abs(trinv(op) - trinv_exp),
        jnp.max(jnp.abs(trinv_Ki(op) - trinv_Ki_exp)),
        jnp.max(jnp.abs(solve_normal_eq(op, arprior) - a_exp)),
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
