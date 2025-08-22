# %% Stable‑AR prior demo (JAX) — round‑trip, Jacobian, differentiable stability

"""
Demonstrates
1. Gaussian prior  a ~ N(0, λ(P) I)  with  λ(P)=c/P,
2. **Exact** Levinson–Durbin forward & backward recursions (plain Python loops, fully differentiable),
3. differentiable stability score  s(a)=1−max(|φ_p(a)|),
4. round‑trip accuracy plus Jacobian / Hessian of φ(a) at the origin.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jacobian

jax.config.update("jax_enable_x64", True)

# ────────────────────────────────────────────────────────────────
# Prior variance schedule  λ(P)=c/P
# ────────────────────────────────────────────────────────────────


def lambda_P(P: int, c: float = 0.3) -> float:
    return c / float(P)


# ────────────────────────────────────────────────────────────────
# Levinson–Durbin  (PACF  ↔  AR) — textbook indices
# ────────────────────────────────────────────────────────────────


def pacf_to_ar(phi: jnp.ndarray) -> jnp.ndarray:
    """Partial autocorrelations → AR coefficients (stable)."""
    P = int(phi.shape[0])
    a = jnp.zeros(P)
    for m in range(1, P + 1):  # 1 … P
        k_m = phi[m - 1]  # φ_m
        if m > 1:
            # update first m‑1 coef: a_i ← a_i − k_m·a_{m−i}
            head = a[: m - 1] - k_m * a[m - 2 :: -1]
            a = a.at[: m - 1].set(head)
        a = a.at[m - 1].set(k_m)  # a_m = φ_m
    return a


def ar_to_pacf(a: jnp.ndarray) -> jnp.ndarray:
    """AR coefficients → partial autocorrelations."""
    P = int(a.shape[0])
    phi = jnp.zeros(P)
    a_curr = a
    for m in range(P, 0, -1):  # P … 1
        k_m = a_curr[m - 1]  # φ_m
        phi = phi.at[m - 1].set(k_m)
        if m > 1:
            denom = 1.0 - k_m**2
            head = (a_curr[: m - 1] + k_m * a_curr[m - 2 :: -1]) / denom
            a_curr = head  # drop last coef for next iter
    return phi


# ────────────────────────────────────────────────────────────────
# Differentiable stability score; positive ⇒ stable
# ────────────────────────────────────────────────────────────────


def stability_score(a: jnp.ndarray) -> jnp.ndarray:
    return 1.0 - jnp.max(jnp.abs(ar_to_pacf(a)))


# ────────────────────────────────────────────────────────────────
# Demo for several orders P
# ────────────────────────────────────────────────────────────────
for P in [1, 2, 3, 5, 30]:
    print(f"\n=== Order  P = {P} ===")
    lam = lambda_P(P)
    key = jr.PRNGKey(P)
    a_prior = jr.normal(key, (P,)) * jnp.sqrt(lam)

    phi = ar_to_pacf(a_prior)
    a_round = pacf_to_ar(phi)

    print("λ(P)                 =", lam)
    print("a  (prior draw)      =", a_prior)
    print("φ(a)                 =", phi)
    print("round‑trip ‖Δa‖∞      =", float(jnp.max(jnp.abs(a_round - a_prior))))
    print("stability score s(a) =", float(stability_score(a_prior)))

# ────────────────────────────────────────────────────────────────
# Jacobian & Hessian of φ(a) at the origin
# ────────────────────────────────────────────────────────────────
# %%
key = jr.PRNGKey(5645)
for P in [1, 2, 3, 4, 5]:
    a = jr.normal(key, (P,))
    J0 = jacobian(ar_to_pacf)(a)
    # H0 = hessian(ar_to_pacf)(a)
    print(f"\nJacobian of φ at a = {a} (P={P}):\n", J0)
    # print log abs det
    print(f"log det abs J0 at a = {a}", jnp.linalg.slogdet(jnp.abs(J0))[1])
    # print("\nHessian of φ at a = 0 (≈0):\n", H0)

# %%
for P in [2, 3, 4, 5]:
    key = jr.PRNGKey(P)
    phi = jr.uniform(key, (P,), minval=-0.9, maxval=0.9)
    J_fwd = jax.jacfwd(pacf_to_ar)(phi)  # ∂a/∂φ
    print(P, jnp.linalg.slogdet(J_fwd)[1])  # always 0.0 (= log 1)

# %%

for P in range(2, 6):
    key = jr.PRNGKey(P)
    phi = jr.uniform(key, (P,), minval=-0.9, maxval=0.9)
    J = jax.jacfwd(pacf_to_ar)(phi)  # ∂a/∂φ
    print(
        P,
        jnp.linalg.slogdet(J)[1],
        jnp.log(jnp.prod((1 - phi[1:] ** 2) ** jnp.arange(1, P))),
    )


# %%

P = 4
lam = 0.3 / P
beta = 1.0
key = jr.PRNGKey(0)
a = jr.normal(key, (P,)) * 0.05  # tiny step from 0


def L(a):
    phi = ar_to_pacf(a)  # your backward LD
    return jnp.sum(jnp.log1p(-(phi**2)))


print("L(a) ≈", L(jnp.zeros(P)), "  (should be 0)")
print("∇L(0) ≈", jax.grad(L)(jnp.zeros(P)))  # zero
print("H(0)  ≈\n", jax.hessian(L)(jnp.zeros(P)))
# should print ≈ -2 * I

# %%

P, N = 12, int(1e6)  # lots of uniform PACF samples
key = jr.PRNGKey(0)
phi = jr.uniform(key, (N, P), minval=-1, maxval=1)  # uniform in cube
a = jax.vmap(pacf_to_ar)(phi)  # map to AR space
Cq = jnp.cov(a.T)  # empirical Σ*

print("μ ≈", jnp.mean(a, 0))  # ≈ 0
print("diag Σ* ≈", jnp.diag(jnp.atleast_2d(Cq)))
print("off-diag pattern:\n", Cq)

# %%
# show results
import matplotlib.pyplot as plt

plt.plot(jnp.diag(jnp.atleast_2d(Cq)))
plt.title("Diagonal of empirical covariance matrix")
plt.xlabel("coefficient index")
plt.ylabel("Variance")
plt.grid()
plt.show()

# Show Cq matrix
plt.imshow(Cq, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Empirical covariance matrix: natural decay")
plt.xlabel("coefficient index")
plt.ylabel("coefficient index")
plt.show()

# %%
