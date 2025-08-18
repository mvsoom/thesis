# %%
import jax

from iklp.gig import compute_gig_expectations, gig_dkl_from_gamma, sample_gig

key = jax.random.PRNGKey(486)

rho = 1 / 0.45
gamma = 5.7
b = 45.1

for tau in [0.0, 0.9]:
    Ex_samples = sample_gig(key, gamma, rho, tau, size=1000).mean()
    Exinv_samples = (1 / sample_gig(key, gamma, rho, tau, size=1000)).mean()
    print("E[x] samples:", Ex_samples)
    print("1/E[1/x] samples:", 1 / Exinv_samples)

    Ex, Exinv = compute_gig_expectations(gamma, rho, tau)
    print("E[x]:", Ex)
    print("1/E[1/x]:", 1 / Exinv)

    # D_KL example (may require optional _gap extension)
    try:
        D_kl = gig_dkl_from_gamma(Ex, Exinv, rho, tau, gamma, b)
        print("D_kl:", D_kl)
    except Exception as e:
        print("Error computing D_kl:", e)

# Vmap test
rho = jax.numpy.array([1 / 0.45, 1 / 0.55])
tau = jax.numpy.array([0.0, 0.9])
Ex, Exinv = jax.vmap(compute_gig_expectations, in_axes=(None, 0, 0))(
    gamma, rho, tau
)
print("E[x]:", Ex)
print("1/E[1/x]:", 1 / Exinv)