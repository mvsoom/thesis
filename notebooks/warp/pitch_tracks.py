# %%
import jax
import jax.numpy as jnp
import plotly.express as px

from prism.svi import svi_basis
from utils import load_egg

payload = load_egg("svi/aplawd/runs/M=16_iter=1_kernelname=rbf.ipynb")

qsvi = payload["qsvi"]
whiten = payload["whiten"]
unwhiten = payload["unwhiten"]

master_key = jax.random.PRNGKey(0)

# %%
# Draw prior samples from learned model


def psi(t):
    return svi_basis(qsvi, t)


tau_test, dtau = jnp.linspace(0, 30 * 2, 1000, retstep=True)
Psi_test = jax.vmap(psi)(tau_test)

master_key, subkey = jax.random.split(master_key)

eps = jax.random.normal(subkey, shape=(Psi_test.shape[1], 5))
y = Psi_test @ eps

g = unwhiten(y)
T = 10.0**g
f = 1000.0 / T

px.line(f).update_traces(x=tau_test).update_layout(
    xaxis_title="tau (cycles)",
    yaxis_title="instantaneous fundamental frequency (Hz)",
    title="Generative pitch track model",
).show()

# %%
# Show phase tracks t(tau)

master_key, subkey = jax.random.split(master_key)

t0 = jax.random.normal(subkey, shape=(5)) * 3

t = t0 + jnp.cumsum(T, axis=0) * dtau

px.line(t).update_traces(x=tau_test).update_layout(
    xaxis_title="normalized time tau (cycles)",
    yaxis_title="phase t (ms)",
    title="Generative instanteneous phase model",
).show()
