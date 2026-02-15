# %% [markdown]
# # VI
#
# Speed:
#
# - GPU: linear in $r$, insensitive to `batch_size`, 2x - 10x speedup for x32 compared to x64
#
# Possible optimizations:
#

# %%
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly import colors as pc

from iklp.hyperparams import Hyperparams
from iklp.periodic import periodic_kernel_phi, periodic_mock_data
from iklp.state import compute_expectations, init_state, init_test_stable_state
from iklp.vi import (
    compute_elbo_bound,
    update_delta_a,
    vi_step,
)

update_delta_a = jax.jit(update_delta_a)
vi_step = jax.jit(vi_step)
compute_elbo_bound = jax.jit(compute_elbo_bound)
compute_expectations = jax.jit(compute_expectations)


# %%
with jax.default_device(jax.devices("cpu")[0]):
    f0, Phi = periodic_kernel_phi(I=400, M=2048, fs=16000)

# %%
h = Hyperparams(Phi)
key = jax.random.PRNGKey(3334)

if False:
    state = init_test_stable_state(key, h, remove_dc=True)
else:
    k1, k2 = jax.random.split(key, 2)
    f0_star, x = periodic_mock_data(k1, f0, Phi)
    state = init_state(k2, x, h)

# %%
# Warmup
update_delta_a(state)
vi_step(state)
compute_elbo_bound(state)
compute_expectations(state)

# %%
fig = go.Figure()

# Updating q(a) = delta(a* - a) as the very first update
# is known to yield better convergence
# as it is initalized to zeroes
state = update_delta_a(state)

score = -jnp.inf
criterion = 0.001
n_iter = 10

for i in range(n_iter):
    E = compute_expectations(state)
    print(f"argmax_i(theta) = {np.argmax(E.theta)}")

    state = vi_step(state)

    color = pc.sample_colorscale("RdBu", i / max(n_iter - 1, 1))[0]
    fig.add_trace(
        go.Scatter(
            x=np.asarray(f0),
            y=np.asarray(E.theta),
            mode="lines",
            line=dict(color=color),
            opacity=0.8,
            name=f"iter {i}",
        )
    )

    lastscore = score
    score = compute_elbo_bound(state)

    if i == 0:
        improvement = 1.0
    else:
        improvement = (score - lastscore) / jnp.abs(lastscore)

    print(
        "iteration {}: bound = {:.2f} ({:+.5f} improvement)".format(
            i, score, improvement
        )
    )
    if improvement < 0.0:
        print("Diverged")
        break
    if improvement < criterion:
        print("Converged")
        break
    if jnp.isnan(improvement) and i > 0:
        print("NaN")
        break

fig.update_xaxes(title="F0")
fig.update_yaxes(title="E[theta]", type="log")
fig.show()


# %%
idx = int(np.argmax(np.asarray(E.theta)))
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.asarray(E.theta), mode="lines", name="E[theta]"))
# vline at argmax
fig.add_vline(x=idx, line_color="red", line_dash="dash")
fig.update_xaxes(title="index")
fig.update_yaxes(title="E[theta]")
fig.show()


# %%
x = np.asarray(state.data.x)
fs = 16000
N = len(x)
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(N, d=1 / fs)
power = np.abs(X) ** 2 / N
power_db = 10 * np.log10(power + 1e-6)

fig = go.Figure()
fig.add_trace(go.Scatter(x=freqs, y=power_db, mode="lines", name="power"))
fig.add_vline(
    x=float(np.asarray(f0)[int(np.argmax(np.asarray(E.theta)))]),
    line_color="red",
    line_dash="dash",
    opacity=0.5,
)
fig.update_xaxes(title="Frequency (Hz)", range=[100, 400])
fig.update_yaxes(title="Power (dB)")
fig.update_layout(title="Power Spectrum (dB)")
fig.show()
