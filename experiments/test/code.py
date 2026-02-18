# %% tags=["parameters", "export"]
seed = 0

i = 0

# %%
print(seed)
print(i)

# %%
# Can we get the path to this notebook?
from pathlib import Path

cwd = Path().resolve()

print(cwd)


# %%
import jax.numpy as jnp

# %% tags=["export"]
a = jnp.array([10, 100, 1000, 10000])
scalar = jnp.array(-123.456)
frames = [  # list of dicts (per-frame metrics)
    {"t": [7, 8], "loss": 0.92},
    {"t": [1, 2], "loss": 0.54},
    {"t": [4, 3], "loss": 0.46},
]

dd = [
    {"a": {"b": [-5, -6], "c": 2}},
    {"b": {"d": 3, "e": 4}},
]
