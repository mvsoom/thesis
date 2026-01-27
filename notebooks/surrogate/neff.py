# %%
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", False)
# jax.config.update("jax_platform_name", "cpu") # GPU = 20x speedup


from surrogate import source

lf_sample = source.get_lf_samples()

neffs = np.array([len(s["t"]) for s in lf_sample])


# %%
# dump neffs to csv that R can read with index column sample_idx and neff column
import pandas as pd

neffs = neffs[:100]
df = pd.DataFrame({"sample_idx": np.arange(len(neffs)), "neff": neffs})
df.to_csv("../../experiments/lf/neffs.csv", index=False)
