# %%
import os

print(os.getcwd())

# %%
import matplotlib.pyplot as plt
import numpy as np


def sample_pitchedness(
    aw: float, bw: float, ae: float, be: float, n_samples: int = 100000
):
    nu_w = np.random.gamma(shape=aw, scale=1 / bw, size=n_samples)
    nu_e = np.random.gamma(shape=ae, scale=1 / be, size=n_samples)
    pitchedness = nu_w / (nu_w + nu_e)
    return pitchedness


# Example usage
aw, bw, ae, be = 1.0, 1.0, 1.0, 1.0
samples = sample_pitchedness(aw, bw, ae, be)

plt.hist(samples, bins=30, alpha=0.7)
plt.xlabel("Pitchedness")
plt.ylabel("Frequency")
plt.title("Histogram of Pitchedness Samples")
plt.show()

# Produces a uniform distribution of pitchedness values between 0 and 1!

# %%
# Try and skew towards high pitchedness


def sample_pitchedness2(mean_pitchedness):
    aw = mean_pitchedness / (1 - mean_pitchedness)
    bw, ae, be = 1.0, 1.0, 1.0

    return sample_pitchedness(aw, bw, ae, be)


samples = sample_pitchedness2(0.05)

plt.hist(samples, bins=30, alpha=0.7)
plt.xlabel("Pitchedness")
plt.ylabel("Frequency")
plt.title("Histogram of Pitchedness Samples")
plt.xlim(0, 1)
plt.show()

print("Mean pitchedness:", np.mean(samples))

samples = sample_pitchedness2(0.95)

plt.hist(samples, bins=30, alpha=0.7)
plt.xlabel("Pitchedness")
plt.ylabel("Frequency")
plt.title("Histogram of Pitchedness Samples")
plt.xlim(0, 1)
plt.show()

print("Mean pitchedness:", np.mean(samples))

# %%
from utils import plotting

plotting.iplot(
    samples,
    histogram=True,
    binwidth=0.01,
    _with="boxes fill pattern 1 border lt -1",
    xrange=(0, 1),
    title="TEST100 Distribution of pitchedness with E(pitchedness) = 0.95",
    export="test",
)
