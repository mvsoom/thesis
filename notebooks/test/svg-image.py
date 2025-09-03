# file: make_plot.py
# %%
import matplotlib as mpl
import numpy as np

mpl.rcParams.update(
    {
        "text.usetex": False,  # browser-only demo: no TeX dependency
        "svg.fonttype": "none",  # keep text as text
    }
)

import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 512)
y = np.sin(x)

plt.figure(figsize=(4.5, 3))
plt.plot(x, y)

plt.xlabel(r"$x$")
plt.ylabel(r"$\sin(x)$")
plt.title(r"$\int_0^{2\pi}\sin(x)\,dx=0$")
plt.tight_layout()
plt.savefig("plot.svg", transparent=True, bbox_inches="tight")
