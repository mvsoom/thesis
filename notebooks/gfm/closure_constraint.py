# %%
# Sample a piecewise polynomial from the prior with closure constraint

# NOTE: degree of polarity isn't influenced by closure constraint
# but as H becomes large and closure = True, it all looks like LF basically, and this isnt the case for closure = False
import contextlib
import os

import gnuplotlib as gp
import numpy as np
from matplotlib import pyplot as plt

from gfm.poly import sample_poly

N = 256  # number of samples per waveform
tc = 1  # open phase from [0, 7] msec
t = np.linspace(0, tc, N)
dt = t[1] - t[0]

sigma_a = 1.0


def sample_u(closure, H, d):
    b = np.random.uniform(0.0, tc, size=H)
    du = sample_poly(d, H, t, tc, b=b, closure=closure)
    u = np.cumsum(du) * dt
    u /= np.max(np.abs(u))
    return u


os.makedirs("fig/closure", exist_ok=True)

# %%
plot_args = dict(
    # terminal="svg size 200,200", # => generates nasty bug in typst erroring with "plugin errored with: Capabilities insufficient" => because gnuplotlib outputs `set output "/dev/fd/DUMPONLY"` and Typst doesnt allow file creation
    unset=["border", "xtics", "ytics", "xlabel", "ylabel", "key"],
    cmds=[
        "set xrange [-0.25:2.5]",
        "set yrange [-1.25:3.5]",
    ],
)

np.random.seed(0)

for closure in [True, False]:
    for H in [10, 100, 1000]:
        for d in [0, 1, 2, 3]:
            curves = []

            for i in range(4):
                u = sample_u(closure=closure, H=H, d=d)

                dx = 1.25 * (i // 2)
                dy = 2.25 * (i % 2)

                curves.append(
                    (
                        t + dx,
                        u + dy,
                        {
                            "with": f'filledcurves y1={dy} lc rgb "#888888" fs solid 0.4'
                        },
                    )
                )
                curves.append((t + dx, u + dy, {"with": "lines lc black"}))

                plt.plot(t + dx, u + dy)

            id = f"closure={int(closure)}_H={H}_d={d}"
            plt.title(id)
            plt.show()

            filename = f"fig/closure/{id}.gp"

            with open(filename, "w") as f, contextlib.redirect_stdout(f):
                gp.plot(
                    *curves, ascii=True, dump=True, notest=True, **plot_args
                )

# %%
