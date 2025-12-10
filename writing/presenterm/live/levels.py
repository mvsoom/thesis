#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import jax
import jax.numpy as jnp
import numpy as np

from gp.blr import blr_from_mercer
from gp.periodic import PeriodicSE
from utils.jax import vk

RNG = np.random.default_rng(1234)

# ============================================================
# USER-SUPPLIED HOOKS (YOU FILL THESE IN LATER)
# ============================================================
from utils import lfmodel

lf_dgf = jax.jit(lfmodel.dgf)

# %%
# Generate examplar
T = 6.0
N = 800
NUM_PERIODS = 3

t, dt = np.linspace(0.0, T * NUM_PERIODS, N, retstep=True)


def generate_examplar(Rd):
    p = lfmodel.convert_lf_params({"T0": T, "Rd": Rd}, "Rd -> T")

    du = np.zeros_like(t)

    for i in range(NUM_PERIODS):
        du = du + np.array(lf_dgf(t - i * T, p))
    u = np.cumsum(du) * dt

    # Do gauge
    te = t[np.argmin(du)]  # instant of peak excitation
    ct = t - te  # center time axis
    to = 0.0 - te  # time of opening where du is zero

    power = (du**2).sum() * dt / T
    du /= np.sqrt(power)
    u /= np.sqrt(power)

    d = {
        "Rd": Rd,
        "ct": ct,
        "to": to,
        "du": du,
        "u": u,
    }
    return d


def sample_example_lf(params=None):
    Rd = RNG.uniform(0.3, 2.7)
    lf = generate_examplar(Rd)
    return lf


def sample_example(col):
    match col:
        case 0:
            kernel = 0.2 * PeriodicSE(
                ell=jnp.array(1.0), period=jnp.array(4.0), J=20
            )
            gp = blr_from_mercer(kernel, t)
            f = gp.sample(vk())

        case 3:
            Rd = RNG.uniform(0.3, 2.7)
            lf = sample_example_lf()
            f = lf["u"]

    return t, f


# EXAMPLES = [sample_example() for _ in range(10)]


def fit_level(col):
    match col:
        case 1:
            fit_level_1()
        case 2:
            fit_level_2()


from tinygp import GaussianProcess


def build_gp(t, theta):
    kernel = jnp.exp(theta["log_amp"]) * PeriodicSE(
        ell=jnp.exp(theta["log_ell"]), period=jnp.exp(theta["log_ell"]), J=20
    )

    return GaussianProcess(kernel, t, diag=1e-6)


def fit_level_1():
    # TODO
    exemplars = [sample_example(3) for _ in range(10)]

    def total_log_likelihood(theta, exemplars):
        ll = 0.0
        for t, f in exemplars:
            gp = build_gp(t, theta)
            ll += gp.log_probability(f)
        return ll


def fit_level_2():
    pass  # nothing yet


# ============================================================
# TERMINAL UTILITIES
# ============================================================


def cls():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def read_key():
    c = sys.stdin.read(1)
    if c == "\x1b":  # escape
        c2 = sys.stdin.read(2)
        if c2 == "[D":
            return "LEFT"
        if c2 == "[C":
            return "RIGHT"
        return None
    return c


# ============================================================
# GNUPLOT SETUP
# ============================================================

gp = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE, text=True)


def gp_cmd(s):
    gp.stdin.write(s + "\n")
    gp.stdin.flush()


gp_cmd("set term kitty background rgb 'black' size 1800,900")
gp_cmd("unset key")
gp_cmd("set border lc rgb '#4c566a'")
gp_cmd("set tics textcolor rgb '#4c566a'")
gp_cmd("set xlabel 'x' tc rgb '#d8dee9'")
gp_cmd("set ylabel 'f(x)' tc rgb '#d8dee9'")
gp_cmd(f"set xrange [0:{NUM_PERIODS * T}]")
gp_cmd("set yrange [-2.5:2.5]")


# ============================================================
# STATE
# ============================================================

n_cols = 4
active_col = 3  # start on simulator
n_samples = 1

# per-column stored samples: list of lists [(x,y), ...]
samples = [[] for _ in range(n_cols)]

# initialise simulator column with samples
for _ in range(n_samples):
    samples[3].append(sample_example(3))


KERNEL = "periodickernel"  # spack:1 or whitenoise
KERNEL_THETA = {
    "sigma_a": 1.0,
    "ell": 3,
    "T": T,
    "sigma_b": 1.0,
    "sigma_c": 1.0,
}

# ============================================================
# DRAW
# ============================================================

titles = [
    "Level 0: prior",
    "Level 1: GP learned prior",
    "Level 2: imitation prior",
    "Simulator",
]


def redraw():
    gp_cmd("set multiplot layout 1,4")

    for col in range(n_cols):
        left = 0.02 + col * 0.24
        right = left + 0.22

        gp_cmd(f"set lmargin at screen {left}")
        gp_cmd(f"set rmargin at screen {right}")
        gp_cmd("set tmargin at screen 0.88")
        gp_cmd("set bmargin at screen 0.12")

        title = titles[col]
        if col == active_col:
            title += "  <"

        gp_cmd(f"set title '{title}' tc rgb 'white'")

        plot_terms = []
        for _ in samples[col]:
            plot_terms.append("'-' w l lw 1 lc rgb '#88c0d0'")

        if not plot_terms:
            plot_terms = ["0 w p pt 7 ps 0"]

        gp_cmd("plot " + ",".join(plot_terms))

        for x, y in samples[col]:
            for xi, yi in zip(x, y):
                gp_cmd(f"{xi} {yi}")
            gp_cmd("e")

    gp_cmd("unset multiplot")


# ============================================================
# ACTIONS
# ============================================================


def resample_column(col):
    samples[col] = [sample_example(col) for _ in range(n_samples)]


def fit_column(col):
    fit_level(col)
    # nothing visual yet, but keep hook


# ============================================================
# MAIN LOOP
# ============================================================

cls()
redraw()

try:
    with Keys():
        while True:
            k = read_key()
            if k is None:
                continue

            if k == "q":
                break

            elif k == "LEFT":
                active_col = (active_col - 1) % n_cols

            elif k == "RIGHT":
                active_col = (active_col + 1) % n_cols

            elif k == " ":
                resample_column(active_col)

            elif k == "f":
                fit_column(active_col)

            redraw()

finally:
    try:
        gp.stdin.close()
    except Exception:
        pass
    cls()
