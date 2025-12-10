#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess

from gp.blr import blr_from_mercer
from gp.periodic import PeriodicSE
from pack.refine import learn_surrogate_blr
from utils import lfmodel
from utils.jax import vk

RNG = np.random.default_rng(1234)

# ============================================================
# LF EXEMPLARS
# ============================================================

lf_dgf = jax.jit(lfmodel.dgf)

T = 6.0  # single-period duration for LF model
N = 800
NUM_PERIODS = 3  # how many periods to tile
t, dt = np.linspace(0.0, T * NUM_PERIODS, N, retstep=True)


def generate_examplar(Rd):
    p = lfmodel.convert_lf_params({"T0": T, "Rd": Rd}, "Rd -> T")

    du = np.zeros_like(t)
    for i in range(NUM_PERIODS):
        du = du + np.array(lf_dgf(t - i * T, p))
    u = np.cumsum(du) * dt

    # gauge + normalization (as in your working code)
    te = t[np.argmin(du)]  # instant of peak excitation
    ct = t - te  # centred time axis
    to = 0.0 - te

    power = (du**2).sum() * dt / T
    du /= np.sqrt(power)
    u /= np.sqrt(power)

    return {
        "Rd": Rd,
        "ct": ct,
        "to": to,
        "du": du,
        "u": u,
    }


def sample_example_lf():
    Rd = RNG.uniform(0.3, 2.7)
    lf = generate_examplar(Rd)
    return lf


# tiny helper for refine.py so _as_arrays sees .t and .du
class Exemplar:
    def __init__(self, t, y):
        self.t = t
        self.du = y


# ============================================================
# KERNEL / PRIOR SETUP
# ============================================================

# Level 0 prior: deliberately wrong-ish
ELL_L0 = 1.0  # too small
SIGMA_L0 = 0.5  # a bit small
PERIOD_L0 = 4.5  # deliberately off

# Level 1: we will learn ell; amp & period are "true"
SIGMA_L1 = 1.0
PERIOD_TRUE = T  # correct period

# Learned ell for level 1 (starts at same as L0)
ell_level1 = ELL_L0

# Level 2: imitation prior (BLR) learned from exemplars
blr_level2 = None

# ============================================================
# USER-FACING SAMPLING HOOK
# ============================================================


def sample_example(col):
    global ell_level1, blr_level2

    if col == 0:
        # Level 0: fixed wrong-ish kernel
        kernel = SIGMA_L0 * PeriodicSE(
            ell=jnp.array(ELL_L0),
            period=jnp.array(PERIOD_L0),
            J=20,
        )
        gp = blr_from_mercer(kernel, t)
        f = gp.sample(vk())
        return t, np.asarray(f)

    if col == 1:
        # Level 1: learned ell, correct amp & period
        kernel = SIGMA_L1 * PeriodicSE(
            ell=jnp.array(ell_level1),
            period=jnp.array(PERIOD_TRUE),
            J=20,
        )
        gp = blr_from_mercer(kernel, t)
        f = gp.sample(vk())
        return t, np.asarray(f)

    if col == 2:
        # Level 2: imitation prior
        if blr_level2 is None:
            # fall back to Level 1 behaviour before fitting
            kernel = SIGMA_L1 * PeriodicSE(
                ell=jnp.array(ell_level1),
                period=jnp.array(PERIOD_TRUE),
                J=20,
            )
            gp = blr_from_mercer(kernel, t)
            f = gp.sample(vk())
            return t, np.asarray(f)
        else:
            f = blr_level2.sample(vk())
            return t, np.asarray(f)

    if col == 3:
        # Simulator
        lf = sample_example_lf()
        f = lf["u"]
        return t, np.asarray(f)

    # should never happen
    return t, np.zeros_like(t)


# ============================================================
# LEVEL-1 FIT: LEARN ELL BY 1D SEARCH
# ============================================================

def build_gp_for_ell(t_data, ell):
    kernel = SIGMA_L1 * PeriodicSE(
        ell=jnp.array(ell),
        period=jnp.array(PERIOD_TRUE),
        J=20,
    )
    return GaussianProcess(kernel, t_data, diag=1e-6)


def total_log_likelihood_ell(ell, exemplars):
    # exemplars are (t, f) tuples from the simulator
    ell = float(ell)
    ll = 0.0
    for t_ex, f_ex in exemplars:
        t_ex_j = jnp.asarray(t_ex)
        f_ex_j = jnp.asarray(f_ex)
        gp = build_gp_for_ell(t_ex_j, ell)
        ll = ll + gp.log_probability(f_ex_j)
    return float(ll)


def fit_level_1():
    global ell_level1

    # fixed small exemplar pool
    exemplars = [sample_example(3) for _ in range(1)]

    lo, hi = 1.0, 10.0

    for step in range(12):
        mids = np.array(
            [
                lo,
                0.5 * (lo + hi),
                hi,
            ]
        )

        vals = []
        for m in mids:
            kernel = SIGMA_L1 * PeriodicSE(
                ell=jnp.array(float(m)),
                period=jnp.array(PERIOD_TRUE),
                J=20,
            )
            ll = 0.0
            for t_ex, f_ex in exemplars:
                gp = GaussianProcess(kernel, jnp.array(t_ex), diag=1e-2)
                ll += float(gp.log_probability(jnp.array(f_ex)))
            vals.append(ll)

        vals = np.array(vals)
        best = mids[np.argmax(vals)]

        if best == lo:
            hi = 0.5 * (lo + hi)
        elif best == hi:
            lo = 0.5 * (lo + hi)
        else:
            lo = lo + 0.25 * (hi - lo)
            hi = hi - 0.25 * (hi - lo)

        ell_level1 = float(best)

        samples[1] = [sample_example(1)]
        redraw()


# ============================================================
# LEVEL-2 FIT: IMITATION PRIOR VIA BLR ENVELOPE
# ============================================================


def fit_level_2():
    """Learn BLR surrogate prior matching posteriors over many exemplars."""
    global blr_level2, ell_level1

    # we assume fit_level_1 has run, so ell_level1 is sensible
    kernel = SIGMA_L1 * PeriodicSE(
        ell=jnp.array(ell_level1),
        period=jnp.array(PERIOD_TRUE),
        J=20,
    )

    # more exemplars here; fitting is just linear algebra
    n_exemplars = 40
    exemplars = []
    for _ in range(n_exemplars):
        lf = sample_example_lf()
        # use u as "du" just for the talk – visually nicer
        exemplars.append(Exemplar(t, lf["u"]))

    blr_level2 = learn_surrogate_blr(
        kernel,
        exemplars,
        evaluation_times=t,
        noise_variance=1e-4,
        envelope_jitter=1e-8,
        enforce_zero_mean=False,
    )


def fit_level(col):
    if col == 1:
        fit_level_1()
    elif col == 2:
        fit_level_2()


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
n_samples = 1  # max one per column in this demo

# per-column stored samples: list of (x, y), but at most one each
samples = [[] for _ in range(n_cols)]

# initialise simulator column
for _ in range(n_samples):
    samples[3].append(sample_example(3))

titles = [
    "Level 0: prior (wrong kernel)",
    "Level 1: GP learned prior",
    "Level 2: imitation prior (BLR)",
    "Simulator",
]

status_msgs = ["", "", "", ""]  # extra status text per column


# ============================================================
# DRAW
# ============================================================


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
        if status_msgs[col]:
            title += f"  [{status_msgs[col]}]"
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
                gp_cmd(f"{float(xi)} {float(yi)}")
            gp_cmd("e")

    gp_cmd("unset multiplot")


# ============================================================
# ACTIONS
# ============================================================


def resample_column(col):
    samples[col] = [sample_example(col) for _ in range(n_samples)]


def fit_column(col):
    if col not in (1, 2):
        return

    # show that we’re working
    status_msgs[col] = "optimizing..."
    redraw()

    fit_level(col)

    status_msgs[col] = "trained"
    # after training, don’t change existing samples, but future space-presses
    # will use updated priors
    redraw()


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
