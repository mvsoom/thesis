#!/usr/bin/env python3

import subprocess
import sys
import termios
import time
import tty

import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess

from gp.blr import blr_from_mercer
from gp.periodic import PeriodicSE
from pack.refine import learn_surrogate_blr  # level-2 imitation prior
from utils import lfmodel
from utils.jax import vk

# ============================================================
# RNG
# ============================================================

RNG = np.random.default_rng(1234)

# ============================================================
# LF MODEL
# ============================================================

lf_dgf = jax.jit(lfmodel.dgf)

T = 6.0
N = 800
NUM_PERIODS = 3

t, dt = np.linspace(0.0, T * NUM_PERIODS, N, retstep=True)
t_jax = jnp.array(t)


def generate_examplar(Rd):
    p = lfmodel.convert_lf_params({"T0": T, "Rd": Rd}, "Rd -> T")
    du = np.zeros_like(t)
    for i in range(NUM_PERIODS):
        du = du + np.array(lf_dgf(t - i * T, p))
    u = np.cumsum(du) * dt
    power = (du**2).sum() * dt / T
    u /= np.sqrt(power)
    du /= np.sqrt(power)
    return {"t": t_jax, "du": jnp.array(du), "u": jnp.array(u)}


def sample_example_lf():
    Rd = RNG.uniform(0.3, 2.7)
    ex = generate_examplar(Rd)
    # for the simulator column we only need u
    return np.array(ex["u"]), ex


# ============================================================
# LEVEL PARAMETERS
# ============================================================

# Level 0: deliberately wrong
ELL0 = 1.0
AMP0 = 0.2
PERIOD0 = 2.7

# Level 1: correct amp + period, learn ell
AMP1 = 1.0
PERIOD1 = T

ell_level1 = ELL0
optimizing_level1 = False

# Level 2: imitation prior (BLR built from exemplars)
imitation_blr = None
optimizing_level2 = False

# how many exemplars for level 1 & 2
NUM_EXEMPLARS_L1 = 5
NUM_EXEMPLARS_L2 = 20

# ============================================================
# GP BUILDERS FOR LEVEL 1
# ============================================================


def build_gp_level1(t_jax, ell_value):
    kernel = AMP1 * PeriodicSE(
        ell=jnp.array(ell_value),
        period=jnp.array(PERIOD1),
        J=20,
    )
    return GaussianProcess(kernel, t_jax, diag=1e-2)


@jax.jit
def loglik_single(ell, y):
    gp = build_gp_level1(t_jax, ell)
    return gp.log_probability(y)


@jax.jit
def total_log_likelihood(ell, Y):
    return jnp.sum(jax.vmap(lambda y: loglik_single(ell, y))(Y))


# ============================================================
# SAMPLING BY COLUMN
# ============================================================


def sample_example(col):
    global imitation_blr

    if col == 0:
        # level 0 prior: deliberately wrong amp + period + ell
        kernel = AMP0 * PeriodicSE(
            ell=jnp.array(ELL0),
            period=jnp.array(PERIOD0),
            J=20,
        )
        gp0 = blr_from_mercer(kernel, t)
        return t, np.array(gp0.sample(vk()))

    if col == 1:
        # level 1 prior: correct amp + period, learned ell
        kernel = AMP1 * PeriodicSE(
            ell=jnp.array(ell_level1),
            period=jnp.array(PERIOD1),
            J=20,
        )
        gp1 = blr_from_mercer(kernel, t)
        return t, np.array(gp1.sample(vk()))

    if col == 2:
        # level 2 imitation prior: BLR trained from exemplars
        if imitation_blr is None:
            # nothing learned yet: show empty line
            return t, np.zeros_like(t)
        f = np.array(imitation_blr.sample(vk()))
        return t, f

    if col == 3:
        # simulator column: ground-truth LF model samples
        u, _ = sample_example_lf()
        return t, u

    return t, np.zeros_like(t)


# ============================================================
# LEVEL 1 FITTING (1D SEARCH, UNCHANGED)
# ============================================================


def fit_level_1():
    global ell_level1, optimizing_level1, samples

    optimizing_level1 = True

    # exemplar batch: use LF simulator u as "data"
    Y = jnp.stack(
        [jnp.array(sample_example_lf()[0]) for _ in range(NUM_EXEMPLARS_L1)],
        axis=0,
    )

    # initial bracket
    lo, hi = 1.0, 10.0

    for step in range(12):
        mids = np.array(
            [
                lo,
                0.5 * (lo + hi),
                hi,
            ]
        )

        vals = np.array(
            [float(total_log_likelihood(float(m), Y)) for m in mids]
        )

        best = mids[np.argmax(vals)]

        # shrink bracket
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
        time.sleep(0.05)

    optimizing_level1 = False


# ============================================================
# LEVEL 2 FITTING (IMITATION PRIOR VIA BLR)
# ============================================================


def fit_level_2():
    """
    Build a surrogate BLR prior (imitation prior) from many LF exemplars,
    using the Level-1 kernel hyperparameters (AMP1, PERIOD1, ell_level1).
    """
    global imitation_blr, optimizing_level2, samples

    optimizing_level2 = True
    redraw()

    # build Mercer kernel matching Level-1 GP
    kernel = AMP1 * PeriodicSE(
        ell=jnp.array(ell_level1),
        period=jnp.array(PERIOD1),
        J=20,
    )

    # collect exemplars with (t, du)
    exemplars = []
    for _ in range(NUM_EXEMPLARS_L2):
        ex = generate_examplar(RNG.uniform(0.3, 2.7))
        exemplars.append({"t": ex["t"], "du": ex["u"]})

    # learn surrogate BLR prior from exemplars
    imitation_blr = learn_surrogate_blr(
        kernel,
        exemplars,
        evaluation_times=t_jax,
        noise_variance=1e-4,
        envelope_jitter=1e-9,
        enforce_zero_mean=False,
    )

    # show one fresh sample from imitation prior
    samples[2] = [sample_example(2)]
    optimizing_level2 = False
    redraw()


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
    if c == "\x1b":
        c2 = sys.stdin.read(2)
        if c2 == "[D":
            return "LEFT"
        if c2 == "[C":
            return "RIGHT"
        return None
    return c


# ============================================================
# GNUPLOT
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
active_col = 3

samples = [[] for _ in range(n_cols)]
samples[3] = [sample_example(3)]

titles = [
    "Level 0 learning (none)",
    "Level 1 learning",
    "Level 2 learning",
    "Simulator",
]


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
        if col == 1 and optimizing_level1:
            title += " (optimizing...)"
        if col == 2 and optimizing_level2:
            title += " (optimizing...)"
        if col == active_col:
            title += "  <"

        gp_cmd(f"set title '{title}' tc rgb 'white'")

        if samples[col]:
            gp_cmd("plot '-' w l lw 2 lc rgb '#88c0d0'")
            x, y = samples[col][0]
            for xi, yi in zip(x, y):
                gp_cmd(f"{xi} {yi}")
            gp_cmd("e")
        else:
            gp_cmd("plot 0 w p pt 7 ps 0")

    gp_cmd("unset multiplot")
    gp_cmd("pause 0")


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
                samples[active_col] = [sample_example(active_col)]
            elif k == "f":
                if active_col == 1 and not optimizing_level1:
                    fit_level_1()
                elif active_col == 2 and not optimizing_level2:
                    fit_level_2()

            redraw()
finally:
    try:
        gp.stdin.close()
    except Exception:
        pass
    cls()
