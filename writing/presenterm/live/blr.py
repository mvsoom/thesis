#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import numpy as np

# ---------- gnuplot ----------

gp = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE, text=True)


def gp_cmd(s):
    gp.stdin.write(s + "\n")
    gp.stdin.flush()


# ---------- key handling ----------


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ---------- BLR / GP setup ----------

np.random.seed(0)

N_GRID = 400
M_BASIS = 20
SIGMA_NOISE = 0.1

x_grid = np.linspace(0.0, 1.0, N_GRID)
n_idx = np.arange(M_BASIS)

# cosine basis, independent of kernel hyperparams
Phi_grid = np.sqrt(2.0) * np.cos(2.0 * np.pi * x_grid[:, None] * n_idx[None, :])

# ground truth GP uses fixed lengthscale; prior used for BLR is adjustable
ELL_TRUE = 0.25
ell_prior = 0.25


def spectral_variances(ell):
    # periodic SE spectrum on circle (up to a constant)
    w = 2.0 * np.pi * n_idx
    s = np.exp(-0.5 * (ell * w) ** 2)
    # avoid exactly zero variances
    return np.maximum(s, 1e-6)


# sample ground truth weights once from GP with ELL_TRUE
s_true = spectral_variances(ELL_TRUE)
w_true = np.random.normal(scale=np.sqrt(s_true), size=M_BASIS)
f_true_grid = Phi_grid @ w_true

x_data = []
y_data = []


def f_true(x):
    phi_x = np.sqrt(2.0) * np.cos(2.0 * np.pi * x * n_idx)
    return float(phi_x @ w_true)


def compute_posterior_curve():
    if not x_data:
        # no data yet: zero mean
        return np.zeros_like(x_grid)

    X = np.asarray(x_data)
    Y = np.asarray(y_data)

    Phi = np.sqrt(2.0) * np.cos(2.0 * np.pi * X[:, None] * n_idx[None, :])

    s = spectral_variances(ell_prior)
    S_inv = 1.0 / s

    A = (Phi.T @ Phi) / (SIGMA_NOISE**2)
    A += np.diag(S_inv)

    b = Phi.T @ Y / (SIGMA_NOISE**2)

    w_mean = np.linalg.solve(A, b)
    return Phi_grid @ w_mean


f_post_grid = compute_posterior_curve()

# ---------- gnuplot layout ----------

gp_cmd('set term kitty size 1600,900 background rgb "black"')
gp_cmd("set encoding utf8")
gp_cmd("unset key")
gp_cmd("set border lc rgb 'gray'")
gp_cmd("set tics textcolor rgb 'gray'")
gp_cmd("set xrange [0:1]")
gp_cmd("set yrange [-2:2]")
gp_cmd("set multiplot layout 1,2")
gp_cmd(
    "set label 10 'j/k: lengthscale   space: add point   c: clear   q: quit'"
    " at screen 0.05,0.97 tc rgb 'white'"
)


def redraw():
    # left: data and BLR function
    gp_cmd("set lmargin at screen 0.07")
    gp_cmd("set rmargin at screen 0.52")
    gp_cmd("set tmargin at screen 0.93")
    gp_cmd("set bmargin at screen 0.1")
    gp_cmd("set xrange [0:1]")
    gp_cmd("set yrange [-2:2]")
    gp_cmd(
        "set title 'BLR from GP prior (periodic SE kernel)' tc rgb 'white' font ',12'"
    )
    gp_cmd(
        "set label 1 sprintf('ell_prior = %.3f', %.6f) at graph 0.02,0.9 "
        "tc rgb 'white'" % (ell_prior, ell_prior)
    )

    gp_cmd(
        "plot '-' w p pt 7 ps 1.5 lc rgb 'white', "
        "'-' w l lw 3 lc rgb '#00e5ff', "
        "'-' w l lw 1 lc rgb 'gray'"
    )

    # data points
    for x, y in zip(x_data, y_data):
        gp_cmd(f"{x} {y}")
    gp_cmd("e")

    # posterior mean
    for x, y in zip(x_grid, f_post_grid):
        gp_cmd(f"{x} {y}")
    gp_cmd("e")

    # true function
    for x, y in zip(x_grid, f_true_grid):
        gp_cmd(f"{x} {y}")
    gp_cmd("e")

    # right: spectral weights
    gp_cmd("set lmargin at screen 0.55")
    gp_cmd("set rmargin at screen 0.98")
    gp_cmd("set tmargin at screen 0.93")
    gp_cmd("set bmargin at screen 0.1")
    gp_cmd("set xrange [0:%d]" % (M_BASIS - 1))
    gp_cmd("set yrange [0:1.05]")
    gp_cmd(
        "set title 'Spectral weights (which explanations are cheap)' "
        "tc rgb 'white' font ',12'"
    )
    gp_cmd("unset label 1")

    s = spectral_variances(ell_prior)
    s_norm = s / s[0] if s[0] > 0 else s

    gp_cmd("plot '-' w impulses lw 2 lc rgb 'yellow'")
    for i, val in enumerate(s_norm):
        gp_cmd(f"{i} {val}")
    gp_cmd("e")


redraw()

# ---------- event loop ----------

try:
    with Keys():
        while True:
            ch = sys.stdin.read(1)
            if ch == "q":
                break
            elif ch == " ":
                x_new = np.random.uniform(0.0, 1.0)
                y_new = f_true(x_new) + np.random.normal(scale=SIGMA_NOISE)
                x_data.append(float(x_new))
                y_data.append(float(y_new))
            elif ch == "j":
                ell_prior *= 0.7
            elif ch == "k":
                ell_prior *= 1.3
            elif ch == "c":
                x_data.clear()
                y_data.clear()
            else:
                continue

            f_post_grid = compute_posterior_curve()
            redraw()
finally:
    try:
        gp_cmd("unset multiplot")
        gp.stdin.close()
    except Exception:
        pass
