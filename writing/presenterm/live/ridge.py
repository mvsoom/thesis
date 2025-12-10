#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import numpy as np


def cls():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------------- gnuplot ----------------

gp = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE, text=True)


def gp_cmd(s):
    gp.stdin.write(s + "\n")
    gp.stdin.flush()


# ---------------- raw keys ----------------


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ---------------- problem ----------------

# Two data points
x_data = np.array([0.25, 0.75])
y_data = np.sin(2 * np.pi * x_data)


# Three basis functions: 1, x, x^2
def phi(x):
    return np.stack([np.ones_like(x), x, x**2], axis=-1)


Phi = phi(x_data)  # shape (2,3)

# Minimum-norm particular solution
a0 = Phi.T @ np.linalg.solve(Phi @ Phi.T, y_data)

# Nullspace direction
_, _, Vt = np.linalg.svd(Phi, full_matrices=True)
v_null = Vt[-1]
v_null /= np.linalg.norm(v_null)

# Plot grid
x_grid = np.linspace(0, 1, 400)
Phi_grid = phi(x_grid)

# ---------------- parameters ----------------

t_null = 0.0
log10_lambda = -2.0  # ridge scale
log10_aniso = 0.0  # anisotropy on a_3 (quadratic term)

# ---------------- helpers ----------------


def prior_cov():
    # Anisotropy on a_3
    r = 10.0**log10_aniso
    return np.array([1.0, 1.0, r])


def prior_precision():
    return 1.0 / prior_cov()


def a_from_t(t):
    return a0 + t * v_null


def curve_from_t(t):
    return Phi_grid @ a_from_t(t)


def ridge_cost_from_t(t):
    a = a_from_t(t)
    lam = 10.0**log10_lambda
    P = prior_precision()
    return lam * np.sum(P * a * a)


def t_star():
    # argmin_t (a0 + t v)^T P (a0 + t v)
    P = np.diag(prior_precision())
    num = v_null @ (P @ a0)
    den = v_null @ (P @ v_null) + 1e-12
    return -num / den


# ---------------- gnuplot setup ----------------

gp_cmd('set term kitty size 1600,900 background rgb "black" enhanced')
gp_cmd("unset key")
gp_cmd("set border lc rgb '#4c566a'")
gp_cmd("set tics textcolor rgb '#4c566a'")

# ---------------- redraw ----------------


def redraw():
    f = curve_from_t(t_null)
    t_opt = t_star()
    cov = prior_cov()

    gp_cmd("set multiplot layout 1,2")

    # ===== LEFT: curves =====
    gp_cmd("set lmargin at screen 0.07")
    gp_cmd("set rmargin at screen 0.53")
    gp_cmd("set tmargin at screen 0.90")
    gp_cmd("set bmargin at screen 0.10")
    gp_cmd("set xrange [0:1]")
    gp_cmd("set yrange [-2.5:2.5]")
    gp_cmd("set xlabel 'x' tc rgb '#d8dee9'")
    gp_cmd("set ylabel 'f(x)' tc rgb '#d8dee9'")
    gp_cmd(
        "set title 'All curves in the nullspace fit the data equally well' tc rgb 'white'"
    )

    gp_cmd(
        "set label 1 "
        "'min_{a} ||Φ a - y||^{2} + λ a^{T} Σ^{-1} a' "
        "at graph 0.65,0.93 tc rgb 'white' font ',12'"
    )
    gp_cmd(
        "set label 2 "
        "sprintf('λ = 10^{%.0f},  Σ = diag(1, 1, %.2f)', %.0f, %.2f) "
        "at graph 0.65,0.85 tc rgb 'white' font ',12'"
        % (log10_lambda, cov[2], log10_lambda, cov[2])
    )

    gp_cmd(
        "set label 3 "
        "'a/z: nullspace   j/k: λ   l/m: anisotropy on a_3   q: quit' "
        "at graph 0.02,0.05 tc rgb '#d8dee9' font ',9'"
    )

    gp_cmd(
        "plot '-' w l lw 3 lc rgb '#88c0d0', '-' w p pt 7 ps 2 lc rgb '#bf616a'"
    )

    for x, y in zip(x_grid, f):
        gp_cmd(f"{x} {y}")
    gp_cmd("e")

    for x, y in zip(x_data, y_data):
        gp_cmd(f"{x} {y}")
    gp_cmd("e")

    # ===== RIGHT: cost along nullspace =====
    gp_cmd("unset label 1")
    gp_cmd("unset label 2")
    gp_cmd("unset label 3")

    gp_cmd("set lmargin at screen 0.57")
    gp_cmd("set rmargin at screen 0.97")
    gp_cmd("set tmargin at screen 0.90")
    gp_cmd("set bmargin at screen 0.10")
    gp_cmd("set xrange [-3:3]")
    gp_cmd("set yrange [0:2.0]")
    gp_cmd("set xlabel 'nullspace coordinate t' tc rgb '#d8dee9'")
    gp_cmd("set ylabel 'prior cost' tc rgb '#d8dee9'")
    gp_cmd("set title 'Ridge cost along nullspace direction' tc rgb 'white'")

    t_grid = np.linspace(-3, 3, 400)
    J = np.array([ridge_cost_from_t(t) for t in t_grid])

    gp_cmd(
        "plot "
        "'-' w l lw 2 lc rgb '#ebcb8b', "
        "'-' w p pt 7 ps 2 lc rgb '#bf616a', "
        "'-' w p pt 7 ps 1.5 lc rgb '#a3be8c'"
    )

    for t, j in zip(t_grid, J):
        gp_cmd(f"{t} {j}")
    gp_cmd("e")

    gp_cmd(f"{t_null} {ridge_cost_from_t(t_null)}")
    gp_cmd("e")

    gp_cmd(f"{t_opt} {ridge_cost_from_t(t_opt)}")
    gp_cmd("e")

    gp_cmd("unset multiplot")


# ---------------- loop ----------------

cls()
redraw()

try:
    with Keys():
        while True:
            ch = sys.stdin.read(1)
            if ch == "q":
                break
            elif ch == "a":
                t_null -= 0.1
            elif ch == "z":
                t_null += 0.1
            elif ch == "j":
                log10_lambda -= 1.0
            elif ch == "k":
                log10_lambda += 1.0
            elif ch == "l":
                log10_aniso -= 0.2
            elif ch == "m":
                log10_aniso += 0.2
            redraw()
finally:
    gp.stdin.close()
    cls()
