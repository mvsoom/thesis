#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import numpy as np


def cls():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ------------------------------------------------------------
# gnuplot
# ------------------------------------------------------------
gp = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE, text=True)


def gp_cmd(s):
    gp.stdin.write(s + "\n")
    gp.stdin.flush()


# ------------------------------------------------------------
# raw key handling
# ------------------------------------------------------------
class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ------------------------------------------------------------
# squared exponential kernel
# ------------------------------------------------------------
def k_sqexp(x, y, ell):
    x = x[:, None]
    y = y[None, :]
    return np.exp(-0.5 * (x - y) ** 2 / ell**2)


# ------------------------------------------------------------
# gnuplot setup
# ------------------------------------------------------------
gp_cmd("set term kitty background rgb 'black' size 1600,900")
gp_cmd("unset key")
gp_cmd("set xrange [0:10]")
gp_cmd("set yrange [-3:3]")
gp_cmd("set xlabel 'x' tc rgb '#d8dee9'")
gp_cmd("set ylabel 'f(x)' tc rgb '#d8dee9'")
gp_cmd("set border lc rgb '#4c566a'")
gp_cmd("set tics textcolor rgb '#4c566a'")

gp_cmd(
    "set label 10 "
    "'space: sample   a/z: move x   j/k: ℓ   p: mean   v: variance   c: clear   q: quit' "
    "at screen 0.05,0.96 tc rgb '#d8dee9'"
)

# ------------------------------------------------------------
# state
# ------------------------------------------------------------
xs = np.linspace(0.0, 10.0, 300)
rng = np.random.default_rng()

ell = 0.5
x_next = 0.0
dx = 0.25

X = []
Y = []

show_mean = False  # p
show_variance = False  # v


# ------------------------------------------------------------
# GP posterior
# ------------------------------------------------------------
def posterior():
    if not X:
        return np.zeros_like(xs), np.ones_like(xs)

    Xv = np.array(X)
    Yv = np.array(Y)

    K = k_sqexp(Xv, Xv, ell) + 1e-12 * np.eye(len(X))
    Ks = k_sqexp(xs, Xv, ell)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yv))

    mu = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    var = 1.0 - np.sum(v * v, axis=0)

    return mu, np.sqrt(np.maximum(var, 0))


def conditional_at_x():
    if not X:
        return 0.0, 1.0

    Xv = np.array(X)
    Yv = np.array(Y)

    K = k_sqexp(Xv, Xv, ell) + 1e-12 * np.eye(len(X))
    kx = k_sqexp(np.array([x_next]), Xv, ell)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yv))

    mu = (kx @ alpha).item()
    v = np.linalg.solve(L, kx.T)
    std = np.sqrt(max(1.0 - (v * v).sum(), 0))

    return mu, std


# ------------------------------------------------------------
# redraw
# ------------------------------------------------------------
def redraw():
    plot_terms = []

    mu = std = None
    if show_mean or show_variance:
        mu, std = posterior()

    if show_mean:
        plot_terms.append("'-' w l lw 2 lc rgb '#88c0d0'")

    if show_variance and show_mean:
        plot_terms += [
            "'-' w l lc rgb '#81a1c1'",
            "'-' w l lc rgb '#81a1c1'",
        ]

    if X:
        plot_terms.append("'-' w p pt 7 ps 1.4 lc rgb '#bf616a'")

    # lengthscale bar + guide
    plot_terms += [
        "'-' w l lw 4 lc rgb '#a3be8c'",
        "'-' w l dt 2 lc rgb '#4c566a'",
    ]

    # vertical uncertainty only if v is on
    if show_variance:
        plot_terms += [
            "'-' w l lw 3 lc rgb '#d08770'",
            "'-' w p pt 7 ps 1.5 lc rgb '#d08770'",
        ]

    # sideways Gaussian: ALWAYS ON
    plot_terms.append("'-' w l lc rgb '#d08770'")

    gp_cmd("plot " + ",".join(plot_terms))

    # posterior mean
    if show_mean:
        for x, y in zip(xs, mu):
            gp_cmd(f"{x} {y}")
        gp_cmd("e")

    # posterior bands
    if show_variance and show_mean:
        for x, y in zip(xs, mu + 2 * std):
            gp_cmd(f"{x} {y}")
        gp_cmd("e")

        for x, y in zip(xs, mu - 2 * std):
            gp_cmd(f"{x} {y}")
        gp_cmd("e")

    # samples
    if X:
        for x, y in zip(X, Y):
            gp_cmd(f"{x} {y}")
        gp_cmd("e")

    # lengthscale bar
    y_ls = 2.8
    gp_cmd(f"{9.5 - ell} {y_ls}")
    gp_cmd(f"9.5 {y_ls}")
    gp_cmd("e")

    # guide line
    gp_cmd(f"{x_next} -3")
    gp_cmd(f"{x_next} 3")
    gp_cmd("e")

    mu_n, std_n = conditional_at_x()

    # vertical bar (only if v on)
    if show_variance:
        gp_cmd(f"{x_next} {mu_n - 2 * std_n}")
        gp_cmd(f"{x_next} {mu_n + 2 * std_n}")
        gp_cmd("e")

        gp_cmd(f"{x_next} {mu_n}")
        gp_cmd("e")

    # sideways Gaussian (always)
    ys = np.linspace(mu_n - 3 * std_n, mu_n + 3 * std_n, 80)
    pdf = np.exp(-0.5 * ((ys - mu_n) / std_n) ** 2)
    pdf /= pdf.max()
    xs_pdf = x_next + 0.35 * pdf

    for x, y in zip(xs_pdf, ys):
        gp_cmd(f"{x} {y}")
    gp_cmd("e")


# ------------------------------------------------------------
# sampling
# ------------------------------------------------------------
def sample_next():
    global x_next

    mu_n, std_n = conditional_at_x()
    y = mu_n + std_n * rng.standard_normal()

    X.append(x_next)
    Y.append(y)
    x_next = min(10.0, x_next + dx)


# ------------------------------------------------------------
# reset
# ------------------------------------------------------------
def clear_all():
    global X, Y, x_next
    X = []
    Y = []
    x_next = 0.0


# ------------------------------------------------------------
# run
# ------------------------------------------------------------
cls()
redraw()

try:
    with Keys():
        while True:
            c = sys.stdin.read(1)
            if c == "q":
                break
            elif c == " ":
                sample_next()
            elif c == "a":
                x_next = max(0.0, x_next - dx)
            elif c == "z":
                x_next = min(10.0, x_next + dx)
            elif c == "j":
                ell *= 0.9
            elif c == "k":
                ell *= 1.1
            elif c == "p":
                show_mean = not show_mean
            elif c == "v":
                show_variance = not show_variance
            elif c == "c":
                clear_all()

            redraw()
finally:
    gp.stdin.close()
    cls()
