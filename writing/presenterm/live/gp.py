#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import numpy as np

# ------------------------------------------------------------
# Key bindings (shown once, pedagogical order)
# ------------------------------------------------------------
print(
    "\n"
    "Gaussian process blackboard demo\n"
    "\n"
    "space : sample at current x and advance\n"
    "a / z : move x left / right\n"
    "j / k : decrease / increase lengthscale\n"
    "p     : toggle global posterior mean and bands\n"
    "c     : clear and restart\n"
    "q     : quit\n"
)


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
gp_cmd('set term kitty background rgb "black" size 1600,900')
gp_cmd("unset key")
gp_cmd("set xrange [0:10]")
gp_cmd("set yrange [-3:3]")
gp_cmd("set border lc rgb 'gray'")
gp_cmd("set tics textcolor rgb 'gray'")

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

show_posterior = False


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


# ------------------------------------------------------------
# redraw
# ------------------------------------------------------------
def redraw():
    plot_terms = []

    if show_posterior:
        mu, std = posterior()
        plot_terms += [
            "'-' w l lc rgb '#88c0d0'",  # mean
            "'-' w l lc rgb '#81a1c1'",  # +2 sigma
            "'-' w l lc rgb '#81a1c1'",  # -2 sigma
        ]

    if X:
        plot_terms.append("'-' w p pt 7 ps 1.4 lc rgb '#bf616a'")

    plot_terms += [
        "'-' w l lw 4 lc rgb '#a3be8c'",  # lengthscale bar
        "'-' w l dt 2 lc rgb '#4c566a'",  # guide line
        "'-' w l lw 3 lc rgb '#d08770'",  # cond +-2 sigma
        "'-' w p pt 7 ps 1.5 lc rgb '#d08770'",  # cond mean
        "'-' w l lc rgb '#d08770'",  # sideways Gaussian
    ]

    gp_cmd("plot " + ",".join(plot_terms))

    # global posterior
    if show_posterior:
        for x, y in zip(xs, mu):
            gp_cmd(f"{x} {y}")
        gp_cmd("e")

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

    # conditional at x_next
    if not X:
        mu_n, std_n = 0.0, 1.0
    else:
        Xv = np.array(X)
        Yv = np.array(Y)
        K = k_sqexp(Xv, Xv, ell) + 1e-12 * np.eye(len(X))
        kx = k_sqexp(np.array([x_next]), Xv, ell)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yv))
        mu_n = (kx @ alpha).item()
        v = np.linalg.solve(L, kx.T)
        std_n = np.sqrt(max(1.0 - (v * v).sum(), 0))

    # vertical guide
    gp_cmd(f"{x_next} -3")
    gp_cmd(f"{x_next} 3")
    gp_cmd("e")

    # uncertainty bar
    gp_cmd(f"{x_next} {mu_n - 2 * std_n}")
    gp_cmd(f"{x_next} {mu_n + 2 * std_n}")
    gp_cmd("e")

    # mean dot
    gp_cmd(f"{x_next} {mu_n}")
    gp_cmd("e")

    # sideways Gaussian
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

    if not X:
        y = rng.standard_normal()
    else:
        Xv = np.array(X)
        Yv = np.array(Y)
        K = k_sqexp(Xv, Xv, ell) + 1e-12 * np.eye(len(X))
        kx = k_sqexp(np.array([x_next]), Xv, ell)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yv))
        mu = (kx @ alpha).item()
        v = np.linalg.solve(L, kx.T)
        var = max(1.0 - (v * v).sum(), 0)

        y = mu + np.sqrt(var) * rng.standard_normal()

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
                redraw()
            elif c == "a":
                x_next = max(0.0, x_next - dx)
                redraw()
            elif c == "z":
                x_next = min(10.0, x_next + dx)
                redraw()
            elif c == "j":
                ell *= 0.9
                redraw()
            elif c == "k":
                ell *= 1.1
                redraw()
            elif c == "p":
                show_posterior = not show_posterior
                redraw()
            elif c == "c":
                clear_all()
                redraw()
finally:
    gp.stdin.close()
    cls()
