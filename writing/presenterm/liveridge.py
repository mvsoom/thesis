#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import numpy as np

# ---------------- gnuplot ----------------

gp = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE, text=True)


def gp_cmd(s):
    gp.stdin.write(s + "\n")
    gp.stdin.flush()


# ---------------- keys ----------------


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ---------------- model + data ----------------

rng = np.random.default_rng(0)

x_min, x_max = 0.0, 1.0
y_min, y_max = -2.0, 2.0

n_per_cluster = 6
noise_std = 0.15

poly_deg = 10
deg_min, deg_max = 2, 20

loglam = -12.0  # effectively unregularized to start
loglam_min, loglam_max = -20.0, 10.0


def make_x(spread=0.035):
    c1 = 0.15 + spread * rng.standard_normal(n_per_cluster)
    c2 = 0.45 + spread * rng.standard_normal(n_per_cluster)
    c3 = 0.75 + spread * rng.standard_normal(n_per_cluster)
    x = np.concatenate([c1, c2, c3])
    return np.clip(np.sort(x), x_min, x_max)


def f_true(x):
    return 0.8 * x - 0.4 + 0.25 * np.sin(4 * np.pi * x)


def sample_y(x):
    return f_true(x) + noise_std * rng.standard_normal(size=x.shape)


def design(x, deg):
    z = 2.0 * (x - 0.5)
    cols = [np.ones_like(z)]
    for k in range(1, deg + 1):
        cols.append(z**k)
    return np.stack(cols, axis=1)


def ridge_fit(x, y, mask, deg, loglam_val):
    xm = x[mask]
    ym = y[mask]
    phi = design(xm, deg)
    jitter = 1e-10
    lam = 0.0 if loglam_val <= loglam_min else 10.0**loglam_val
    a = phi.T @ phi + (lam + jitter) * np.eye(phi.shape[1])
    b = phi.T @ ym
    return np.linalg.solve(a, b)


def eval_fit(w, xg, deg):
    return design(xg, deg) @ w


# initial dataset
x_obs = make_x()
y_obs = sample_y(x_obs)
active = np.ones_like(x_obs, dtype=bool)

# ---------------- plotting ----------------

print(
    "j/k: λ  |  [/]: degree  |  x: drop point  |  space: resample y  |  c: new data  |  q: quit"
)

gp_cmd("set term kitty size 1600,900 enhanced background rgb 'black'")
gp_cmd("unset key")
gp_cmd(f"set xrange [{x_min}:{x_max}]")
gp_cmd(f"set yrange [{y_min}:{y_max}]")
gp_cmd("set border lc rgb 'gray'")
gp_cmd("set tics textcolor rgb 'gray'")
gp_cmd("set xtics 0,0.2,1")
gp_cmd("set ytics -2,0.5,2")
gp_cmd("set xlabel 'x'")
gp_cmd("set ylabel 'y'")


def redraw():
    gp_cmd("unset label")
    gp_cmd("set title 'Ridge regression (polynomial basis)' tc rgb 'white'")

    n_active = int(active.sum())
    gp_cmd(
        "set label 1 sprintf('degree = %d', %d) at graph 0.03,0.94 tc rgb 'white'"
        % (poly_deg, poly_deg)
    )
    gp_cmd(
        "set label 2 sprintf('λ = 10^{%.0f}', %.1f) at graph 0.03,0.88 tc rgb 'white'"
        % (loglam, loglam)
    )
    gp_cmd(
        "set label 3 sprintf('N = %d', %d) at graph 0.03,0.82 tc rgb 'white'"
        % (n_active, n_active)
    )

    w = ridge_fit(x_obs, y_obs, active, poly_deg, loglam)
    xg = np.linspace(x_min, x_max, 500)
    yg = eval_fit(w, xg, poly_deg)

    gp_cmd(
        "plot '-' w l lw 3 lc rgb '#00d0d0', '-' w p pt 7 ps 1.4 lc rgb 'white'"
    )

    for xv, yv in zip(xg, yg):
        gp_cmd(f"{xv} {yv}")
    gp_cmd("e")

    for xv, yv, m in zip(x_obs, y_obs, active):
        if m:
            gp_cmd(f"{xv} {yv}")
    gp_cmd("e")


redraw()

# ---------------- event loop ----------------

try:
    with Keys():
        while True:
            ch = sys.stdin.read(1)
            if ch == "q":
                break

            elif ch == "j":
                loglam = max(loglam - 1.0, loglam_min)
                redraw()

            elif ch == "k":
                loglam = min(loglam + 1.0, loglam_max)
                redraw()

            elif ch == "a":
                poly_deg = max(poly_deg - 1, deg_min)
                redraw()

            elif ch == "z":
                poly_deg = min(poly_deg + 1, deg_max)
                redraw()

            elif ch == "x":
                idx = np.where(active)[0]
                if len(idx) > 2:
                    active[rng.choice(idx)] = False
                    redraw()

            elif ch == " ":
                y_obs = sample_y(x_obs)
                active[:] = True
                redraw()

            elif ch == "c":
                x_obs = make_x()
                y_obs = sample_y(x_obs)
                active = np.ones_like(x_obs, dtype=bool)
                redraw()

finally:
    try:
        gp.stdin.close()
    except Exception:
        pass
