#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

import numpy as np

# ============================================================
# USER-SUPPLIED HOOKS (YOU FILL THESE IN LATER)
# ============================================================


def sample_example(params=None):
    """
    Return one waveform sample from the simulator model.
    x-axis is [0, 10].
    """
    # placeholder: slightly noisy sine with shape
    x = np.linspace(0.0, 10.0, 400)
    y = np.sin(2 * np.pi * x / 10.0)
    y += 0.2 * np.sin(4 * np.pi * x / 10.0)
    y += 0.05 * np.random.randn(len(x))
    return x, y


def fit_level(col_idx):
    """
    Fit operation for a given column index.
    Stub for now.
    """
    # later: hyperparam learning / amplitude learning
    pass


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
gp_cmd("set xrange [0:10]")
gp_cmd("set yrange [-2.5:2.5]")


# ============================================================
# STATE
# ============================================================

n_cols = 4
active_col = 3  # start on simulator
n_samples = 5

# per-column stored samples: list of lists [(x,y), ...]
samples = [[] for _ in range(n_cols)]

# initialise simulator column with samples
for _ in range(n_samples):
    samples[3].append(sample_example())


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
    samples[col] = [sample_example() for _ in range(n_samples)]


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
                active_col = max(0, active_col - 1)

            elif k == "RIGHT":
                active_col = min(n_cols - 1, active_col + 1)

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
