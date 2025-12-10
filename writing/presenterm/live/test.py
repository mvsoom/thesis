#!/usr/bin/env python3

import subprocess
import sys
import termios
import tty

# --- gnuplot ---
gp = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE, text=True)


def gp_cmd(s):
    gp.stdin.write(s + "\n")
    gp.stdin.flush()


# --- keys ---
class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# --- setup ---
gp_cmd('set term kitty background rgb "black"')
gp_cmd("unset key")
gp_cmd("set xrange [0:10]")
gp_cmd("set yrange [-1.5:1.5]")

a = 1.0


def redraw():
    gp_cmd(f"a={a}")
    gp_cmd("plot sin(a*x)")


redraw()

# --- event loop ---
try:
    with Keys():
        while True:
            ch = sys.stdin.read(1)
            if ch == "q":
                break
            elif ch == "j":
                a -= 0.1
                redraw()
            elif ch == "k":
                a += 0.1
                redraw()
finally:
    try:
        gp.stdin.close()
    except Exception:
        pass
