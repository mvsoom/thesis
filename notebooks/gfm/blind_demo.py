# %%
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from utils.lfmodel import dgf, fant_params


def conv(u, h):
    return np.convolve(u, h, mode="full")


def make_frame(u, h, d, w=800, hgt=400):
    # supersample a bit for nicer lines
    scale = 2
    W = w * scale
    H = hgt * scale

    img = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(img)

    def plot(x, y0, color):
        n = len(x)
        xs = np.linspace(0, W - 1, n)
        yh = H // 3 - 20 * scale
        x = x / (np.max(np.abs(x)) + 1e-9)
        ys = y0 + (1 - x) * yh * 0.45
        for i in range(n - 1):
            dr.line(
                [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=color, width=3
            )

    # excitation cumulative display
    U = np.cumsum(u) / len(u)
    plot(U, 30 * scale, (180, 0, 0))
    plot(h, H // 3 + 30 * scale, (0, 100, 0))
    plot(d, 2 * H // 3 + 30 * scale, (0, 0, 180))

    # downsample to final size
    img = img.resize((w, hgt), Image.Resampling.LANCZOS)

    # color negate
    img = ImageOps.invert(img)

    # add border
    img = ImageOps.expand(img, border=8)

    return img


def compensate(u, h, j, p, du, R=4):
    n_u = len(u)
    n_h = len(h)
    L = n_u + n_h - 1

    hu = np.zeros(L)
    for k in range(L):
        i = k - j
        if 0 <= i < n_h:
            hu[k] = h[i]
    rhs = -du * hu

    idxs = range(p - R, p + R + 1)
    cols = []
    for p2 in idxs:
        col = np.zeros(L)
        for k in range(L):
            i = k - p2
            if 0 <= i < n_u:
                col[k] = u[i]
        cols.append(col)
    A = np.stack(cols, axis=1)

    dh, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    u[j] += du
    for offset, val in enumerate(dh):
        p2 = p - R + offset
        if 0 <= p2 < n_h:
            h[p2] += val
    return u, h


def make_formant_ir(formants, bandwidth, fs, dur_ms):
    n = int(dur_ms * fs / 1000.0)
    t = np.arange(n) / fs
    h = np.zeros(n)
    for f in formants:
        a = np.exp(-np.pi * bandwidth * t)
        h += a * np.sin(2 * np.pi * f * t)
    h /= np.max(np.abs(h)) + 1e-12
    return h


# --- periodic excitation u_long -----------------------------------------

p = fant_params("female vowel")
t0 = np.linspace(0, p["T0"], 100)
u_one = np.asarray(dgf(t0, p), copy=True)

num_periods = 4
u_long = np.tile(u_one, num_periods)
period = len(u_one)

# random-walk phase offset
phi = 0.0
sigma = 0.02


def shift_u(u_long, phi):
    n = len(u_long)
    idx = (np.arange(n) + phi) % n
    i0 = idx.astype(int)
    i1 = (i0 + 1) % len(u_long)
    w = idx - i0
    return (1 - w) * u_long[i0] + w * u_long[i1]


# initial u
u = shift_u(u_long, phi)


# --- filter h ------------------------------------------------------------

dt = t0[1] - t0[0]
fs = 1000.0 / dt

formants = [500.0, 1000.0, 1500.0]
bandwidth = 80.0

h = make_formant_ir(formants, bandwidth, fs, dur_ms=15.0)
h = np.asarray(h, copy=True)

d = conv(u, h)


# --- animation loop ------------------------------------------------------

frames = []
steps = 10000

freq_u = 0.003
freq_h = 0.002

for ti in range(steps):
    # update phase offset (random walk)
    phi += sigma * np.random.randn()
    phi = phi % period
    u = shift_u(u_long, phi)

    noise_amp = 0.02
    pert = noise_amp * np.convolve(
        np.random.randn(len(u)), np.ones(15) / 15.0, mode="same"
    )
    u = u + pert

    # where the perturbation hits u and h
    j = int((len(u) - 1) * 0.5 * (1 + np.sin(2 * np.pi * freq_u * ti)))
    p = int((len(h) - 1) * 0.5 * (1 + np.sin(2 * np.pi * freq_h * ti + 1.7)))

    du = 0.03 * np.sin(2 * np.pi * ti / 250.0)

    u, h = compensate(u, h, j, p, du, R=5)

    if ti % 10 == 0:
        frames.append(make_frame(u.copy(), h.copy(), d.copy()))

frames[0].save(
    "blind_demo_smooth.gif",
    save_all=True,
    append_images=frames[1:],
    duration=15,
    loop=0,
)
