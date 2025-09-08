# %%
# Arc-cos(1) GP on a schematic glottal-flow-like piecewise signal
import numpy as np

from utils.plots import plt, retain

rng = np.random.default_rng(567)

# --- piecewise "glottal flow" (one cycle), centroid near x=0 ---
# vertices: (t0,0) -> (tm,A) -> (tc,0); choose t0+tm+tc=0 so centroid ~0
A = 1.0
t0, tm, tc = -0.30, 0.10, 0.20  # (-0.3 + 0.1 + 0.2)/3 = 0


def gf(t):
    t = np.asarray(t)
    up = (t >= t0) & (t < tm)
    dn = (t >= tm) & (t <= tc)
    y = np.zeros_like(t, dtype=float)
    y[up] = A * (t[up] - t0) / (tm - t0)
    y[dn] = A * (tc - t[dn]) / (tc - tm)
    return y


# --- data ---
noise = 0.001
x = np.sort(rng.uniform(-0.55, 0.55, 120))
y = gf(x) + rng.normal(0.0, noise, size=x.size)
t = np.linspace(-0.8, 0.8, 600)
y_true = gf(t)


# --- arc-cos(1) augmented kernel ---
def arccos1_aug_kernel(xv, xw, amp2=1.0, wvar=1.0, bvar=1.0, l=0.08):
    X, Y = np.meshgrid(xv / l, xw / l, indexing="ij")
    nx = np.sqrt(bvar + wvar * X * X)
    ny = np.sqrt(bvar + wvar * Y * Y)
    dot = bvar + wvar * X * Y
    cos_th = np.clip(dot / (nx * ny), -1.0, 1.0)
    th = np.arccos(cos_th)
    return (amp2 / np.pi) * (nx * ny) * (np.sin(th) + (np.pi - th) * cos_th)


def gp_post(X, y, Xs, sigma2, **hp):
    K = arccos1_aug_kernel(X, X, **hp) + sigma2 * np.eye(X.size)
    Ks = arccos1_aug_kernel(Xs, X, **hp)
    Kss = arccos1_aug_kernel(Xs, Xs, **hp)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    m = Ks @ alpha
    V = np.linalg.solve(L, Ks.T)
    C = Kss - V.T @ V
    C = (C + C.T) * 0.5
    jitter = 1e-10 * np.eye(Xs.size)
    Ls = np.linalg.cholesky(C + jitter)
    sd = np.sqrt(np.maximum(np.diag(C), 0.0))
    return m, sd, Ls

wvar, bvar, l = 0.5, 0.5, 0.1
m, sd, Ls = gp_post(x, y, t, sigma2=0.0001, amp2=1, wvar=wvar, bvar=bvar, l=l)

# --- samples from posterior ---
nsamples = 10
z = rng.standard_normal((t.size, nsamples))
samps = m[:, None] + Ls @ z

# --- plot ---
fig, ax = plt.subplots()  # figsize=(8, 3.2))
ax.plot(t, y_true, "k", lw=1, label="Glottal flow model (Alku+ 2002)")
ax.scatter(x, y, s=10, alpha=0.6)
ax.plot(t, m, lw=2, label="Arccos($n=1$) GP mean")
ax.fill_between(
    t, m - 1.96 * sd, m + 1.96 * sd, alpha=0.18, label="Samples and 95% band"
)
ax.plot(t, samps, alpha=0.35, lw=1.2, color="grey")

ax.set_xlabel("Time (a.u.)")
ax.set_ylabel("Amplitude")
ax.set_xlim(t.min(), t.max())
ax.legend()
fig.tight_layout()

retain(fig)