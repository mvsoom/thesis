# %%
import matplotlib.pyplot as plt
import numpy as np


def make_piecewise(n=140, noise=0.01):
    x = np.linspace(0.0, 1.0, n)
    f = np.piecewise(
        x,
        [x < 0.28, (x >= 0.28) & (x < 0.62), x >= 0.62],
        [
            lambda t: 0.5 * np.ones_like(t),
            lambda t: 0.5 + 1.1 * (t - 0.28),
            lambda t: 0.83 + 0.25 * (t - 0.62),
        ],
    )
    y = f + np.random.default_rng(11).normal(0, noise, n)
    return x, y


def arccos1_aug_kernel(xv, xw, amp2=1.0, wvar=1.0, bvar=1.2, l=0.12):
    X, Y = np.meshgrid(xv, xw, indexing="ij")
    Xs, Ys = X / l, Y / l
    nx = np.sqrt(bvar + wvar * Xs * Xs)
    ny = np.sqrt(bvar + wvar * Ys * Ys)
    dot = bvar + wvar * Xs * Ys
    cos_th = np.clip(dot / (nx * ny), -1, 1)
    th = np.arccos(cos_th)
    return (amp2 / np.pi) * (nx * ny) * (np.sin(th) + (np.pi - th) * cos_th)


def gp_post(X, y, Xs, sigma2, **hp):
    K = arccos1_aug_kernel(X, X, **hp)
    Ks = arccos1_aug_kernel(Xs, X, **hp)
    Kss = arccos1_aug_kernel(Xs, Xs, **hp)
    L = np.linalg.cholesky(K + sigma2 * np.eye(len(X)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    m = Ks @ alpha
    V = np.linalg.solve(L, Ks.T)
    C = Kss - V.T @ V
    return m, np.sqrt(np.maximum(np.diag(C), 0))


# generate data
x, y = make_piecewise()
xs = np.linspace(0, 1, 260)
m, sd = gp_post(x, y, xs, 1e-6, amp2=1.0, wvar=1.0, bvar=1.2, l=0.12)

plt.scatter(x, y, s=12, alpha=0.7)
plt.plot(xs, m, lw=2, label="arc-cos GP mean")
plt.fill_between(xs, m - 1.96 * sd, m + 1.96 * sd, alpha=0.2)
plt.legend()
plt.show()
