# Step function example from https://tinygp.readthedocs.io/en/stable/tutorials/transforms.html
# %%
import matplotlib.pyplot as plt
import numpy as np

random = np.random.default_rng(567)

noise = 0.1

x = np.sort(random.uniform(-1, 1, 100))
y = 2 * (x > 0) - 1 + random.normal(0.0, noise, len(x))
t = np.linspace(-2.5, 2.5, 500)

plt.plot(t, 2 * (t > 0) - 1, "k", lw=1, label="truth")
plt.plot(x, y, ".k", label="data")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.3, 1.3)
plt.xlabel("x")
plt.ylabel("y")
_ = plt.legend()

# %%


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


wvar = 1.0
bvar = 0.0001
l = 0.01

m, sd = gp_post(x, y, t, 1e-6, amp2=1.0, wvar=wvar, bvar=bvar, l=l)

plt.scatter(x, y, s=12, alpha=0.7)
plt.plot(t, m, lw=2, label="arc-cos GP mean")
plt.fill_between(t, m - 1.96 * sd, m + 1.96 * sd, alpha=0.2)
plt.legend()
plt.show()

# %%
# Show on larger interval
tt = np.linspace(-50, 50, 1000)
m, sd = gp_post(x, y, tt, 1e-6, amp2=1.0, wvar=wvar, bvar=bvar, l=l)

plt.scatter(x, y, s=12, alpha=0.7)
plt.plot(tt, m, lw=2, label="arc-cos GP mean")
plt.fill_between(tt, m - 1.96 * sd, m + 1.96 * sd, alpha=0.2)
plt.legend()
plt.show()
# %%
plt.plot(sd)
# %%
