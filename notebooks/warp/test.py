# %%
import matplotlib.pyplot as plt
import numpy as np


def k_sqexp(x, ell, sig):
    x = np.asarray(x, float)
    d2 = (x[:, None] - x[None, :]) ** 2
    return (sig**2) * np.exp(-0.5 * d2 / (ell**2))


def trapz_cum(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    dx = np.diff(x)
    area = 0.5 * dx * (y[:-1] + y[1:])
    out = np.empty_like(x)
    out[0] = 0.0
    out[1:] = np.cumsum(area)
    return out


# hyperparams (set these from your bootstrap later)
T0 = 0.007  # seconds, ~7 ms
mu = np.log(T0)  # mean of log-period
sigma_logT = 0.25  # SD of log-period (example)
ell = 40  # lengthscale in "pitch periods" (example, long -> tiny jitter)

# tau grid: continuous cycle index covering roughly a 32 ms frame (~4-5 periods at 7 ms)
tau_max = 10
m = 600
tau = np.linspace(0.0, tau_max, m)

K = k_sqexp(tau, ell=ell, sig=sigma_logT) + 1e-8 * np.eye(m)

rng = np.random.default_rng(3)


def sample_warp(n_samp=5, c_prior="uniform_window"):
    g = rng.multivariate_normal(mean=np.full(m, mu), cov=K, size=n_samp)
    T = np.exp(g)  # period (seconds) as function of tau
    t_int = np.array(
        [trapz_cum(T[i], tau) for i in range(n_samp)]
    )  # integral part

    if c_prior == "uniform_window":
        c = rng.uniform(
            0.0, 0.032, size=n_samp
        )  # seconds, random absolute offset
    elif c_prior == "zero":
        c = np.zeros(n_samp)
    else:
        raise ValueError("bad c_prior")

    t = t_int + c[:, None]
    return g, T, t, c


g_s, T_s, t_s, c_s = sample_warp()

plt.figure()
for i in range(len(T_s)):
    plt.plot(tau, 1.0 / T_s[i])
plt.xlabel("tau (cycle index, continuous)")
plt.ylabel("f0(tau) [Hz]")
plt.title("Prior samples: f0(tau)=1/exp(g(tau))")
plt.ylim(50, 450)
plt.show()

plt.figure()
for i in range(len(t_s)):
    plt.plot(tau, 1000.0 * t_s[i])
plt.xlabel("tau")
plt.ylabel("t(tau) [ms]")
plt.title("Prior samples: t(tau)=c+integral exp(g)")
plt.show()


# %%
def sample_g(n):
    return rng.multivariate_normal(np.full(m, mu), K, size=n)


# draw many samples to estimate marginals
n_samp = 10000
g = sample_g(n_samp)
T = np.exp(g)  # period in seconds
f0 = 1.0 / T  # Hz

# ---- marginal pitch distribution ----
f0_all = f0.ravel()

plt.figure()
plt.hist(f0_all, bins=80, density=True)
plt.xlabel("f0 [Hz]")
plt.ylabel("density")
plt.title("Marginal pitch distribution from prior")
plt.xlim(50, 450)
plt.show()

# ---- period-to-period jitter ----
# clinical jitter: mean absolute successive difference / mean period
# computed per trajectory, then pooled

jitters = []
for i in range(n_samp):
    Ti = T[i]
    # restrict to interior to avoid edge effects
    dTi = np.abs(np.diff(Ti))
    jitter_i = np.mean(dTi) / np.mean(Ti)
    jitters.append(jitter_i)

jitters = np.asarray(jitters)

plt.figure()
plt.hist(100 * jitters, bins=60, density=True)
plt.xlabel("Jitter [%]")
plt.ylabel("density")
plt.title("Period-to-period jitter (ell = {})".format(ell))
plt.show()

np.mean(jitters), np.std(jitters)


# %%
def build_A_trapz(tau_grid, tau_obs):
    """
    Build A so that (A f)[i] approximates integral_0^{tau_obs[i]} f(u) du
    given f evaluated on tau_grid.
    Uses trapezoids on full grid intervals plus a partial trapezoid for the last interval.
    """
    tau_grid = np.asarray(tau_grid, float)
    tau_obs = np.asarray(tau_obs, float)
    dx = np.diff(tau_grid)
    A = np.zeros((len(tau_obs), len(tau_grid)))

    for i, tobs in enumerate(tau_obs):
        if tobs <= tau_grid[0]:
            continue

        if tobs >= tau_grid[-1]:
            w = 0.5 * dx
            A[i, :-1] += w
            A[i, 1:] += w
            continue

        j = np.searchsorted(tau_grid, tobs) - 1  # interval [j, j+1]
        if j > 0:
            w = 0.5 * dx[:j]
            A[i, :j] += w
            A[i, 1 : j + 1] += w

        t0 = tau_grid[j]
        t1 = tau_grid[j + 1]
        dt = t1 - t0
        d = tobs - t0

        # weights for integral over partial interval with linear interp between f[j] and f[j+1]
        A[i, j] += d - 0.5 * (d * d) / dt
        A[i, j + 1] += 0.5 * (d * d) / dt

    return A


# NOTE: conditioning will invert matrices; keep m modest while prototyping.
m = 220
tau = np.linspace(0.0, tau_max, m)

K = k_sqexp(tau, ell=ell, sig=sigma_logT) + 1e-8 * np.eye(m)
K_chol = np.linalg.cholesky(K)


def Kinv_mv(v):
    # K^{-1} v
    return np.linalg.solve(K_chol.T, np.linalg.solve(K_chol, v))


rng = np.random.default_rng(3)


def T_to_t(T, tau_grid, c=0.0):
    return trapz_cum(T, tau_grid) + c


# %%

# ----------------------------
# Likelihood + MAP conditioning
# ----------------------------

# Likelihood:
# y | g, c ~ N( c + A exp(g), sig_eps^2 I )
#
# We profile out c in closed form (isotropic noise):
# c_hat(g) = mean( y - A exp(g) )


def solve_c_hat(y, f):
    return np.mean(y - f)


def nlp_and_grad(g, y, A, sig_eps):
    """
    Negative log posterior (up to constants) and gradient wrt g:

    nlp(g) = 0.5 (g-mu)^T K^{-1} (g-mu) + 0.5/s^2 || y - c_hat(g) - A exp(g) ||^2
    """
    eg = np.exp(g)
    f = A @ eg
    c_hat = solve_c_hat(y, f)
    r = (y - c_hat) - f

    dm = g - mu
    nlp_prior = 0.5 * dm @ Kinv_mv(dm)
    grad_prior = Kinv_mv(dm)

    invs2 = 1.0 / (sig_eps * sig_eps)
    nlp_like = 0.5 * invs2 * (r @ r)

    # grad_like = -(1/s^2) * diag(exp(g)) A^T r
    grad_like = -invs2 * (eg * (A.T @ r))

    return nlp_prior + nlp_like, grad_prior + grad_like, c_hat


def map_opt(y, A, sig_eps, iters=80, step=0.2):
    """
    Barebones gradient descent with step shrink/expand.
    """
    g = np.full(m, mu)
    val, grad, c_hat = nlp_and_grad(g, y, A, sig_eps)

    for _ in range(iters):
        g_try = g - step * grad
        val_try, grad_try, c_hat_try = nlp_and_grad(g_try, y, A, sig_eps)

        if val_try < val:
            g, val, grad, c_hat = g_try, val_try, grad_try, c_hat_try
            step *= 1.05
        else:
            step *= 0.5

        if np.linalg.norm(grad) / np.sqrt(m) < 5e-5:
            break

    return g, c_hat


def laplace_posterior_sampler(g_map, c_hat, y, A, sig_eps, K, K_chol, rng):
    """
    Returns a function draw(ns) that produces ns approximate posterior samples of g
    using linearization at g_map and conditioning-by-sampling.

    Model:
      y ~= c_hat + A exp(g) + eps, eps ~ N(0, sig_eps^2 I)
    Linearize at g_map:
      y_tilde := y - c_hat - f_map ~= J x + eps, x ~ N(0, K), x = g - g_map
    """
    eg = np.exp(g_map)
    f_map = A @ eg
    y_tilde = y - c_hat - f_map

    # J = A diag(exp(g_map))
    # so J x = A (exp(g_map) * x) where * is elementwise
    # We'll materialize J as (n_obs x m) for simplicity; n_obs is tiny.
    J = A * eg[None, :]

    n_obs = y.shape[0]
    R = (sig_eps * sig_eps) * np.eye(n_obs)

    # S = J K J^T + R  (n_obs x n_obs)
    # compute JK = J @ K without forming K^{-1}
    JK = J @ K
    S = JK @ J.T + R
    S_chol = np.linalg.cholesky(S)

    # Precompute solve(S, .) via chol
    def solve_S(v):
        # v shape (n_obs,) or (n_obs, k)
        tmp = np.linalg.solve(S_chol, v)
        return np.linalg.solve(S_chol.T, tmp)

    # Precompute K J^T (m x n_obs)
    KJt = K @ J.T

    # Precompute sqrt(R) for sampling eps
    R_chol = sig_eps * np.eye(n_obs)

    # draw function: conditioning by sampling for linear Gaussian model
    #
    # Algorithm (exact for linear Gaussian):
    #   x0 ~ N(0, K)
    #   e0 ~ N(0, R)
    #   y0 = J x0 + e0
    #   x = x0 + K J^T S^{-1} (y_tilde - y0)
    #   g = g_map + x
    def draw(ns):
        # sample x0 via K_chol
        z = rng.standard_normal((ns, K.shape[0]))
        x0 = z @ K_chol.T  # (ns, m)

        # sample e0
        w = rng.standard_normal((ns, n_obs))
        e0 = w @ R_chol.T  # (ns, n_obs)

        # y0
        y0 = x0 @ J.T + e0  # (ns, n_obs)

        # innovation
        innov = y_tilde[None, :] - y0  # (ns, n_obs)

        # delta = KJt solve(S, innov^T)
        delta = (KJt @ solve_S(innov.T)).T  # (ns, m)

        x = x0 + delta
        return g_map[None, :] + x

    return draw


# ----------------------------
# Emulation 1: condition on mock estimates (O(5) points)
# ----------------------------

# sample a "true" warp from the prior
g_true = sample_g(1)[0]
c_true = 0.010

tau_obs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
A = build_A_trapz(tau, tau_obs)

y_true = c_true + (A @ np.exp(g_true))

# emulate detector
b_true = 0.0015
sig_eps = 0.0007
y_hat = y_true + b_true + rng.normal(0.0, sig_eps, size=len(tau_obs))

# MAP
g_map, c_hat = map_opt(y_hat, A, sig_eps)
t_map = T_to_t(np.exp(g_map), tau, c=c_hat)

# posterior sampler (Laplace)
draw = laplace_posterior_sampler(
    g_map=g_map,
    c_hat=c_hat,
    y=y_hat,
    A=A,
    sig_eps=sig_eps,
    K=K,
    K_chol=K_chol,
    rng=rng,
)

g_post = draw(3)
t_post = [T_to_t(np.exp(gp), tau, c=c_hat) for gp in g_post]

# plot
plt.figure()
plt.plot(tau, 1000.0 * t_map, "k", lw=2, label="MAP")
for i, tp in enumerate(t_post):
    plt.plot(tau, 1000.0 * tp, "--", lw=1.5, label=f"posterior {i + 1}")
plt.scatter(tau_obs, 1000.0 * y_hat, c="r", zorder=5, label="observations")
plt.xlabel("tau")
plt.ylabel("t(tau) [ms]")
plt.title("Conditioning on 5 mock GCI estimates")
plt.legend()
plt.show()

# %%

# ----------------------------
# Emulation 2: condition on a single pseudo point
# The slope of curves fanning out encodes prior f0 variation
# Still needs WORK
# ----------------------------

tau_anchor = np.array([0.0])
A1 = build_A_trapz(tau, tau_anchor)

y_anchor = np.array([0.012])  # 12 ms
sig_anchor = 1e-5

# MAP
g_map2, c_hat2 = map_opt(y_anchor, A1, sig_anchor)
t_map2 = T_to_t(np.exp(g_map2), tau, c=c_hat2)

# posterior sampler
draw2 = laplace_posterior_sampler(
    g_map=g_map2,
    c_hat=c_hat2,
    y=y_anchor,
    A=A1,
    sig_eps=sig_anchor,
    K=K,
    K_chol=K_chol,
    rng=rng,
)

g_post2 = draw2(400)
t_post2 = [T_to_t(np.exp(gp), tau, c=c_hat2) for gp in g_post2]

# plot
plt.figure()
plt.plot(tau, 1000.0 * t_map2, "k", lw=2, label="MAP")
pitches = []
for i, tp in enumerate(t_post2):
    plt.plot(
        tau,
        1000.0 * tp,
        "--",
        lw=1.5,
        alpha=0.1,
        label="posterior" if i == 0 else None,
    )
    pitches += [1 / (np.diff(tp)[0] / np.diff(tau)[0])]
plt.scatter(
    tau_anchor, 1000.0 * y_anchor, c="r", zorder=5, label="pseudo point"
)
plt.xlabel("tau")
plt.ylabel("t(tau) [ms]")
plt.title("Conditioning on single pseudo point t(0)=12 ms")
plt.legend()
plt.show()

plt.figure()
plt.hist(pitches, bins=60, density=True)
plt.xlabel("f0 at tau=0 [Hz]")
plt.ylabel("density")
plt.title("Posterior marginal pitch at tau=0")
plt.show()

# FIXME: put tau_anchor = np.array([3.0]) and observe collapse of posterior towards tau = 0
