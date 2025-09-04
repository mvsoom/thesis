"""Maximum-entropy (I-projection) updates for AR coefficient priors with soft divisibility features"""
# %%

from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as P

# ----------------------------
# Core: MaxEnt / I-projection
# ----------------------------


def me_update(
    mu: np.ndarray,
    Sigma: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    jitter: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if C is None or C.size == 0:
        return mu.copy(), Sigma.copy()

    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    A = C @ Sigma @ C.T
    tr = float(np.trace(A)) if A.size else 0.0
    A = A + (jitter * max(tr, 1.0)) * np.eye(A.shape[0])
    b = d - C @ mu
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        lam = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        lam = np.linalg.pinv(A, rcond=1e-12) @ b
    mu_star = mu + Sigma @ C.T @ lam
    return mu_star, Sigma.copy()


# ----------------------------
# Polynomial helpers (ascending powers, q[0]=1)
# ----------------------------


def _poly_trim(c: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if c.size == 0:
        return c
    k = c.size - 1
    while k > 0 and abs(c[k]) <= tol:
        k -= 1
    return c[: k + 1].copy()


def _poly_gcd(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    a = _poly_trim(a, tol)
    b = _poly_trim(b, tol)
    if a.size == 0:
        return _poly_trim(b, tol)
    if b.size == 0:
        return _poly_trim(a, tol)
    A = a.copy()
    B = b.copy()
    while B.size > 0 and np.linalg.norm(B, ord=np.inf) > tol:
        q, r = P.polydiv(A, B)
        r = _poly_trim(r, tol)
        A, B = B, r
    g = _poly_trim(A, tol)
    if g.size == 0:
        return np.array([1.0])
    if abs(g[0]) > tol:
        g = g / g[0]
    return g


def _poly_lcm(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    a = _poly_trim(a, tol)
    b = _poly_trim(b, tol)
    if a.size == 0:
        return _poly_trim(b, tol)
    if b.size == 0:
        return _poly_trim(a, tol)
    g = _poly_gcd(a, b, tol)
    prod = P.polymul(a, b)
    q, r = P.polydiv(prod, g)
    if np.linalg.norm(_poly_trim(r, tol), ord=np.inf) > 10 * tol:
        # fall back to treating as distinct (product)
        return _poly_trim(prod, tol)
    q = _poly_trim(q, tol)
    if abs(q[0]) > tol:
        q = q / q[0]
    return q


def lcm_of_features(
    features: list[np.ndarray], tol: float = 1e-10
) -> np.ndarray:
    if not features:
        return np.array([1.0])
    m = np.array([1.0])
    for q in features:
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or not np.isclose(q[0], 1.0):
            raise ValueError("Each feature q must be 1D with q[0]==1.")
        m = _poly_lcm(m, q, tol=tol)
    if not np.isclose(m[0], 1.0):
        m = m / m[0]
    return _poly_trim(m, tol)


# ----------------------------
# Remainder-map constraints
# ----------------------------


def remainder_constraint_matrix(
    q: np.ndarray, Pdeg: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build F, c so that remainder of A(z)=1+sum_{i=1}^Pdeg a_i z^i modulo Q(z) is
        r(z) = c + sum_{i=1}^Pdeg a_i * col_i,
    with deg r < deg Q. Divisibility <=> r(z) == 0 <=> F a + c = 0.
    Returns:
        F: (m, Pdeg), c: (m,), where m = deg(Q).
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not np.isclose(q[0], 1.0):
        raise ValueError("q must be monic at constant term: q[0]==1.")
    m = q.size - 1
    if m == 0:
        return np.zeros((0, Pdeg)), np.zeros((0,))

    def rem_of_power(i: int) -> np.ndarray:
        e = np.zeros(i + 1, dtype=float)
        e[i] = 1.0
        _, r = P.polydiv(e, q)
        r = _poly_trim(r, tol=1e-14)
        if r.size < m:
            r = np.pad(r, (0, m - r.size))
        return r[:m]

    c = rem_of_power(0)
    F_cols = []
    for i in range(1, Pdeg + 1):
        F_cols.append(rem_of_power(i))
    F = np.stack(F_cols, axis=1) if F_cols else np.zeros((m, 0))
    return F, c


def constraints_from_features(
    Pdeg: int, features: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute M = lcm(features) and return (C, d) for T(a)=C a - d with T(a)=0 equivalent to M | A.
    Here C = F_M and d = -c_M where F_M a + c_M is the remainder of A modulo M.
    """
    M = lcm_of_features(features)
    F, c = remainder_constraint_matrix(M, Pdeg)
    C = F
    d = -c
    return C, d


# ----------------------------
# Feature builders Q(L) (q[0]=1)
# ----------------------------


def unit_root(which: str = "dc") -> np.ndarray:
    if which == "dc":
        return np.array([1.0, -1.0])
    if which == "nyquist":
        return np.array([1.0, 1.0])
    raise ValueError("which must be 'dc' or 'nyquist'.")


def real_pole(rho: float) -> np.ndarray:
    return np.array([1.0, -float(rho)])


def pole_pair(r: float, omega: float) -> np.ndarray:
    q1 = -2.0 * float(r) * float(np.cos(omega))
    q2 = float(r) ** 2
    return np.array([1.0, q1, q2])


def spectral_tilt(rho: float, k: int = 1) -> np.ndarray:
    rho = float(rho)
    base = np.array([1.0, -rho])
    out = np.array([1.0])
    for _ in range(int(k)):
        out = np.convolve(out, base)
    if not np.isclose(out[0], 1.0):
        out = out / out[0]
    return out


def seasonal_root(s: int) -> np.ndarray:
    s = int(s)
    q = np.zeros(s + 1, dtype=float)
    q[0] = 1.0
    q[s] = -1.0
    return q


def custom_factor(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not np.isclose(q[0], 1.0):
        raise ValueError(
            "custom_factor expects a 1D monic vector with q[0]==1."
        )
    return q


# ----------------------------
# High-level wrappers
# ----------------------------


def me_soft_divisibility(
    mu: np.ndarray,
    Sigma: np.ndarray,
    Pdeg: int,
    features: list[np.ndarray],
    jitter: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    C, d = constraints_from_features(Pdeg, features)
    if C is None or d is None or C.size == 0:
        return np.asarray(mu, float), np.asarray(Sigma, float)
    return me_update(mu, Sigma, C, d, jitter=jitter)


def gaussian_condition_on_divisibility(
    mu: np.ndarray,
    Sigma: np.ndarray,
    Pdeg: int,
    features: list[np.ndarray],
    jitter: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard divisibility: condition the Gaussian on F a + c = 0.
    """
    M = lcm_of_features(features)
    F, c = remainder_constraint_matrix(M, Pdeg)
    if F.size == 0:
        return np.asarray(mu, float), np.asarray(Sigma, float)
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    A = F @ Sigma @ F.T
    tr = float(np.trace(A)) if A.size else 0.0
    A = A + (jitter * max(tr, 1.0)) * np.eye(A.shape[0])
    b = F @ mu + c
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        alpha = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(A, rcond=1e-12) @ b
    mu_c = mu - Sigma @ F.T @ alpha
    Sigma_c = Sigma - Sigma @ F.T @ np.linalg.pinv(A, rcond=1e-12) @ F @ Sigma
    return mu_c, Sigma_c


# -----------------
# Small sanity demo
# -----------------

if __name__ == "__main__":
    from IPython.display import display

    from ar import spectrum
    from utils.plots import plt

    Pdeg = 6
    lam = 0.1
    mu0 = np.zeros(Pdeg)
    Sigma0 = lam * np.eye(Pdeg)

    fs = 8000.0
    f1, f3 = 500.0, 2000.0
    w1, w3 = 2.0 * np.pi * f1 / fs, 2.0 * np.pi * f3 / fs
    r1, r3 = 0.95, 0.92
    q1 = pole_pair(r1, w1)
    q3 = pole_pair(r3, w3)
    qtilt = spectral_tilt(rho=0.8, k=2)

    features = [q1, q3, qtilt]

    C, d = constraints_from_features(Pdeg, features)
    mu_star, Sigma_star = me_soft_divisibility(mu0, Sigma0, Pdeg, features)

    def get_power_spectrum_samples(mu, Sigma, n=5):
        a = np.random.default_rng().multivariate_normal(mu, Sigma, size=n)

        def power_spectrum(a):
            f, p = spectrum.ar_power_spectrum(a, fs)
            return 10 * np.log10(p)

        return np.array([power_spectrum(ai) for ai in a])

    np.set_printoptions(precision=4, suppress=True)
    print("M (lcm) =", lcm_of_features(features))
    print(
        "C shape =",
        None if C is None else C.shape,
        "d shape =",
        None if d is None else d.shape,
    )
    print("mu0     =", mu0)
    print("mu*     =", mu_star)

    f, power0 = spectrum.ar_power_spectrum(mu0, fs)
    f, power_star = spectrum.ar_power_spectrum(mu_star, fs)

    def plot_samples(ax, f, S, color, label="Samples"):
        lines = ax.plot(f, S.T, color=color, alpha=0.3)
        lines[0].set_label(label)
        for ln in lines[1:]:
            ln.set_label("_nolegend_")
        return lines

    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

    # left: initial
    ax0.plot(f, 10 * np.log10(power0), color="C0", lw=2, label="Mean")
    s0 = get_power_spectrum_samples(mu0, Sigma0, n=5)
    plot_samples(ax0, f, s0, color="C0")
    ax0.set_title("Initial: $a \sim \mathcal{N}(0, \lambda I_P)$")
    ax0.set_xlabel("Frequency (Hz)")
    ax0.set_ylabel("Power (dB)")
    ax0.legend(loc="upper right")

    # right: after ME update
    ax1.plot(f, 10 * np.log10(power_star), color="C1", lw=2, label="Mean")
    s_star = get_power_spectrum_samples(mu_star, Sigma_star, n=10)
    plot_samples(ax1, f, s_star, color="C1")
    ax1.set_title("After ME update: $a \sim \mathcal{N}(\\mu^*, \\Sigma^*)$")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.legend(loc="upper right")

    fig.suptitle(f"Prior spectra ($P={Pdeg}$)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    display(fig)
