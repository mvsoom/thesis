# Filon quadrature for:
#   I(m,f) = ∫_{t1}^{t2} exp(i m atan(t - t0)) * exp(-i 2 pi f t) dt
#
# Error tolerance: we get 1e-8 for m up to 50, then degrades to 1e-4 for m=500
# This is acceptable as the c_m^(d) coeffs decay at least as m^{-2}.
# Errors do not depend on scale of t1,t2,t0 (ie if normalized (in units of T) or in absolute msec units)
#
# Design choices (simple + safe):
# - For f == 0: change variable u = atan(t - t0), dt = sec^2(u) du, then Filon in u
# - For f  > 0: plain Filon in t on uniform panels over [t1,t2], NO special centering at t0
# - fixed N and fixed PANELS (JAX friendly)
# - complex64 everywhere
#
# Oracle: scipy quad with weight="cos"/"sin" (your chunked oracle)
# %%

import random
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from gfm.ack import DiagonalTACK, compute_Jd

# ============================================================
# PARAMS
# ============================================================

DEFAULT_M = 500

N = 129  # odd > 1
PANELS = 32  # total panels

DTYPE = jnp.complex128

ORACLE_PARTS = 64
ORACLE_EPS = 1e-11
ORACLE_LIMIT = 300


def quad_oracle(
    m, f, t1, t2, t0, normalized=True, d=0, sigma_b=1.0, sigma_c=1.0
):
    """Numerically integrate"""
    w = 2.0 * np.pi * f
    edges = np.linspace(t1, t2, ORACLE_PARTS + 1)
    beta = sigma_b / sigma_c

    if normalized:
        Jd0 = compute_Jd(d, 1.0, 0.0)
        scale = 1 / np.sqrt(Jd0)

        def cos_m(t):
            tau = t - t0
            return scale * np.cos(m * np.arctan(tau / beta))

        def sin_m(t):
            tau = t - t0
            return scale * np.sin(m * np.arctan(tau / beta))
    else:
        scale = 1.0 / np.sqrt(2.0 * np.pi)

        def cos_m(t):
            tau = t - t0
            poly = sigma_c**d * (beta**2 + tau**2) ** (d / 2)
            return scale * poly * np.cos(m * np.arctan(tau / beta))

        def sin_m(t):
            tau = t - t0
            poly = sigma_c**d * (beta**2 + tau**2) ** (d / 2)
            return scale * poly * np.sin(m * np.arctan(tau / beta))

    tot = 0.0 + 0.0j

    if w == 0.0:
        for i in range(ORACLE_PARTS):
            a, b = edges[i], edges[i + 1]
            re = quad(
                cos_m,
                a,
                b,
                epsabs=ORACLE_EPS,
                epsrel=ORACLE_EPS,
                limit=ORACLE_LIMIT,
            )[0]
            im = quad(
                sin_m,
                a,
                b,
                epsabs=ORACLE_EPS,
                epsrel=ORACLE_EPS,
                limit=ORACLE_LIMIT,
            )[0]
            tot += re + 1j * im
        return tot

    for i in range(ORACLE_PARTS):
        a, b = edges[i], edges[i + 1]
        Ac = quad(
            cos_m,
            a,
            b,
            weight="cos",
            wvar=w,
            epsabs=ORACLE_EPS,
            epsrel=ORACLE_EPS,
            limit=ORACLE_LIMIT,
        )[0]
        Bs = quad(
            sin_m,
            a,
            b,
            weight="sin",
            wvar=w,
            epsabs=ORACLE_EPS,
            epsrel=ORACLE_EPS,
            limit=ORACLE_LIMIT,
        )[0]
        Bc = quad(
            sin_m,
            a,
            b,
            weight="cos",
            wvar=w,
            epsabs=ORACLE_EPS,
            epsrel=ORACLE_EPS,
            limit=ORACLE_LIMIT,
        )[0]
        As = quad(
            cos_m,
            a,
            b,
            weight="sin",
            wvar=w,
            epsabs=ORACLE_EPS,
            epsrel=ORACLE_EPS,
            limit=ORACLE_LIMIT,
        )[0]
        tot += (Ac + Bs) + 1j * (Bc - As)

    return tot


# ============================================================
# FILON CORE
# ============================================================


@jax.jit
def filon_abg(theta):
    ath = jnp.abs(theta)

    def series(th):
        th2 = th * th
        th3 = th2 * th
        th4 = th2 * th2
        th5 = th3 * th2
        th6 = th4 * th2
        th7 = th5 * th2
        th8 = th4 * th4

        alpha = 2 * th3 / 45 - 2 * th5 / 315 + 2 * th7 / 4725
        beta = (
            2 / 3
            + 2 * th2 / 15
            - 4 * th4 / 105
            + 2 * th6 / 567
            - 4 * th8 / 22275
        )
        gamma = 4 / 3 - 2 * th2 / 15 + th4 / 210 - th6 / 11340
        return alpha, beta, gamma

    def closed(th):
        s = jnp.sin(th)
        c = jnp.cos(th)
        th2 = th * th
        th3 = th2 * th
        alpha = (th2 + th * s * c - 2 * s * s) / th3
        beta = (2 * th + 2 * th * c * c - 4 * s * c) / th3
        gamma = 4 * (s - th * c) / th3
        return alpha, beta, gamma

    return jax.lax.cond(ath <= (1.0 / 6.0), series, closed, theta)


@jax.jit
def filon_tab_iexp(ftab, a, b, omega):
    # ∫ f(x) exp(i omega x) dx over [a,b], ftab on uniform mesh, N odd
    n = ftab.shape[0]
    h = (b - a) / (n - 1)
    theta = omega * h
    alpha, beta, gamma = filon_abg(theta)

    j = jnp.arange(n)
    x = a + h * j

    cosx = jnp.cos(omega * x)
    sinx = jnp.sin(omega * x)

    even = jnp.arange(0, n, 2)
    odd = jnp.arange(1, n - 1, 2)

    c2n = (ftab[even] * cosx[even]).sum() - 0.5 * (
        ftab[0] * cosx[0] + ftab[-1] * cosx[-1]
    )
    c2nm1 = (ftab[odd] * cosx[odd]).sum()

    s2n = (ftab[even] * sinx[even]).sum() - 0.5 * (
        ftab[0] * sinx[0] + ftab[-1] * sinx[-1]
    )
    s2nm1 = (ftab[odd] * sinx[odd]).sum()

    re = h * (
        alpha * (ftab[-1] * jnp.sin(omega * b) - ftab[0] * jnp.sin(omega * a))
        + beta * c2n
        + gamma * c2nm1
    )
    im = h * (
        alpha * (ftab[0] * jnp.cos(omega * a) - ftab[-1] * jnp.cos(omega * b))
        + beta * s2n
        + gamma * s2nm1
    )

    return (re + 1j * im).astype(DTYPE)


# ============================================================
# ONE INTEGRAL (scalar m,f). caller can vmap
# ============================================================


@jax.jit
def filon_one(m, f, t1, t2, t0):
    # We follow your convention: caller vmaps; this stays scalar.
    a = jnp.minimum(t1, t2)
    b = jnp.maximum(t1, t2)
    sgn = jnp.where(t2 >= t1, 1.0, -1.0)

    w = 2 * jnp.pi * f
    omega_t = -w  # exp(-i w t) = exp(i omega_t t)

    def eval_t():
        edges = jnp.linspace(a, b, PANELS + 1)

        def body(i, acc):
            p = edges[i]
            q = edges[i + 1]

            j = jnp.arange(N)
            h = (q - p) / (N - 1)
            t = p + h * j

            psi = jnp.arctan(t - t0)
            ftab = jnp.exp(1j * m * psi).astype(DTYPE)

            return acc + filon_tab_iexp(ftab, p, q, omega_t)

        acc0 = jnp.array(0.0 + 0.0j, DTYPE)
        return jax.lax.fori_loop(0, PANELS, body, acc0)

    val = eval_t()
    return sgn * val


def filon_pack(k: DiagonalTACK, m, f, t1, t2):
    # We follow your convention: caller vmaps; this stays scalar.
    a = jnp.minimum(t1, t2)
    b = jnp.maximum(t1, t2)
    sgn = jnp.where(t2 >= t1, 1.0, -1.0)

    w = 2 * jnp.pi * f
    omega_t = -w  # exp(-i w t) = exp(i omega_t t)

    def eval_t():
        edges = jnp.linspace(a, b, PANELS + 1)

        def body(i, acc):
            p = edges[i]
            q = edges[i + 1]

            j = jnp.arange(N)
            h = (q - p) / (N - 1)
            t = p + h * j

            ftab = jax.vmap(lambda t: k.fourier_integrand(t, m))(t).astype(
                DTYPE
            )

            return acc + filon_tab_iexp(ftab, p, q, omega_t)

        acc0 = jnp.array(0.0 + 0.0j, DTYPE)
        return jax.lax.fori_loop(0, PANELS, body, acc0)

    val = eval_t()
    return sgn * val


f = 0
m = 0
t0 = 1.23
t1 = 0.1
t2 = 4.5
normalized = False
sigma_b = 1.5
sigma_c = 1.5
d = 2

k = DiagonalTACK(
    d=d, normalized=normalized, sigma_b=sigma_b, sigma_c=sigma_c, center=t0
)

print("old:", filon_one(m, f, t1, t2, t0))
print("new:", jax.jit(filon_pack)(k, m, f, t1, t2))
print(
    "oracle:",
    quad_oracle(
        m,
        f,
        t1,
        t2,
        t0,
        normalized=normalized,
        d=d,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
    ),
)

# %%

# ============================================================
# BULK WRAPPER (for your tests)
# ============================================================


def filon_bulk_jit(ms, freqs, t1, t2, t0):
    ms = jnp.asarray(ms)
    freqs = jnp.asarray(freqs)

    def one_m(m):
        return jax.vmap(lambda f: filon_one(m, f, t1, t2, t0))(freqs)

    return jax.vmap(one_m)(ms)


# ============================================================
# TESTS
# ============================================================


def pick_even(x, k):
    idx = np.linspace(0, len(x) - 1, k, dtype=int)
    return idx, x[idx]


def pick_random(x, k):
    idx = np.random.choice(len(x), size=k, replace=False)
    idx = np.sort(idx)
    return idx, x[idx]


def harmonic_series(T, fs, include_dc=True):
    F0 = 1.0 / T
    n = np.floor(0.8 * fs / (2 * F0))
    if include_dc:
        return F0 * np.arange(0, n + 1)
    else:
        return F0 * np.arange(1, n + 1)


T1, T2, PERIOD = 0.0, 4.5, 7.0  # msec
FS = 20.0  # kHz


def smoke():
    t1, t2, T, t0 = T1, T2, PERIOD, 0.0
    ms = np.arange(0, DEFAULT_M + 1)
    freqs = harmonic_series(T, FS)

    ms_j = jnp.asarray(ms)
    freqs_j = jnp.asarray(freqs)
    timed_bulk = jax.jit(filon_bulk_jit)

    t0s = time.perf_counter()
    vals = timed_bulk(ms_j, freqs_j, t1, t2, t0)
    vals.block_until_ready()
    t1s = time.perf_counter()
    vals_np = np.array(vals)

    reps = 5
    t2s = time.perf_counter()
    for _ in range(reps):
        timed_bulk(ms_j, freqs_j, t1, t2, t0).block_until_ready()
    t3s = time.perf_counter()

    idx, test_freqs = pick_even(freqs, 5)

    for m in [0, 1, 10, 50, 100, 250, 500]:
        for fi, f in zip(idx, test_freqs):
            ref = quad_oracle_one_m_chunked(t1, t2, t0, m, f)
            err = abs(ref - vals_np[m, fi])
            print(f"smoke m={m:3d} f={f:4.1f} err={err:.3e}")

    print(f"jit+first_run_sec = {t1s - t0s:.3f}")
    print(f"steady_avg_sec    = {(t3s - t2s) / reps:.3f}  (reps={reps})")


def run_tests(scale=1.0, fs=FS, num_cases=20, tol_abs=1e-5):
    # random.seed(100)
    # np.random.seed(100)

    ms = np.arange(0, DEFAULT_M + 1)
    worst = 0.0

    T = PERIOD * scale
    fs = FS / scale

    for k in range(num_cases):
        t1 = 0.0
        t2 = random.uniform(0.3 * T, 1.0 * T)
        t0 = random.uniform(t1, t2)

        f0 = random.uniform(1 / T * 0.5, 1 / T * 3)
        freqs = harmonic_series(1 / f0, fs)

        vals = filon_bulk_jit(ms, freqs, t1, t2, t0)
        vals_np = np.array(vals)

        fidx, test_freqs = pick_even(freqs, 10)
        _, test_ms = pick_even(ms, 10)

        for fi, f in zip(fidx, test_freqs):
            for m in test_ms:
                ref = quad_oracle_one_m_chunked(t1, t2, t0, m, f)
                err = abs(ref - vals_np[m, fi])
                worst = max(worst, err)
                if err > tol_abs:
                    raise AssertionError(
                        f"FAIL case (even)={k} m={m} f={f} err={err}"
                    )

        print(f"case (even) {k + 1:02d}/{num_cases} ok")

        fidx, test_freqs = pick_random(freqs, 10)
        _, test_ms = pick_random(ms, 10)

        for fi, f in zip(fidx, test_freqs):
            for m in test_ms:
                ref = quad_oracle_one_m_chunked(t1, t2, t0, m, f)
                err = abs(ref - vals_np[m, fi])
                worst = max(worst, err)
                if err > tol_abs:
                    raise AssertionError(
                        f"FAIL case (random)={k} m={m} f={f} err={err}"
                    )

        print(f"case (random) {k + 1:02d}/{num_cases} ok")

    print("ALL TESTS PASSED")
    print(f"worst abs err = {worst:.3e}")


if __name__ == "__main__":
    print(f"N={N} PANELS={PANELS} dtype={DTYPE}")
    smoke()
    run_tests()  # msec
    run_tests(scale=0.001)  # sec
    run_tests(scale=1 / PERIOD)  # sec
