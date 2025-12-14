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

import math
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

# ============================================================
# PARAMS
# ============================================================

DEFAULT_M = 500

N = 129  # odd > 1
PANELS = 32  # total panels

DTYPE = jnp.complex64

ORACLE_PARTS = 64
ORACLE_EPS = 1e-11
ORACLE_LIMIT = 300


# ============================================================
# ORACLE (unchanged)
# ============================================================


def quad_oracle_one_m_chunked(t1, t2, t0, m, f):
    w = 2.0 * np.pi * f
    edges = np.linspace(t1, t2, ORACLE_PARTS + 1)

    def cos_m(t):
        return np.cos(m * np.arctan(t - t0))

    def sin_m(t):
        return np.sin(m * np.arctan(t - t0))

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

    def eval_dc_u():
        # u = atan(t - t0), dt = sec^2(u) du
        ua = jnp.arctan(a - t0)
        ub = jnp.arctan(b - t0)
        edges = jnp.linspace(ua, ub, PANELS + 1)

        omega_u = m.astype(jnp.float32)  # carrier exp(i m u)

        def body(i, acc):
            p = edges[i]
            q = edges[i + 1]

            j = jnp.arange(N)
            h = (q - p) / (N - 1)
            u = p + h * j

            sec2 = (1.0 / jnp.cos(u)) ** 2
            ftab = sec2.astype(DTYPE)

            return acc + filon_tab_iexp(ftab, p, q, omega_u)

        acc0 = jnp.array(0.0 + 0.0j, DTYPE)
        return jax.lax.fori_loop(0, PANELS, body, acc0)

    val = jax.lax.cond(w == 0.0, eval_dc_u, eval_t)
    return sgn * val


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


def smoke():
    t1, t2, t0 = 0.0, 19.0, 8.8
    ms = np.arange(0, DEFAULT_M + 1)
    freqs = 0.1 * np.arange(0, 150)

    vals = filon_bulk_jit(ms, freqs, t1, t2, t0)
    vals_np = np.array(vals)

    for m in [0, 1, 10, 50, 100, 250, 500]:
        for f in [0.0, 0.3, 2.7, 9.9]:
            fi = int(round(f / 0.1))
            ref = quad_oracle_one_m_chunked(t1, t2, t0, m, f)
            err = abs(ref - vals_np[m, fi])
            print(f"smoke m={m:3d} f={f:4.1f} err={err:.3e}")


def run_tests(num_cases=30, tol_abs=1e-4):
    random.seed(5778)
    np.random.seed(7858)

    ms = np.arange(0, DEFAULT_M + 1)
    worst = 0.0

    for k in range(num_cases):
        t1 = 0.0
        t2 = random.uniform(0.1, 0.8)
        t0 = random.uniform(t1, t2)

        f0 = random.uniform(0.05, 0.45)
        H = int(min(60, math.floor(10.0 / f0)))
        freqs = f0 * np.arange(0, H + 1)

        vals = filon_bulk_jit(ms, freqs, t1, t2, t0)
        vals_np = np.array(vals)

        for fi in [0, len(freqs) // 2, -1]:
            f = freqs[fi]
            for m in [0, 1, 10, 50, 100, 250, 500]:
                ref = quad_oracle_one_m_chunked(t1, t2, t0, m, f)
                err = abs(ref - vals_np[m, fi])
                worst = max(worst, err)
                if err > tol_abs:
                    raise AssertionError(f"FAIL case={k} m={m} f={f} err={err}")

        print(f"case {k + 1:02d}/{num_cases} ok")

    print("ALL TESTS PASSED")
    print(f"worst abs err = {worst:.3e}")


# ============================================================
# WALLTIME CHECK (jit compile + steady-state)
# ============================================================


def walltime_check():
    t1, t2, t0 = 0.0, 0.6, 0.5
    ms = np.arange(0, DEFAULT_M + 1)
    freqs = 0.1 * np.arange(0, 150)

    ms_j = jnp.asarray(ms)
    freqs_j = jnp.asarray(freqs)

    def bulk(ms_j, freqs_j, t1, t2, t0):
        def one_m(m):
            return jax.vmap(lambda f: filon_one(m, f, t1, t2, t0))(freqs_j)

        return jax.vmap(one_m)(ms_j)

    bulk_jit = jax.jit(bulk)

    # compile + run once
    t0s = time.perf_counter()
    out = bulk_jit(ms_j, freqs_j, t1, t2, t0)
    out.block_until_ready()
    t1s = time.perf_counter()

    # steady-state timing
    reps = 5
    t2s = time.perf_counter()
    for _ in range(reps):
        out = bulk_jit(ms_j, freqs_j, t1, t2, t0)
        out.block_until_ready()
    t3s = time.perf_counter()

    print(f"jit+first_run_sec = {t1s - t0s:.3f}")
    print(f"steady_avg_sec    = {(t3s - t2s) / reps:.3f}  (reps={reps})")


if __name__ == "__main__":
    print(f"N={N} PANELS={PANELS} dtype={DTYPE}")
    smoke()
    run_tests()
    walltime_check()
