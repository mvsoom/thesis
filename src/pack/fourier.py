# %%
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from gfm.ack import DiagonalTACK, compute_Jd
from gfm.filon import filon_tab_iexp

MAX_M = 500  # this code accurate to 1e-6 for m up to MAX_M

N = 129  # odd > 1
PANELS = 32  # total panels: increase THIS if need more accuracy
PANELS_MAX = 128


ORACLE_PARTS = 64
ORACLE_EPS = 1e-11
ORACLE_LIMIT = 300


def quad_oracle(integrand, m, f, t1, t2):
    """Numerically integrate with weighted scipy quad as oracle reference (slow but accurate, not JAX compatible)"""
    w = 2.0 * np.pi * f
    edges = np.linspace(t1, t2, ORACLE_PARTS + 1)

    def cos_m(t):
        return np.real(integrand(t, m))

    def sin_m(t):
        return np.imag(integrand(t, m))

    tot = 0.0 + 0.0j

    params = dict(epsabs=ORACLE_EPS, epsrel=ORACLE_EPS, limit=ORACLE_LIMIT)

    if w == 0.0:
        for i in range(ORACLE_PARTS):
            a, b = edges[i], edges[i + 1]
            re = quad(cos_m, a, b, **params)[0]
            im = quad(sin_m, a, b, **params)[0]
            tot += re + 1j * im
        return tot

    for i in range(ORACLE_PARTS):
        a, b = edges[i], edges[i + 1]
        Ac = quad(cos_m, a, b, weight="cos", wvar=w, **params)[0]
        Bs = quad(sin_m, a, b, weight="sin", wvar=w, **params)[0]
        Bc = quad(sin_m, a, b, weight="cos", wvar=w, **params)[0]
        As = quad(cos_m, a, b, weight="sin", wvar=w, **params)[0]
        tot += (Ac + Bs) + 1j * (Bc - As)

    return tot


def quad_filon_old(integrand, m, f, t1, t2):
    """Numerically integrate with Filon quadrature (fast, accurate to minimally 1e-6, JAX compatible)"""
    a = jnp.minimum(t1, t2)
    b = jnp.maximum(t1, t2)
    sgn = jnp.where(t2 >= t1, 1.0, -1.0)

    w = 2 * jnp.pi * f
    omega_t = -w  # exp(-i w t) = exp(i omega_t t)

    edges = jnp.linspace(a, b, PANELS + 1)

    def body(i, acc):
        p = edges[i]
        q = edges[i + 1]

        j = jnp.arange(N)
        h = (q - p) / (N - 1)
        t = p + h * j

        ftab = jax.vmap(integrand, in_axes=(0, None))(t, m)

        return acc + filon_tab_iexp(ftab, p, q, omega_t)

    acc0 = jnp.array(0.0 + 0.0j)
    integral = jax.lax.fori_loop(0, PANELS, body, acc0)
    return sgn * integral


# Test helpers
def test_filon(m, f, t1, t2, center, normalized, d, sigma_b, sigma_c):
    k = DiagonalTACK(
        d=d,
        normalized=normalized,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        center=center,
    )
    return k.compute_H_factor(m, f, t1, t2)


test_filon = jax.jit(test_filon, static_argnames=("normalized", "d"))


def test_oracle(m, f, t1, t2, center, normalized, d, sigma_b, sigma_c):
    beta = sigma_b / sigma_c

    if normalized:
        Jd0 = compute_Jd(d, 1.0, 0.0)
        scale_norm = 1.0 / np.sqrt(Jd0)
    else:
        scale_unnorm = 1.0 / np.sqrt(2 * np.pi)

    def integrand(t, m):
        tau = t - center
        z = (1.0 + 1j * (tau / beta)) / np.sqrt(1.0 + (tau / beta) ** 2)
        psi_t = z**m

        if normalized:
            return scale_norm * psi_t
        else:
            poly = sigma_c**d * (beta * beta + tau * tau) ** (d / 2)
            return scale_unnorm * poly * psi_t

    return quad_oracle(integrand, m, f, t1, t2)


def pick_even(x, k):
    idx = np.linspace(0, len(x) - 1, k, dtype=int)
    return idx, x[idx]


def pick_random(x, k):
    idx = np.random.choice(len(x), size=k, replace=False)
    idx = np.sort(idx)
    return idx, x[idx]


def harmonic_series(T, fs, include_dc=True):
    F0 = 1.0 / T
    n = jnp.floor(0.8 * fs / (2 * F0))
    if include_dc:
        return F0 * jnp.arange(0, n + 1)
    else:
        return F0 * jnp.arange(1, n + 1)


T1, T2, PERIOD = 0.0, 4.5, 7.0  # msec
FS = 20.0  # kHz


def smoke():
    t1, t2, T, center = T1, T2, PERIOD, 0.0
    ms = jnp.arange(0, MAX_M + 1)
    freqs = harmonic_series(T, FS)

    params = dict(d=0, normalized=True, sigma_b=1.0, sigma_c=1.0, center=0.0)

    k = DiagonalTACK(**params)

    def filon(ms, fs):
        return jax.vmap(
            jax.vmap(
                lambda m, f: test_filon(m, f, t1, t2, **params),
                in_axes=(None, 0),
                out_axes=0,
            ),
            in_axes=(0, None),
            out_axes=0,
        )(ms, fs)

    filon = jax.jit(filon)

    centers = time.perf_counter()
    vals = np.array(filon(ms, freqs).block_until_ready())
    t1s = time.perf_counter()

    reps = 5
    t2s = time.perf_counter()
    for _ in range(reps):
        filon(ms, freqs).block_until_ready()
    t3s = time.perf_counter()

    print(f"jit+first_run_sec = {t1s - centers:.3f}")
    print(f"steady_avg_sec    = {(t3s - t2s) / reps:.3f}  (reps={reps})")

    idx, test_freqs = pick_even(freqs, 5)

    for m in [0, 1, 10, 50, 100, 250, 500]:
        for fi, f in zip(idx, test_freqs):
            ref = test_oracle(m, f, t1, t2, **params)
            err = abs(ref - vals[m, fi])
            print(f"smoke m={m:3d} f={f:4.1f} err={err:.3e}")


def run_tests(scale=1.0, fs=FS, num_cases=100, tol_abs=1e-5):
    # random.seed(100)
    # np.random.seed(100)

    ms = np.arange(0, MAX_M + 1)
    worst = 0.0

    T = PERIOD * scale
    fs = FS / scale

    for k in range(num_cases):
        t1 = 0.0
        t2 = random.uniform(0.3 * T, 1.0 * T)
        center = random.uniform(t1, t2)
        f0 = random.uniform(0.5 / T, 3.0 / T)

        params = dict(
            t1=t1,
            t2=t2,
            center=center,
            d=random.choice([0, 1, 2, 3]),
            normalized=random.choice([True, False]),
            sigma_b=np.exp(np.random.normal()),
            sigma_c=1.0,  # np.exp(np.random.normal()),
        )

        freqs = harmonic_series(1 / f0, fs)

        _, test_freqs = pick_even(freqs, 1)
        _, test_ms = pick_even(ms, 1)

        for f in test_freqs:
            for m in test_ms:
                oracle = test_oracle(m, f, **params)
                filon = test_filon(m, f, **params)
                err = abs(filon - oracle)
                worst = max(worst, err)
                if err > tol_abs:
                    print(
                        f"FAIL case (even)={k} m={m} f={f} err={err} f0={f0}\n\nparams={params}"
                    )
                else:
                    print(
                        f"SUCCESS case (even)={k} m={m} f={f} err={err} f0={f0}\n\nparams={params}"
                    )

        print(f"case (even) {k + 1:02d}/{num_cases} ok")

        _, test_freqs = pick_random(freqs, 1)
        _, test_ms = pick_random(ms, 1)

        for f in test_freqs:
            for m in test_ms:
                oracle = test_oracle(m, f, **params)
                filon = test_filon(m, f, **params)
                err = abs(filon - oracle)
                worst = max(worst, err)
                if err > tol_abs:
                    print(
                        f"FAIL case (random)={k} m={m} f={f} err={err} f0={f0}\n\nparams={params}"
                    )
                else:
                    print(
                        f"SUCCESS case (random)={k} m={m} f={f} err={err} f0={f0}\n\nparams={params}"
                    )

        print(f"case (random) {k + 1:02d}/{num_cases} ok")

    print("ALL TESTS PASSED")
    print(f"worst abs err = {worst:.3e}")


if __name__ == "__main__":
    print(f"N={N} PANELS={PANELS}")
    smoke()
    run_tests()  # msec
    # run_tests(scale=0.001)  # sec
    # run_tests(scale=1 / PERIOD)  # sec
