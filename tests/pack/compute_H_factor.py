# %%
from __future__ import annotations

import random

import jax
import numpy as np

from gfm.ack import DiagonalTACK
from pack import FILON_N, FILON_PANELS, PACK

MAX_M = 500
ABS_TOL = 1e-4

T1, T2, PERIOD = 0.0, 4.5, 7.0  # msec
FS = 20.0  # kHz


def harmonic_series(T, fs, include_dc=True):
    F0 = 1.0 / T
    n = int(np.floor(0.8 * fs / (2.0 * F0)))
    start = 0 if include_dc else 1
    return F0 * np.arange(start, n + 1)


def pick_even(x, k):
    idx = np.linspace(0, len(x) - 1, k, dtype=int)
    return idx, x[idx]


def pick_random(x, k):
    idx = np.sort(np.random.choice(len(x), size=k, replace=False))
    return idx, x[idx]


def build_pack(t1, t2, period, center, normalized, d, sigma_b, sigma_c, J=32):
    kernel = DiagonalTACK(
        d=d,
        normalized=normalized,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        center=center,
    )
    return PACK(
        k=kernel,
        period=period,
        t1=t1,
        t2=t2,
        J=J,
        M=MAX_M,
    )


def filon_value(pack: PACK, m, f):
    return complex(jax.device_get(pack.compute_H_factor(m, f)))


def oracle_value(pack: PACK, m, f):
    return complex(pack.compute_H_factor_oracle(m, f))


def verify_grid(pack: PACK, params, ms, freqs, tol):
    worst = 0.0
    for f in freqs:
        for m in ms:
            filon = filon_value(pack, int(m), float(f))
            oracle = oracle_value(pack, int(m), float(f))
            err = abs(filon - oracle)
            worst = max(worst, err)
            if err > tol:
                raise AssertionError(
                    f"FAIL m={m} f={f} err={err}\n\tparams={params}\n"
                )
    return worst


def smoke():
    print(f"N={FILON_N} PANELS={FILON_PANELS}")

    params = dict(
        t1=T1,
        t2=T2,
        period=PERIOD,
        center=0.0,
        d=0,
        normalized=True,
        sigma_b=1.0,
        sigma_c=1.0,
    )

    pack = build_pack(**params)
    freqs = harmonic_series(PERIOD, FS)
    _, test_freqs = pick_even(freqs, 5)

    for m in [0, 1, 10, 50, 100, 250, 500]:
        for f in test_freqs:
            filon = filon_value(pack, m, float(f))
            oracle = oracle_value(pack, m, float(f))
            err = abs(filon - oracle)
            print(f"smoke m={m:3d} f={f:4.1f} err={err:.3e}")


def run_tests(scale=1.0, fs=FS, num_cases=10, tol_abs=ABS_TOL):
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
            period=T,
            center=center,
            d=random.choice([0, 1, 2, 3]),
            normalized=random.choice([True, False]),
            sigma_b=np.exp(np.random.normal()),
            sigma_c=1.0,
        )

        pack = build_pack(**params)
        freqs = harmonic_series(1 / f0, fs)

        _, test_freqs = pick_even(freqs, 10)
        _, test_ms = pick_even(ms, 10)

        worst = max(
            worst,
            verify_grid(pack, params, test_ms, test_freqs, tol_abs),
        )
        print(f"case (even) {k + 1:02d}/{num_cases} ok")

        _, test_freqs = pick_random(freqs, 10)
        _, test_ms = pick_random(ms, 10)

        worst = max(
            worst,
            verify_grid(pack, params, test_ms, test_freqs, tol_abs),
        )
        print(f"case (random) {k + 1:02d}/{num_cases} ok")

    print("ALL TESTS PASSED")
    print(f"worst abs err = {worst:.3e}")


if __name__ == "__main__":
    smoke()

    run_tests()  # msec
    run_tests(scale=0.001)  # sec
    run_tests(scale=1 / PERIOD)  # sec
