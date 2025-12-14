# %%

import math
import random

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from scipy.integrate import quad

# ============================================================
# GLOBAL CONFIG (BAKED IN)
# ============================================================

jax.config.update("jax_enable_x64", True)

N = 64
PANELS = 256
OMEGA_DC_EPS = 1e-14

ORACLE_EPS = 1e-12
ORACLE_LIMIT = 600


# ============================================================
# CHEBYSHEV-GAUSS-LOBATTO CACHE (BAKED IN)
# ============================================================


def _cgl_nodes(n):
    j = jnp.arange(n)
    return -jnp.cos(jnp.pi * j / (n - 1))


def _cgl_diffmat(x):
    n = x.shape[0]
    j = jnp.arange(n)

    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[-1].set(2.0)

    X = x[:, None]
    dX = X - X.T

    sgn = (-1.0) ** (j[:, None] + j[None, :])
    C = (c[:, None] / c[None, :]) * sgn

    D = jnp.where(jnp.eye(n, dtype=bool), 0.0, C / dX)
    D = D.at[jnp.diag_indices(n)].set(-jnp.sum(D, axis=1))
    return D


def _clenshaw_curtis_weights(n):
    theta = jnp.pi * jnp.arange(n) / (n - 1)
    w = jnp.zeros(n)

    for k in range(1, n - 1):
        s = 0.0
        for m in range(1, (n - 1) // 2 + 1):
            s += (2.0 / (4.0 * m * m - 1.0)) * jnp.cos(2.0 * m * theta[k])
        w = w.at[k].set(2.0 / (n - 1) * (1.0 - s))

    w0 = 1.0 / (n - 1)
    w = w.at[0].set(w0)
    w = w.at[-1].set(w0)
    return w


# bake cache once
_CGL_X = _cgl_nodes(N)
_CGL_D = _cgl_diffmat(_CGL_X).astype(jnp.complex128)
_CGL_W = _clenshaw_curtis_weights(N)


# ============================================================
# INTERNAL LEVIN CORE (SINGLE m, SINGLE omega)
# ============================================================


def _build_g(t, m, t0):
    return jnp.exp(1j * m * jnp.arctan(t - t0))


def _levin_panel(a, b, omega, m, t0):
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    t = mid + half * _CGL_X
    g = _build_g(t, m, t0)

    D = (1.0 / half) * _CGL_D
    I = jnp.eye(N, dtype=jnp.complex128)

    A = D - 1j * omega * I
    p = jla.solve(A, g)

    return p[-1] * jnp.exp(-1j * omega * b) - p[0] * jnp.exp(-1j * omega * a)


def _dc_panel(a, b, m, t0):
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    t = mid + half * _CGL_X
    g = _build_g(t, m, t0)

    return half * jnp.sum(_CGL_W * g)


# ============================================================
# PUBLIC API (SINGLE m, SINGLE f)
# ============================================================


def levin_integral_single(m, f, t1, t2, t0):
    omega = 2.0 * jnp.pi * f

    a = jnp.minimum(t1, t2)
    b = jnp.maximum(t1, t2)
    sgn = jnp.where(t2 >= t1, 1.0, -1.0)

    edges = jnp.linspace(a, b, PANELS + 1)
    is_dc = jnp.abs(omega) <= OMEGA_DC_EPS

    def body(i, acc):
        ai = edges[i]
        bi = edges[i + 1]

        contrib = jax.lax.cond(
            is_dc,
            lambda _: _dc_panel(ai, bi, m, t0),
            lambda _: _levin_panel(ai, bi, omega, m, t0),
            operand=None,
        )
        return acc + contrib

    acc0 = jnp.zeros((), dtype=jnp.complex128)
    acc = jax.lax.fori_loop(0, PANELS, body, acc0)

    return sgn * acc


levin_integral_single_jit = jax.jit(levin_integral_single)


# ============================================================
# SCIPY ORACLE (UNCHANGED, CHUNKED)
# ============================================================


def quad_oracle_one_m_chunked(
    t1, t2, t0, m, f, parts=512, eps=ORACLE_EPS, limit=ORACLE_LIMIT
):
    w = 2.0 * np.pi * f
    edges = np.linspace(t1, t2, parts + 1)

    def cos_m(t):
        return np.cos(m * np.arctan(t - t0))

    def sin_m(t):
        return np.sin(m * np.arctan(t - t0))

    tot = 0.0 + 0.0j

    if w == 0.0:
        for i in range(parts):
            a = edges[i]
            b = edges[i + 1]
            re = quad(cos_m, a, b, epsabs=eps, epsrel=eps, limit=limit)[0]
            im = quad(sin_m, a, b, epsabs=eps, epsrel=eps, limit=limit)[0]
            tot += re + 1j * im
        return tot

    for i in range(parts):
        a = edges[i]
        b = edges[i + 1]
        Acos = quad(
            cos_m,
            a,
            b,
            weight="cos",
            wvar=w,
            epsabs=eps,
            epsrel=eps,
            limit=limit,
        )[0]
        Bsin = quad(
            sin_m,
            a,
            b,
            weight="sin",
            wvar=w,
            epsabs=eps,
            epsrel=eps,
            limit=limit,
        )[0]
        Bcos = quad(
            sin_m,
            a,
            b,
            weight="cos",
            wvar=w,
            epsabs=eps,
            epsrel=eps,
            limit=limit,
        )[0]
        Asin = quad(
            cos_m,
            a,
            b,
            weight="sin",
            wvar=w,
            epsabs=eps,
            epsrel=eps,
            limit=limit,
        )[0]
        tot += (Acos + Bsin) + 1j * (Bcos - Asin)

    return tot


# ============================================================
# TEST SUITE (KEPT, SLIGHTLY ADAPTED)
# ============================================================


if __name__ == "__main__":
    num_cases = 20
    ms = (0, 1, 3, 10, 50, 100, 250, 500)
    seed = 1234
    tol = 5e-10

    random.seed(seed)
    np.random.seed(seed)

    worst = 0.0

    for k in range(num_cases):
        t1 = 0.0
        t2 = random.uniform(0.1, 20.0)
        t0 = random.uniform(0.0, 20.0)

        f0 = random.uniform(0.05, 0.45)
        Hmax = int(math.floor(10.0 / f0))
        H = random.randint(1, min(60, max(1, Hmax)))
        freqs = f0 * np.arange(1, H + 1)

        for f in freqs[:: max(1, H // 5)]:
            for m in ms:
                ref = quad_oracle_one_m_chunked(t1, t2, t0, m, f)
                val = levin_integral_single_jit(m, f, t1, t2, t0)
                err = abs(ref - complex(val))
                worst = max(worst, err)

                if err > tol:
                    raise AssertionError(f"FAIL case={k} m={m} f={f} err={err}")

        print(f"case {k + 1:02d}/{num_cases} ok")

    print("")
    print("ALL TESTS PASSED")
    print(f"worst abs err = {worst:.3e}")

# %%
ms = jnp.arange(500)
fs = 0.1 * jnp.arange(0, 150)


@jax.jit
def testf(ms, fs):
    return jax.vmap(
        lambda m: jax.vmap(
            lambda f: levin_integral_single_jit(m, f, t1, t2, t0)
        )(fs)
    )(ms)


testf(ms, fs)
testf(ms, fs)  # ~5 sec