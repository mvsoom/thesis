"""Filon quadrature in 1D based on https://github.com/alexhroom/filon"""

# %%
import jax
import jax.numpy as jnp


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


def filon_tab_iexp(ftab, a, b, omega):
    """Compute ∫ f(x) exp(i omega x) dx over [a,b]

    ftab: Tabulated f values at uniform mesh over [a,b], where len(ftab) = n must be odd
    """
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

    return re + 1j * im


if __name__ == "__main__":
    import numpy as np
    from scipy.integrate import quad

    a, b = -2.0, 2.0
    omega = 15.0
    n = 129  # odd mesh size for Filon

    def base_fn(x):
        return np.exp(-0.5 * x * x)

    xs = np.linspace(a, b, n)
    ftab = jnp.asarray(base_fn(xs))

    filon_val = filon_tab_iexp(ftab, a, b, omega)

    def integrand(x):
        return base_fn(x) * np.exp(1j * omega * x)

    re = quad(lambda t: np.real(integrand(t)), a, b, limit=200)[0]
    im = quad(lambda t: np.imag(integrand(t)), a, b, limit=200)[0]
    quad_val = re + 1j * im

    err = abs(filon_val - quad_val)
    print(f"filon integral = {filon_val}")
    print(f"quad oracle    = {quad_val}")
    print(f"abs error      = {err:.3e}")
