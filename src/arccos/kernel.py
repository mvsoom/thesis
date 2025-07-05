# %%
# Fourier coefficients for J(theta) extended to [-pi, pi]
# such that J(-theta) = J(theta) for theta in [0, pi].
# https://chatgpt.com/c/684c0f7e-2b98-8011-be48-76623b3c9762
import matplotlib.pyplot as plt
import numpy as np


def prepend_bias(x, bias=1.0):
    x = np.atleast_1d(x)
    return np.concatenate(([bias], x))


def J(theta, n):
    theta = np.abs(theta)  # Extend to [-pi, 0]
    if n == 0:
        return np.pi - theta
    elif n == 1:
        return np.sin(theta) + (np.pi - theta) * np.cos(theta)
    elif n == 2:
        return 3 * np.sin(theta) * np.cos(theta) + (np.pi - theta) * (
            1 + 2 * np.cos(theta) ** 2
        )
    else:
        raise ValueError("n must be 0, 1 or 2")


def k(x, y, n=0, L=np.eye(2)):
    xtilde = L @ prepend_bias(x)
    ytilde = L @ prepend_bias(y)

    xnorm = np.linalg.norm(xtilde)
    ynorm = np.linalg.norm(ytilde)

    ux = xtilde / xnorm
    uy = ytilde / ynorm

    # clip for numerical safety
    dot = np.clip(np.dot(ux, uy), -1.0, 1.0)
    theta = np.arccos(dot)

    c = (1 / np.pi) * (xnorm * ynorm) ** n * J(theta, n)
    return c


def kernel_matrix(x, n=0, L=np.eye(2)):
    return np.vectorize(k, excluded=["n", "L"])(
        x[:, None], x[None, :], n=n, L=L
    )


def kernel_matrix_fast(x, n=0, L=np.eye(2)):
    """
    Fast O(N²) kernel matrix with bias prepending and 2×2 linear map L.
    """
    x = np.asarray(x, dtype=float)
    L = np.asarray(L, dtype=float)

    if n not in (0, 1, 2):
        raise ValueError("n must be 0, 1 or 2")

    # build transformed vectors z_i = L · [1, x_i]^T
    V = np.column_stack((np.ones_like(x), x))  # (N, 2)
    Z = V @ L.T  # (N, 2)

    norms = np.linalg.norm(Z, axis=1)  # (N,)
    G = Z @ Z.T  # Gram matrix
    denom = np.outer(norms, norms)  # ||z_i||·||z_j||
    cosθ = np.clip(G / denom, -1.0, 1.0)  # numerical safety
    θ = np.arccos(cosθ)

    if n == 0:
        Jθ = np.pi - θ
    elif n == 1:
        Jθ = np.sin(θ) + (np.pi - θ) * cosθ
    else:  # n == 2
        Jθ = 3 * np.sin(θ) * cosθ + (np.pi - θ) * (1 + 2 * cosθ**2)

    return (1 / np.pi) * (denom**n) * Jθ


def dumbcholesky(K, nugget=1e-16):
    while True:
        try:
            L = np.linalg.cholesky(K + nugget * np.eye(K.shape[0]))
            return L, nugget
        except np.linalg.LinAlgError:
            # If it fails, increase the regularization term
            nugget *= 10


# sample points
x = np.linspace(-5, 5, 10)
K = kernel_matrix(x, n=1)

K_fast = kernel_matrix_fast(x, n=1)

np.allclose(K, K_fast)

# %%

x = np.linspace(-10, 10, 100)

for n in range(3):
    K = kernel_matrix_fast(x, n=n)
    plt.figure()
    plt.imshow(K, extent=(-5, 5, -5, 5), origin="lower", cmap="viridis")
    plt.title(f"Kernel Matrix (n={n})")
    plt.colorbar()
    plt.show()


# %%
from numpy.random import randn

n = 0

N = 1000
z = randn(N)
x = np.linspace(-1, 1, N)
dx = x[1] - x[0]

sigma0 = 0.1
sigmax = 10.0
rho = 0.1
Sigma = np.array(
    [[sigma0**2, sigmax * sigma0 * rho], [sigmax * sigma0 * rho, sigmax**2]]
)
LSigma = np.linalg.cholesky(Sigma)

K = kernel_matrix_fast(x, n=n, L=LSigma)
L, nugget = dumbcholesky(K)
nugget = 1e-16
print("Nugget:", nugget)

u = np.dot(L, z)
du = np.diff(u) / dx
xdu = x[:-1] + dx / 2
su = np.cumsum(u) * dx

# Plot integral, function, derivative
plt.figure()
plt.suptitle(f"arccos(n={n})")
plt.subplot(3, 1, 1)
plt.plot(x, su, label="Integral", color="red")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(x, u, label="Sampled Function", color="blue")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(xdu, du, label="Derivative", color="orange")
plt.xlabel("x")
plt.ylabel("du/dx")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import sympy as sp


def Jn_expr(n):
    theta = sp.symbols("theta")
    J = sp.pi - theta  # J0
    for k in range(n):
        J = (2 * k + 1) * sp.cos(theta) * J - sp.sin(theta) * sp.diff(J, theta)
    return sp.simplify(J)


def Jn(n, theta):
    Jn_symbolic = Jn_expr(n)
    return sp.lambdify(sp.symbols("theta"), Jn_symbolic, "numpy")(np.abs(theta))


# %%
# Check Fourier series of J0
def fourier_coefficients_J0(m):
    """Fourier coefficients for J0."""
    if m == 0:
        return np.pi
    else:
        # Zero for even m
        return 2 * (1 - (-1) ** m) / (np.pi * m**2)


# partial Fourier reconstruction
thetas = np.linspace(-np.pi, np.pi, 600, endpoint=False)


def J0_series(t, M=50):
    s = fourier_coefficients_J0(0) / 2
    for k in range(1, M + 1):
        s += fourier_coefficients_J0(k) * np.cos(k * t)
    return s


M = 30

plt.figure(figsize=(5, 3))
plt.plot(thetas, Jn(0, thetas), label="exact J0")
plt.plot(thetas, J0_series(thetas, M=M), "--", label=f"{M} cos‑series")
plt.legend()
plt.title("Fourier reconstruction of $J_0(\\theta)$")
plt.xlabel("$\\theta$")
plt.tight_layout()
plt.show()

# %%
# Check Fourier series of J1


def fourier_coefficients_J1(m):
    if m == 0:
        return 8 / np.pi
    elif m == 1:
        return np.pi / 2
    else:
        return 8 / (np.pi * (m**2 - 1) ** 2) if m % 2 == 0 else 0.0


def J1_series(t, M=50):
    s = fourier_coefficients_J1(0) / 2
    for k in range(1, M + 1):
        s += fourier_coefficients_J1(k) * np.cos(k * t)
    return s


M = 30

plt.figure(figsize=(5, 3))
plt.plot(thetas, Jn(1, thetas), label="exact J1")
plt.plot(thetas, J1_series(thetas, M=M), "--", label=f"{M} cos‑series")
plt.legend()
plt.title("Fourier reconstruction of $J_1(\\theta)$")
plt.xlabel("$\\theta$")
plt.tight_layout()
plt.show()

# %%
# J2
import matplotlib.pyplot as plt
import numpy as np

m, th = sp.symbols("m th", integer=True, positive=True)
pi = sp.pi
J2 = 3 * sp.sin(th) * sp.cos(th) + (pi - th) * (1 + 2 * sp.cos(th) ** 2)

# analytic cosine coefficient (even extension)
a_m = (2 / pi) * sp.integrate(J2 * sp.cos(m * th), (th, 0, pi))
a0 = (2 / pi) * sp.integrate(J2, (th, 0, pi))

print(sp.simplify(a0))  # 2*pi
print(sp.simplify(a_m))  # piecewise expression above
print("a2 =", sp.simplify(a_m.subs(m, 2)))  # pi/2


# numeric reconstruction check
def fourier_coefficients_J2(m):
    if m == 0:
        return 2 * np.pi
    if m == 2:
        return np.pi / 2
    if m % 2 == 0:
        return 0.0
    return 128 / (np.pi * m**2 * (m**2 - 4) ** 2)


theta = np.linspace(-np.pi, np.pi, 800, endpoint=False)
series = fourier_coefficients_J2(0) / 2 + sum(
    fourier_coefficients_J2(k) * np.cos(k * theta) for k in range(1, 120)
)
exact = 3 * np.sin(np.abs(theta)) * np.cos(np.abs(theta)) + (
    np.pi - np.abs(theta)
) * (1 + 2 * np.cos(np.abs(theta)) ** 2)

plt.plot(theta, exact, label="exact")
plt.plot(theta, series, "--", label="truncated Fourier")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# J3
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# ----- symbolic expression for J3 on (0,pi) ---------------------------
th = sp.symbols("th", real=True)
pi = sp.pi
J3 = (
    (pi - th) * (9 * sp.cos(th) + 6 * sp.cos(th) ** 3)
    + 15 * sp.sin(th)
    - 11 * sp.sin(th) ** 3
)

# analytic cosine coefficient a_m (even extension)
m = sp.symbols("m", integer=True, positive=True)
a_m_sym = sp.simplify((2 / pi) * sp.integrate(J3 * sp.cos(m * th), (th, 0, pi)))
a0 = sp.simplify((2 / pi) * sp.integrate(J3, (th, 0, pi)))

print(a_m_sym)  # piecewise expression for a_m
print(a0)

print("a0  =", a0)  # 256/(3*pi)
print("a1  =", a_m_sym.subs(m, 1).simplify())  # matches table
print("a2  =", a_m_sym.subs(m, 2).simplify())  # matches table
print("a3  =", a_m_sym.subs(m, 3).simplify())  # 3*pi/4
print("a4  =", a_m_sym.subs(m, 4).simplify())
print("a5  =", a_m_sym.subs(m, 5).simplify())  # 0


# ---- numerical reconstruction to check visually ----------------------
def fourier_coefficients_J3(m):
    if m == 0:
        return float(256 / (3 * np.pi))
    if m == 1:
        return 27 * np.pi / 4
    if m == 3:
        return 3 * np.pi / 4
    if m % 2 == 1:  # odd (not 3)
        return 0.0
    return 6912 / np.pi / ((m**2 - 1) ** 2 * (m**2 - 9) ** 2)


def J3_series(theta, M=10):
    s = fourier_coefficients_J3(0) / 2
    for k in range(1, M + 1):
        s += fourier_coefficients_J3(k) * np.cos(k * theta)
    return s


xx = np.linspace(-np.pi, np.pi, 800, endpoint=False)
plt.plot(xx, Jn(3, xx), label="exact")
plt.plot(xx, J3_series(xx), "--", label="10-term Fourier")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Eigenvalue decay:
# n = 0 => O(m^(-2)) [roughest]
# n = 1 => O(m^(-4))
# n = 2 => O(m^(-6))
# n = 3 => O(m^(-8)) [smoothest]
# Each order increases differentiability by one, and every
# extra derivative knocks the Fourier coefficients down by another m^(-2)
