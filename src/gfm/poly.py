import numpy as np


def repu(x, d):
    return np.heaviside(x, 1) * x**d


def build_phi(t, b, d):
    return repu(t[:, None] - b[None, :], d)


def build_q(b, tc, d):
    r = (tc - b) ** (d + 1) / float(d + 1)
    nr = np.linalg.norm(r)
    if nr == 0.0:
        nr = np.nan
    return r / nr


def process_b(b, H, tc):
    """Process a sampled/uniformly distributed/user-supplied `b`"""
    if type(b) == str:
        if b == "sample":
            b = np.random.uniform(0.0, tc, size=H)
        elif b == "uniform":
            b = np.linspace(0.0, tc, H)
        else:
            raise ValueError(f"Unknown b option: {b}")
    else:
        assert len(b) == H, f"Need H={H} knot locations"
    return b


def sample_poly(d, H, t, tc, sigma_a=1.0, b="uniform", closure=False):
    """Sample (u' | b) from the parametric piecewise polynomial model (4.3)"""
    b = process_b(b, H, tc)

    Phi = build_phi(t, b, d)

    if closure:
        q = build_q(b, tc, d)
        I = np.eye(H)
        Sigma = sigma_a**2 * (I - np.outer(q, q))  # rank H - 1

        # sample from a ~ N(0, Sigma) which is not full rank
        U, S, Vt = np.linalg.svd(Sigma)
        z = np.random.normal(0, 1.0, size=H)
        a = U @ np.diag(np.sqrt(S)) @ z
    else:
        a = np.random.normal(0, sigma_a, size=H)

    f = Phi @ a
    return f


def log_evidence_fixed_sigma(
    t, du, H, d, tc, sigma_a=1.0, b="uniform", sigma_noise=0.5, closure=False
):
    """Calculate p(u' | closure, b) given sigma_a, sigma_noise as in 4.3.1"""
    b = process_b(b, H, tc)

    phi = build_phi(t, b, d)

    if closure:
        q = build_q(b, tc, d)
        I = np.eye(H)
        P = I - np.outer(q, q)
    else:
        P = np.eye(H)

    K = sigma_a**2 * (phi @ P @ phi.T)
    K.flat[:: K.shape[0] + 1] += sigma_noise**2

    U, S, _ = np.linalg.svd(K)
    if np.any(S <= 0.0):
        return -1e300

    quad = du @ ((U * (1.0 / S)) @ (U.T @ du))
    ldet = np.sum(np.log(S))
    N = len(du)
    return -0.5 * (quad + ldet + N * np.log(2.0 * np.pi))


def map_a(t, du, b, d, tc, sigma_a, sigma_noise, closure=False):
    """Calculate the MAP estimate of the coefficients a given (b, u)"""
    H = len(b)
    phi = build_phi(t, b, d)

    if closure:
        q = build_q(b, tc, d)
        I = np.eye(H)
        P = I - np.outer(q, q)
    else:
        P = np.eye(H)
    Sigma_a = sigma_a**2 * P

    K = phi @ Sigma_a @ phi.T
    K.flat[:: K.shape[0] + 1] += sigma_noise**2

    U, S, Vt = np.linalg.svd(K)
    Sinv = 1.0 / S
    Kinv_u = (U * Sinv) @ (U.T @ du)
    return Sigma_a @ phi.T @ Kinv_u
