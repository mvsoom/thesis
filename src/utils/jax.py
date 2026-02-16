import time

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import checkify
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

_KEY = jax.random.PRNGKey(int(time.time_ns()) % (2**32 - 1))


def vk(key=None):
    """Volatile key for testing purposes -- don't use in production"""
    if key is None:
        global _KEY
        _KEY, k = jax.random.split(_KEY)
        return k
    else:
        return jax.random.PRNGKey(key)


def jnp_default(a):
    return struct.field(default_factory=lambda: a)


def static_constant(default_value):
    """Make JAX specialize on this value"""
    return struct.field(
        pytree_node=False,
        default_factory=lambda: default_value,
    )


def maybe32(x):
    """Convert to x32 if not jax_enable_x64"""
    try:
        dtype = jax.dtypes.canonicalize_dtype(x.dtype)
    except AttributeError:
        dtype = jax.dtypes.canonicalize_dtype(type(x))
    return jnp.asarray(x, dtype=dtype)


def _path_to_str(path):
    parts = []
    for k in path:
        if isinstance(k, GetAttrKey):
            parts.append(str(k.name))
        elif isinstance(k, SequenceKey):
            parts.append(f"[{k.idx}]")
        elif isinstance(k, DictKey):
            parts.append(f"['{k.key}']")
        else:
            parts.append(str(k))
    s = ""
    for p in parts:
        if p.startswith("[") or p.startswith("['"):
            s += p
        else:
            s = p if s == "" else s + "." + p
    return s if s else "<root>"


def _leaf_name_from_path(path):
    if not path:
        return "<root>"
    k = path[-1]
    if isinstance(k, GetAttrKey):
        return str(k.name)
    if isinstance(k, SequenceKey):
        return f"[{k.idx}]"
    if isinstance(k, DictKey):
        return str(k.key)
    return str(k)


def check_precisions(tree, return_issues=False):
    issues = []
    paths = []
    leaves = []

    # Collect (path, leaf) pairs in a version-robust way
    def _collect(path, x):
        paths.append(path)
        leaves.append(x)
        return None

    jax.tree_util.tree_map_with_path(_collect, tree)

    for i, (path, x) in enumerate(zip(paths, leaves)):
        leaf_path = _path_to_str(path)
        leaf_name = _leaf_name_from_path(path)

        def report(kind, dtype, shape, value):
            issues.append(
                {
                    "index": i,
                    "path": leaf_path,
                    "name": leaf_name,
                    "kind": kind,
                    "dtype": str(dtype),
                    "shape": tuple(shape) if hasattr(shape, "__iter__") else (),
                    "value": value,
                }
            )

        if isinstance(x, (jnp.ndarray, np.ndarray)):
            dt = x.dtype
            if jnp.issubdtype(dt, jnp.floating):
                if jnp.finfo(dt).bits > 32:
                    report(
                        "float>32",
                        dt,
                        x.shape,
                        x if x.shape == () else "<array>",
                    )
            elif jnp.issubdtype(dt, jnp.integer):
                if np.iinfo(dt).bits > 32:
                    report(
                        "int>32", dt, x.shape, x if x.shape == () else "<array>"
                    )
        elif isinstance(x, float):
            report("python-float", "float", (), x)
        elif isinstance(x, int):
            report("python-int", "int", (), x)

    if not issues:
        print("All numeric leaves are <=32-bit.")
    else:
        print("Found potential width issues:")
        for it in issues:
            print(
                f"  Leaf #{it['index']}: path={it['path']} name={it['name']} "
                f"dtype={it['dtype']} shape={it['shape']} kind={it['kind']} val={it['value']}"
            )

    if return_issues:
        return issues


def safe_cholesky(A, jitter=1e-6):
    A = 0.5 * (A + A.T)
    d = jnp.diag(A)
    scale = jnp.sqrt(jnp.mean(d * d) + 1e-16)
    nugget = jitter * scale
    return jnp.linalg.cholesky(A + nugget * jnp.eye(A.shape[-1], dtype=A.dtype))


def kl_diag_gauss(mu, var, prior_mu, prior_var):
    """
    KL( N(mu,var) || N(prior_mu, prior_var) ) for diagonal covariances, summed over all entries.
    """
    mu = jnp.asarray(mu)
    var = jnp.asarray(var)
    prior_mu = jnp.asarray(prior_mu)
    prior_var = jnp.asarray(prior_var)

    var = jnp.clip(var, a_min=1e-18)
    prior_var = jnp.clip(prior_var, a_min=1e-18)

    t1 = jnp.log(prior_var) - jnp.log(var)
    t2 = (var + (mu - prior_mu) ** 2) / prior_var
    return 0.5 * jnp.sum(t1 - 1.0 + t2)


def pca_reduce(X: jnp.ndarray, latent_dim: int) -> jnp.ndarray:
    """
    Linearly reduce the dimensionality of the input points `X` to `latent_dim` dimensions.

    Matches the original PCA projection: covariance of centered data,
    eigen-decomposition, and projection onto the top components.
    """
    if latent_dim > X.shape[1]:
        raise ValueError("Cannot have more latent dimensions than observed")
    X = jnp.asarray(X)
    X_mean = jnp.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean
    # Divide by N (not N-1) to match the original covariance definition.
    X_cov = (X_centered.T @ X_centered) / X.shape[0]
    _, evecs = jnp.linalg.eigh(X_cov)
    W = evecs[:, -latent_dim:]
    return X_centered @ W


def nocheck(f):
    checked = checkify.checkify(f, errors=checkify.user_checks)

    def wrapped(*args, **kwargs):
        err, out = checked(*args, **kwargs)
        # DO NOT call err.throw()
        return out

    return wrapped


def symmetrize(A):
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def resolve_gpjax_kernel(kernel):
    if kernel == "matern:12":
        return gpx.kernels.Matern12
    elif kernel == "matern:32":
        return gpx.kernels.Matern32
    elif kernel == "matern:52":
        return gpx.kernels.Matern52
    elif kernel == "rbf":
        return gpx.kernels.RBF
    elif kernel == "rationalquadratic":
        return gpx.kernels.RationalQuadratic
    else:
        raise ValueError(f"Unknown kernel: {kernel}")