# https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891-prism/c/6968c34f-250c-8325-8472-43c98510b7cf
"""Mixture of BLR surrogates on variable-length LF waveforms.

This implements an EM-like (soft) mixture model where each component is a Bayesian
linear regression (BLR) in a shared kernel-induced feature map.

Key design choices for speed:
  - Work in M-space (number of inducing points / features), never in T-space.
  - Batch variable-length waveforms by padding + mask.
  - Bucket by length to keep T_max bounded.

You can start with:
  - fixed kernel hyperparams (use your good prior run)
  - fixed inducing grid Z = linspace(0,1,M)
  - learn only mixture weights pi_k, noise sigma_k, and BLR prior parameters
    (mu_w[k], diag(var_w[k]))

Dependencies assume your codebase layout (prism.pack, utils.jax, surrogate.source).
"""
# %%

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax as ox

from prism.pack import NormalizedPACK
from surrogate import source
from utils.jax import safe_cholesky

K = 2
M = 32
nbuckets = 6
B = 64


ntrain = 1000


@dataclass
class Batch:
    tau: jnp.ndarray  # (B, T)
    y: jnp.ndarray  # (B, T)
    mask: jnp.ndarray  # (B, T) float64 in {0,1}


def make_length_buckets(
    lengths: Sequence[int], nbuckets: int
) -> List[jnp.ndarray]:
    """Return a list of index arrays, one per length bucket."""
    lengths = jnp.asarray(lengths)
    qs = jnp.linspace(0.0, 1.0, nbuckets + 1)
    edges = jnp.quantile(lengths, qs)
    buckets = []
    for b in range(nbuckets):
        lo = edges[b]
        hi = edges[b + 1]
        if b == nbuckets - 1:
            idx = jnp.where((lengths >= lo) & (lengths <= hi))[0]
        else:
            idx = jnp.where((lengths >= lo) & (lengths < hi))[0]
        if idx.size > 0:
            buckets.append(idx)
    return buckets


def sample_batch_from_bucket(
    lf_samples: Sequence[Dict],
    bucket_idx: jnp.ndarray,
    B: int,
    key: jax.Array,
) -> Batch:
    """Sample B waveforms from a bucket, pad to T_max within the batch."""
    n = int(bucket_idx.size)
    sel = jax.random.randint(key, (B,), 0, n)
    wav_idx = bucket_idx[sel]

    # Pull python-side arrays (variable length) then pad.
    tau_list = []
    y_list = []
    lens = []
    for i in list(map(int, list(wav_idx))):
        s = lf_samples[i]
        tau = jnp.asarray(s["tau"]).astype(jnp.float64)
        y = jnp.asarray(s["u"]).astype(jnp.float64)
        tau_list.append(tau)
        y_list.append(y)
        lens.append(int(tau.shape[0]))

    T = max(lens)

    tau_pad = jnp.zeros((B, T), dtype=jnp.float64)
    y_pad = jnp.zeros((B, T), dtype=jnp.float64)
    mask = jnp.zeros((B, T), dtype=jnp.float64)

    for b in range(B):
        t = tau_list[b]
        y = y_list[b]
        L = t.shape[0]
        tau_pad = tau_pad.at[b, :L].set(t)
        y_pad = y_pad.at[b, :L].set(y)
        mask = mask.at[b, :L].set(1.0)

    return Batch(tau=tau_pad, y=y_pad, mask=mask)


class FeatureMap:
    """Shared feature map Phi(tau) = Kzz^{-1} k(z, tau)."""

    def __init__(self, kernel, Z: jnp.ndarray):
        self.kernel = kernel
        self.Z = Z
        Kzz = kernel.gram(Z).to_dense()
        self.Lzz = safe_cholesky(Kzz)

    def phi_pad(self, tau_pad: jnp.ndarray) -> jnp.ndarray:
        """tau_pad: (B,T) -> Phi: (B,T,M)"""
        B, T = tau_pad.shape
        x = tau_pad.reshape((-1, 1))
        Kxz = self.kernel.cross_covariance(x, self.Z)  # (B*T, M)
        # Solve Kzz * a = Kxz^T for a, returning (M, B*T)
        A = jax.scipy.linalg.cho_solve((self.Lzz, True), Kxz.T)
        Phi = A.T.reshape((B, T, -1))
        return Phi


# %%
lf_samples = source.get_lf_samples()[:ntrain]


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


for lf_sample in lf_samples:
    lf_sample["tau"] = warp_time(lf_sample["t"], lf_sample["p"]["T0"])


lengths = [int(jnp.asarray(s["tau"]).shape[0]) for s in lf_samples]
buckets = make_length_buckets(lengths, nbuckets)

# %%
# -------------------------
# Kernel + inducing grid
# -------------------------
base = NormalizedPACK(d=1)
kernel = base  # tau-only kernel

Z = jnp.linspace(0.0, 1.0, M).reshape(-1, 1)

# For now, share the same basis across regimes for stability.
fmap = FeatureMap(kernel, Z)
fmap_list = [fmap for _ in range(K)]

# %%


def blr_stats(Phi: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray):
    """Compute sufficient stats in M-space."""
    Phi = Phi * mask[..., None]
    y = y * mask
    G = jnp.einsum("btm,btn->bmn", Phi, Phi)  # (B,M,M)
    p = jnp.einsum("btm,bt->bm", Phi, y)  # (B,M)
    s = jnp.einsum("bt,bt->b", y, y)  # (B,)
    Ti = jnp.sum(mask, axis=1)  # (B,)
    return G, p, s, Ti


def solve_chol_batched(L: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve A x = rhs given cholesky L of A for batched (B,M,M).

    rhs: (B,M)
    returns x: (B,M)
    """
    y = jax.scipy.linalg.solve_triangular(L, rhs, lower=True)
    x = jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L, -1, -2), y, lower=False
    )
    return x


def blr_logp_from_stats(
    G: jnp.ndarray,
    p: jnp.ndarray,
    s: jnp.ndarray,
    Ti: jnp.ndarray,
    mu_w: jnp.ndarray,
    log_var_w: jnp.ndarray,
    log_sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Log p(y | k) for each item in a batch.

    Prior: w ~ N(mu_w, diag(exp(log_var_w)))
    Noise: eps ~ N(0, sigma^2 I)

    All computations are in M-space.
    """
    sigma2 = jnp.exp(2.0 * log_sigma)
    inv_var = jnp.exp(-log_var_w)  # (M,)

    # A = Sigma^{-1} + (1/sigma^2) * G
    A = (1.0 / sigma2) * G + jnp.diag(inv_var)[None, :, :]

    # b = Sigma^{-1} mu + (1/sigma^2) p
    b = (inv_var * mu_w)[None, :] + (1.0 / sigma2) * p

    L = jnp.linalg.cholesky(A)

    Ainv_b = solve_chol_batched(L, b)
    Ainv_p = solve_chol_batched(L, p)

    bAinvb = jnp.einsum("bm,bm->b", b, Ainv_b)
    pAinvp = jnp.einsum("bm,bm->b", p, Ainv_p)

    logdetA = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=1
    )
    muSigInvmu = jnp.sum(mu_w * (inv_var * mu_w))

    # Mask handling:
    # - s and p only include valid entries due to masking.
    # - logdetA depends on G, also masked.
    # - The remaining T-dependent normalizer uses Ti.

    ll = -0.5 * (
        (s / sigma2)
        - (pAinvp / (sigma2 * sigma2))
        + muSigInvmu
        - bAinvb
        + logdetA
        + Ti * (jnp.log(2.0 * jnp.pi) + jnp.log(sigma2))
    )

    return ll


def e_step(
    fmap_list: List[FeatureMap],
    params: Dict[str, jnp.ndarray],
    batch: Batch,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ll (B,K) and responsibilities r (B,K)."""
    K = len(fmap_list)

    ll_ks = []
    for k in range(K):
        Phi = fmap_list[k].phi_pad(batch.tau)
        G, p, s, Ti = blr_stats(Phi, batch.y, batch.mask)
        ll = blr_logp_from_stats(
            G,
            p,
            s,
            Ti,
            params["mu_w"][k],
            params["log_var_w"][k],
            params["log_sigma"][k],
        )
        ll_ks.append(ll)

    ll = jnp.stack(ll_ks, axis=1)
    logits = ll + params["log_pi"][None, :]
    r = jax.nn.softmax(logits, axis=1)
    return ll, r


def loss_fn(
    fmap_list: List[FeatureMap],
    params: Dict[str, jnp.ndarray],
    batch: Batch,
) -> jnp.ndarray:
    ll, r = e_step(fmap_list, params, batch)
    log_pi = params["log_pi"][None, :]
    H = -jnp.sum(r * jnp.log(r + 1e-12), axis=1)
    Q = jnp.sum(r * (ll + log_pi), axis=1) + H
    return -jnp.mean(Q)


def init_params(key: jax.Array, K: int, M: int) -> Dict[str, jnp.ndarray]:
    def jitter(k, shape):
        return (1e-2) * jax.random.normal(k, shape, dtype=jnp.float64)

    def zeros(shape):
        return jnp.zeros(shape, dtype=jnp.float64)

    # Break symmetries with small random jitter
    mu_w = jitter(key, (K, M))

    log_var_w = zeros((K, M))
    log_sigma = zeros((K,))
    log_pi = zeros((K,)) - jnp.log(K)

    return {
        "mu_w": mu_w,
        "log_var_w": log_var_w,
        "log_sigma": log_sigma,
        "log_pi": log_pi,
    }


# @jax.jit
def train_step(
    fmap_list: List[FeatureMap],
    params: Dict[str, jnp.ndarray],
    opt_state,
    batch: Batch,
    opt,
    pi_ema: float,
):
    def loss(params):
        return loss_fn(fmap_list, params, batch)

    val, grads = jax.value_and_grad(loss)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = ox.apply_updates(params, updates)

    # Analytic pi update (EMA for stability)
    _, r = e_step(fmap_list, params, batch)
    r_bar = jnp.mean(r, axis=0) + 1e-12
    new_log_pi = jnp.log(r_bar) - jnp.log(jnp.sum(r_bar))
    params = dict(params)
    params["log_pi"] = (1.0 - pi_ema) * params["log_pi"] + pi_ema * new_log_pi

    return params, opt_state, val, r


def evaluate_nll(
    fmap_list: List[FeatureMap],
    params: Dict[str, jnp.ndarray],
    batch: Batch,
) -> jnp.ndarray:
    """Waveform-level predictive NLL for a batch.

    Uses logsumexp over mixture components.
    """
    ll, _ = e_step(fmap_list, params, batch)
    log_pi = params["log_pi"][None, :]
    lse = jax.scipy.special.logsumexp(ll + log_pi, axis=1)
    return -jnp.mean(lse)


def compute_ESS_k(r, eps=1e-12):
    """
    r: (B, K) responsibilities for a batch
    returns ess: (K,)
    """
    rk_sum = jnp.sum(r, axis=0)  # (K,)
    rk_sq_sum = jnp.sum(r * r, axis=0)  # (K,)
    ess = (rk_sum * rk_sum) / (rk_sq_sum + eps)
    return ess


key = jax.random.key(0)  # vk()
params = init_params(key, K, M)

steps = 100
lr = 1e-3
pi_ema = 0.1

opt = ox.adam(lr)
opt_state = opt.init(params)

for step in range(steps):
    key, k1, k2 = jax.random.split(key, 3)

    # Choose a bucket uniformly among non-empty buckets.
    bidx = jax.random.randint(k1, (), 0, len(buckets))
    bucket_idx = buckets[int(bidx)]

    batch = sample_batch_from_bucket(lf_samples, bucket_idx, B, k2)

    params, opt_state, loss, r = train_step(
        fmap_list, params, opt_state, batch, opt, pi_ema
    )

    if step % 1 == 0:
        pi = jax.nn.softmax(params["log_pi"]).tolist()
        sig = jnp.exp(params["log_sigma"]).tolist()
        ESS_k = compute_ESS_k(r)
        print(
            f"step {step:6d}  loss {float(loss):.3f}  pi {pi}  sigma {sig}  ESS_k {ESS_k}"
        )

# %%

# -------------------------
# Quick sanity eval on a fresh batch
# -------------------------
key, k1, k2 = jax.random.split(key, 3)
bidx = jax.random.randint(k1, (), 0, len(buckets))
batch = sample_batch_from_bucket(lf_samples, buckets[int(bidx)], B, k2)
nll = evaluate_nll(fmap_list, params, batch)
print("final batch nll:", float(nll))
