# A Generative and Conditional Model for Phase Warping via Instantaneous Pitch

## 1. Physical starting point: instantaneous phase and pitch

Voiced speech can be described by a **monotone phase function**. Let

- $t$ be physical time (seconds or samples),
- $\tau(t)$ be the **cycle index / phase**, dimensionless,
- $f_0(t)$ be instantaneous pitch (Hz).

By definition,

$$
\frac{d\tau}{dt} = f_0(t)
$$

This says that phase advances at a rate given by instantaneous pitch.

Equivalently, inverting the relation,

$$
\frac{dt}{d\tau} = \frac{1}{f_0(t(\tau))} = T(\tau)
$$

where $T(\tau)$ is the **instantaneous period** (seconds per cycle).

This inversion is the key modeling move:
we describe **time as a function of phase**, not phase as a function of time.

---

## 2. Why we work with $t(\tau)$ instead of $\tau(t)$

GCIs (glottal closure instants) naturally live at **integer phase locations**:

$$
\tau = 0, 1, 2, \dots
$$

What is observed (or estimated) are the corresponding **times**:

$$
\hat t(\tau_k)
$$

Thus:

- $\tau$ is exact (dimensionless, noiseless),
- uncertainty lives entirely in time $t$.

This makes $t(\tau)$ the natural latent object.

---

## 3. Defining the generative model

### 3.1 Latent process: log-period GP

We model the **log-period** as a Gaussian process over phase:

$$
g(\tau) = \log T(\tau)
$$

$$
g(\cdot) \sim \mathcal{GP}\bigl(\mu,; \sigma^2 k_\ell(\tau,\tau')\bigr)
$$

where:

- $\mu$ sets the **typical pitch**
  ($\mu = \log 7\text{ ms}$ corresponds to about 140 Hz),
- $\sigma^2$ controls **marginal pitch variability**,
- $\ell$ is a **lengthscale in units of cycles**,
- $k_\ell$ is typically squared-exponential or Matérn.

Important:
There is **no white noise term** here. All irregularity comes from the smooth GP.

---

### 3.2 From log-period to time warp

Exponentiate and integrate:

$$
T(\tau) = \exp(g(\tau)) \quad [\text{seconds}]
$$

$$
t(\tau) = t_0 + \int_0^\tau T(u),du
$$

Properties:

- $t(\tau)$ is **strictly monotone**,
- $t'(\tau) = T(\tau) > 0$ automatically,
- units are correct: dimensionless $\tau$ integrates to time.

This is a **monotone GP via integration of a positive process**.

---

## 4. Interpretation in terms of physical quantities

### 4.1 Marginal pitch distribution

Since

$$
\log T(\tau) \sim \mathcal N(\mu, \sigma^2)
$$

we have

$$
\log f_0(\tau) = -\log T(\tau) \sim \mathcal N(-\mu, \sigma^2)
$$

Thus:

- marginal pitch is **lognormal**,
- $\mu$ and $\sigma$ are directly interpretable,
- bounds like 50–450 Hz arise naturally without hard constraints.

---

### 4.2 Period-to-period jitter

Classical (local) jitter is defined as

$$
\text{Jitter} =
\frac{\mathbb E|T_{k+1} - T_k|}{\mathbb E[T_k]}
$$

In this model:

- there is **no independent cycle noise**,
- jitter arises purely from **curvature of $t(\tau)$**,
- controlled entirely by $\ell$.

Large $\ell$
→ $g(\tau)$ almost linear
→ $t(\tau)$ almost affine
→ very low jitter.

Small $\ell$
→ rapid variation across cycles
→ higher jitter.

Thus $\ell$ has a direct physiological interpretation.

(An explicit analytic map $\ell \mapsto \text{jitter}$ can be derived or tabulated empirically; see later.)

---

## 5. Observation model and error calibration

### 5.1 Observation model

Given estimated GCIs:

$$
\hat t(\tau_i) = t(\tau_i) + b + \varepsilon_i
$$

where:

- $b$ is an **algorithm-dependent timing bias**,
- $\varepsilon_i$ is measurement noise.

Important identifiability point:

$$
t_0 \text{ and } b \text{ are not separately identifiable}
$$

Only their sum

$$
c = t_0 + b
$$

is observable without external information.

---

### 5.2 Using ground truth + estimates

If both ground-truth and estimated GCIs are available:

$$
\hat t^{(\text{est})}(\tau_i) = t^{(\text{gt})}(\tau_i) + b + \varepsilon_i
$$

then:

- $b$ **becomes identifiable**,
- noise statistics of $\varepsilon$ can be learned,
- detector bias and uncertainty are separated from physiology.

This calibration should be done **once**, upstream.

---

## 6. Two modes of operation

### 6.1 Pure generative prior

Used when no GCIs are observed.

Procedure:

1. Sample $g(\tau)$ from the GP prior,
2. Construct $t(\tau)$ by integration,
3. Fix the absolute time scale by:

   - sampling $t_0$, or
   - conditioning on one pseudo-point $t(\tau^*) = t^*$.

This yields a distribution over **plausible phase warps** with realistic pitch statistics.

---

### 6.2 Conditioning on GCI estimates

When GCIs are available:

- $\tau_i$ are fixed,
- $\hat t(\tau_i)$ constrain the warp.

Inference becomes:

$$
p(g, c \mid \hat t) \propto
p(\hat t \mid g, c),p(g)
$$

with

$$
p(\hat t \mid g, c) \sim \mathcal N\bigl(c\mathbf 1 + A\exp(g),; \Sigma_\varepsilon\bigr)
$$

where $A$ is a discretized integration operator.

Offsets are absorbed into $c$; physiology remains in $(\mu,\sigma,\ell)$.

---

## 7. Learning vs conditioning

### Learning (hyperparameters)

- Learn $\mu,\sigma,\ell$ from ground truth or well-calibrated data,
- Learn $b,\Sigma_\varepsilon$ from estimator calibration,
- Use approximate marginal likelihood (Laplace or variational).

### Conditioning (per segment)

- Infer $g(\tau)$ and $c$ given observed $\hat t(\tau_i)$,
- Sample posterior warps $t(\tau)$,
- No learning of physiology happens here.

---

## 8. Inversion: sampling in warped and uniform time

Once a sample $t^{(s)}(\tau)$ is drawn:

1. Choose a uniform time grid $t_j$ (e.g. 0–32 ms),
2. Invert the monotone map numerically to obtain
   $$
   \tau^{(s)}(t_j)
   $$
3. Evaluate downstream models in $\tau$-space,
4. Map results back to uniform $t$ if needed.

Because $t(\tau)$ is strictly increasing, inversion is stable and unique.

---

## 9. Summary of the conceptual picture

- Phase is fundamental; time is derived.
- Pitch is a smooth stochastic field over phase.
- Jitter is curvature, not noise.
- Monotonicity is enforced by construction.
- Absolute timing is a gauge, fixed by data or a pseudo-point.
- Estimator pathology is separated from physiology.
- The same model supports generation, conditioning, and uncertainty sampling.

At this point, the model is **physically coherent, statistically identifiable, and computationally workable**.
Anything further (explicit $\ell \leftrightarrow$ jitter mapping, kernel choice refinements, robust likelihoods) is refinement, not structural change.

## 10. Posterior sampling of phase warps (fast, non-MCMC)

Conditioning gives us a posterior over the latent function (g(\tau)) and hence over the warp (t(\tau)). For downstream inference we need **samples**, not just a MAP curve. At the same time, this step must be extremely fast (order milliseconds), since it sits _before_ heavier inference.

This section describes the sampling strategy that satisfies both requirements.

---

### 10.1 The problem restated

After discretization on a (\tau)-grid, we have:

- prior

  $$
  g \sim \mathcal N(\mu \mathbf 1, K)
  $$

- observation model
  $$
  y = c + A \exp(g) + \varepsilon,
  \quad
  \varepsilon \sim \mathcal N(0, \sigma_\varepsilon^2 I)
  $$

This model is **nonlinear** due to the exponential and the integral.

Exact posterior sampling is intractable, and MCMC is too slow.

---

### 10.2 Laplace approximation: the key idea

We approximate the posterior locally around the MAP solution.

Let:

- (g\_\*) be the MAP estimate,
- define (x = g - g\_\*).

Linearize the forward map at (g\_\*):

$$
A \exp(g) \approx
A \exp(g_*) + J x,
\quad
J = A \operatorname{diag}(\exp(g_*)).
$$

Define centered observations:

$$
\tilde y = y - c_* - A \exp(g_*).
$$

Then locally we have a **linear-Gaussian model**:

$$
\tilde y = J x + \varepsilon,
\quad
x \sim \mathcal N(0, K),
\quad
\varepsilon \sim \mathcal N(0, \sigma_\varepsilon^2 I).
$$

This is the classic Gaussian conditioning problem.

---

### 10.3 Conditioning-by-sampling (no large inverses)

Rather than explicitly forming the posterior covariance

$$
(K^{-1} + J^T \Sigma^{-1} J)^{-1},
$$

we use **conditioning by sampling**, which is exact for linear-Gaussian models.

Algorithm:

1. Sample from the prior:

   $$
   x_0 \sim \mathcal N(0, K)
   $$

2. Sample fake observation noise:

   $$
   \varepsilon_0 \sim \mathcal N(0, \sigma_\varepsilon^2 I)
   $$

3. Generate fake observation:

   $$
   y_0 = J x_0 + \varepsilon_0
   $$

4. Correct the sample:

   $$
   x = x_0 + K J^T (J K J^T + \sigma_\varepsilon^2 I)^{-1} (\tilde y - y_0)
   $$

5. Recover a sample of (g):
   $$
   g = g_* + x
   $$

This produces an **exact draw** from the Gaussian posterior of the linearized model.

---

### 10.4 Computational complexity

Let:

- (m) = number of (\tau)-grid points (typically 100–300),
- (n) = number of observations (GCIs), typically 1–5.

Costs:

- One-time per segment:

  - build (J): (O(nm))
  - form and Cholesky factorize
    $$
    S = J K J^T + \sigma_\varepsilon^2 I
    $$
    cost (O(n^3)), negligible

- Per sample:

  - draw (x_0) via prior Cholesky: (O(m^2))
  - one (m \times n) multiply and small linear solve

This is **orders of magnitude faster than MCMC** and easily supports (O(100)) samples per segment.

---

### 10.5 Practical speed considerations

The dominant cost is sampling from the GP prior via the Cholesky of (K).

To reach sub-10 ms latency:

- keep (m) modest (e.g. 128–256 is plenty for 32 ms),
- reuse (K) and its Cholesky across segments (hyperparameters fixed),
- optionally replace SE kernels with:

  - Matérn state-space models (exact, (O(m))),
  - or random Fourier features (approximate, (O(mD))).

The conditioning step itself scales only with the number of GCIs and is essentially free.

---

### 10.6 From samples of (g(\tau)) to samples of (t(\tau))

For each posterior sample (g^{(s)}):

1. Compute periods:

   $$
   T^{(s)}(\tau) = \exp(g^{(s)}(\tau))
   $$

2. Integrate:
   $$
   t^{(s)}(\tau) = c_* + \int_0^\tau T^{(s)}(u),du
   $$

Each sample is a **monotone phase warp** consistent with the observations and the prior.

---

### 10.7 Inversion to uniform time (for downstream models)

Given a sampled warp (t^{(s)}(\tau)):

- choose a uniform time grid (t_j),
- invert numerically to obtain (\tau^{(s)}(t_j)),
- evaluate downstream models in (\tau)-space,
- map back to time if needed.

Because monotonicity is guaranteed, inversion is stable and unique.

---

### 10.8 Summary of the sampling story

- Nonlinearity handled by local linearization.
- No MCMC, no tuning, no burn-in.
- Sampling cost dominated by prior GP draw.
- Supports real-time use.
- Produces physically valid, monotone phase warps.

This completes the model:
**a fast, generative, and conditionable distribution over phase warping functions.**
