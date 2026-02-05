= Process-induced surrogate modeling (PRISM)

This Appendix defines PRISM (process-induced surrogate modeling) as a variational learning procedure that maps a dataset of *independent variable length time series* into a *shared fixed dimensional latent space with uncertainty*, by learning a *global kernel-induced basis* and projecting each example into a *Gaussian posterior over basis amplitudes*. It then introduces *t-PRISM*, a robust variant that replaces Gaussian observation noise by a Student-t likelihood via local latent scale variables, yielding a local/global variational structure aligned with stochastic variational inference in the sense of Hoffman et al. (2013).

Background motivation around collapsing inducing-variable variational parameters follows the Titsias collapsed sparse GP construction.

== Problem setting and goal

=== Data
We have a dataset of $I$ independent examples

$
  cal(D) = {(t_i, y_i)}_(i=1)^I,
$

where example $i$ consists of a time grid

$
  t_i = (t_(i 1), dots, t_(i N_i)), quad N_i "varies with" i,
$

and observations
$
  y_i = (y_(i 1), dots, y_(i N_i)).
$
In practice these may be irregularly sampled, time warped, and stored as NaN padded arrays (the masked waveform model used in `collapsed_elbo_masked` in `svi.py`).

=== Modeling principle
Each example is modeled by an *independent* latent function:

$
  f_i(.) ~ cal(G P)(0, k_theta(., .)),
$

sharing the same kernel hyperparameters $theta$ across $i$, but with independent draws of $f_i$.

=== PRISM goal
PRISM is not only regression. The goal is a *projection operator*

$
  (t_i, y_i) mapsto q(epsilon_i) = cal(N)(mu_i, Sigma_i) in bb(R)^M
$

where:

- $M$ is fixed (the number of inducing features / basis functions),
- $epsilon_i$ are example-specific basis amplitudes,
- $(mu_i, Sigma_i)$ represent *uncertainty in latent space* for each example,
- downstream tasks can be performed in this latent space: density learning, clustering (mixture PPCA), compression, mixtures of cheap BLRs, and so on.

To make this feasible at scale, PRISM must avoid maintaining large per-example variational parameters.

== A shared basis from inducing points: motivation from the Nyström prism view

Choose inducing inputs (basis locations)

$
  Z = (z_1, dots, z_M), quad z_m in bb(R) "(time coordinate"),
$

Define:

$
  K_(ZZ) in bb(R)^(M times M), quad (K_(ZZ))_(m n) = k_theta(z_m, z_n),
$

and for example $i$ with times $t_i$:

$
  K_(Z i) in bb(R)^(M times N_i), quad (K_(Z i))_(m n) = k_theta(z_m, t_(i n)),
$

$
  K_(i i) in bb(R)^(N_i times N_i), quad (K_(i i))_(n n') = k_theta(t_(i n), t_(i n')),
$

A standard sparse or inducing approximation introduces inducing function values

$
  u_i = f_i(Z) in bb(R)^M.
$

Conditionally,

$
  p(f_i(t_i) mid u_i) = cal(N)(K_(i Z) K_(ZZ)^(-1) u_i, K_(i i) - Q_(i i)),
$

where

$
  Q_(i i) = K_(i Z) K_(ZZ)^(-1) K_(Z i).
$

Equivalently, in a whitened weight space view, define $L_(ZZ)$ such that

$
  K_(ZZ) = L_(ZZ) L_(ZZ)^top,
$

and define whitened amplitudes

$
  epsilon_i = L_(ZZ)^(-1) u_i.
$

Then the (Nyström) feature map for a time $t$ is

$
  psi_theta(t\; Z) = L_(ZZ)^(-1) k_theta(Z, t) in bb(R)^M,
$

and the low rank component of the GP can be written as

$
  f_i(t) approx psi_theta(t\; Z)^top epsilon_i.
$

This is exactly the “prism”: it maps each irregular time stamp $t_(i n)$ to a fixed dimensional feature vector $psi(t_(i n))$, so each variable length series becomes a design matrix

$
  Psi_i =
  mat(
    psi(t_(i 1))^top,
    dots.v,
    psi(t_(i N_i))^top
  )
  in bb(R)^(N_i times M).
$

== Gaussian PRISM: full model, ELBOs, and collapse

We begin with Gaussian noise PRISM, because it is the base case implemented in `collapsed_elbo_masked` and the collapse mathematics is clearest in this setting.

=== Likelihood and per-example joint model
Assume homoscedastic Gaussian noise with variance $sigma^2$:

$
  p(y_i mid f_i(t_i), sigma^2) = cal(N)(y_i mid f_i(t_i), sigma^2 I_(N_i)).
$

Introduce inducing variables $u_i = f_i(Z)$ and write the joint:

$
  p(y_i, f_i, u_i)
  = p(y_i mid f_i) p(f_i mid u_i) p(u_i),
$

with

$
  p(u_i) = cal(N)(0, K_(ZZ)).
$

=== Non-collapsed variational family (what we avoid)
A standard sparse variational GP would choose, for each example $i$,

$
  q_i (u_i) = cal(N)(m_i, S_i),
$

and define

$
  q_i (f_i) = integral p(f_i mid u_i) q_i (u_i) dif u_i.
$

A per-example ELBO is then:

$
  cal(L)_i^("SVGP")(m_i, S_i; Z, theta)
  =
  bb(E)_(q_i (f_i))[log p(y_i mid f_i)]
  -
  "KL"(q_i (u_i) || p(u_i)).
$

In PRISM, examples are independent GPs but share $(Z, theta)$. If we do not collapse, the natural construction becomes:

- global: $(Z, theta, sigma^2)$
- local: $(m_i, S_i)$ for each example.

This requires storing and optimizing $I$ Gaussian parameters:

- storage cost $O(I M^2)$ for the $S_i$ alone,
- update cost scales with $I$ and makes minibatching awkward because the local state is large.

This is the core computational motivation for collapse in PRISM.

== Collapsed variational family (Titsias style)
Titsias (2009) constructs an augmented variational family that uses the exact conditional $p(f_i mid u_i)$ and only approximates $p(u_i mid y_i)$, then shows that for Gaussian likelihood the optimal $q_i (u_i)$ can be eliminated analytically, yielding a bound that depends only on $(Z, theta, sigma^2)$.

We write the per-example collapsed bound directly in matrix form because it matches the code.

Define:

$
  A_i = L_(ZZ)^(-1) K_(Z i) / sigma in bb(R)^(M times N_i),
$

and

$
  B_i = I_M + A_i A_i^top in bb(R)^(M times M).
$

Then the Titsias collapsed ELBO for example $i$ is:

$
  cal(L)_i^("coll")(Z, theta, sigma^2)
  =
  log cal(N)(y_i mid 0, Q_(i i) + sigma^2 I)
  - 1/(2 sigma^2) "Tr"(K_(i i) - Q_(i i)).
$

We now express this entirely with $A_i, B_i$.

=== Step 1: log Gaussian term via Woodbury and determinant lemma
We have

$
  Q_(i i) = K_(i Z) K_(ZZ)^(-1) K_(Z i)
  =
  K_(i Z) L_(ZZ)^(-top) L_(ZZ)^(-1) K_(Z i).
$

Define $Psi_i = L_(ZZ)^(-1) K_(Z i)$ so that $Q_(i i) = Psi_i^top Psi_i$.

Then:

$
  Q_(i i) + sigma^2 I
  =
  sigma^2 (I + 1/sigma^2 Psi_i^top Psi_i).
$

But

$
  I + 1/sigma^2 Psi_i^top Psi_i
  =
  I + A_i^top A_i.
$

By the matrix determinant lemma:

$
  det(I + A_i^top A_i) = det(I + A_i A_i^top) = det(B_i).
$

Hence:

$
  log det(Q_(i i) + sigma^2 I)
  =
  N_i log sigma^2 + log det(B_i).
$

For the quadratic form, Woodbury gives:

$
  (Q_(i i) + sigma^2 I)^(-1)
  =
  1/sigma^2 (I - A_i^top B_i^(-1) A_i).
$

Therefore:

$
  y_i^top (Q_(i i) + sigma^2 I)^(-1) y_i
  =
  1/sigma^2 (
    y_i^top y_i - y_i^top A_i^top B_i^(-1) A_i y_i
  ).
$

Let

$
  v_i = A_i y_i in bb(R)^M.
$

Then

$
  y_i^top A_i^top B_i^(-1) A_i y_i = v_i^top B_i^(-1) v_i.
$

So:

$
  log cal(N)(y_i mid 0, Q_(i i) + sigma^2 I)
  =
  -1/2 (
    N_i log(2 pi)
    + N_i log sigma^2
    + log det(B_i)
    + 1/sigma^2 (y_i^top y_i - v_i^top B_i^(-1) v_i)
  ).
$

=== Step 2: trace correction term
We have:

$
  "Tr"(K_(i i) - Q_(i i)) = "Tr"(K_(i i)) - "Tr"(Q_(i i)).
$

But

$
  "Tr"(Q_(i i))
  =
  "Tr"(Psi_i^top Psi_i)
  =
  "Tr"(Psi_i Psi_i^top)
  =
  sum_(n=1)^(N_i) ||psi(t_(i n))||^2
  =
  ||Psi_i||_F^2.
$

In code, $Psi_i$ is `Psi` and this becomes the `Qxx_diag` / `trace(AAT)` style term in `collapsed_elbo_masked`.

Thus:

$
  -1/(2 sigma^2) "Tr"(K_(i i) - Q_(i i))
  =
  -1/(2 sigma^2) ("Tr"(K_(i i)) - "Tr"(Q_(i i))).
$

=== Final per-example collapsed ELBO (Gaussian PRISM)
Combining, the per-example objective is:

$
  cal(L)_i^("coll")(Z, theta, sigma^2)
  =
  -1/2 (
    N_i log(2 pi)
    + N_i log sigma^2
    + log det(B_i)
    + 1/sigma^2 (y_i^top y_i - v_i^top B_i^(-1) v_i)
  )
  - 1/(2 sigma^2) ("Tr"(K_(i i)) - "Tr"(Q_(i i)))
$

The full dataset ELBO is the sum:

$
  cal(L)^("PRISM")(Z, theta, sigma^2)
  =
  sum_(i=1)^I cal(L)_i^("coll")(Z, theta, sigma^2).
$

This is exactly what `batch_collapsed_elbo_masked` implements (with masking and stochastic scaling).

== Stochastic optimization over waveforms (collapsed SVI)

Minibatch a subset $cal(B)$ of waveforms. An unbiased estimate is:

$
  hat(cal(L))^("PRISM"))
  =
  I / (|cal(B)|)
  sum_(i in cal(B)) cal(L)_i^("coll").
$

Then use gradient ascent in $(Z, theta, sigma^2)$.

This is “collapsed SVI” in the pragmatic PRISM sense:

- SVI because we minibatch over independent examples.
- collapsed because $q_i (u_i)$ is not parameterized or stored.

== Projection to latent Gaussians (the “prism output”)

Once $(Z, theta, sigma^2)$ are trained, each example is projected to a Gaussian over amplitudes.

=== Posterior over whitened amplitudes in Gaussian PRISM
In the whitened BLR form,

$
  y_i = Psi_i epsilon_i + epsilon_i^"noise", quad
  epsilon_i^"noise" ~ cal(N)(0, sigma^2 I),
$

with prior

$
  epsilon_i ~ cal(N)(0, I).
$

Then the posterior is

$
  q(epsilon_i mid y_i) = cal(N)(mu_i, Sigma_i),
$

where

$
  Sigma_i = (I + sigma^(-2) Psi_i^top Psi_i)^(-1),
  quad
  mu_i = sigma^(-2) Sigma_i Psi_i^top y_i.
$

This is exactly the computation performed in
`infer_eps_posterior_single` (masked).

=== PRISM latent dataset
PRISM outputs the collection

$
  { (mu_i, Sigma_i) }_(i = 1)^I.
$

This is a *fixed-dimension probabilistic representation* of variable-length examples.
It supports:

- mixture PPCA and mixtures of factor analyzers operating directly on $(mu_i, Sigma_i)$,
- clustering in latent space while respecting posterior uncertainty,
- downstream surrogate construction, such as local linear models in weight space.

== Why collapse is essential for shared global basis learning

This section states the practical reasons explicitly, using the motivating points given earlier.

=== What non-collapsed VI would require in PRISM
If we used ordinary sparse VI with non-Gaussian likelihoods, we would typically need:

- per-example variational posterior $q_i (u_i) = cal(N)(m_i, S_i)$, or
- per-example variational posterior $q_i (epsilon_i) = cal(N)(mu_i, Sigma_i)$, equivalently.

Either way, the variational parameters are per-example.
Maintaining them across SVI minibatches requires storing and updating a list of Gaussians:

$
  { (m_i, S_i) }_(i = 1)^I
  quad "or"
  quad
  { (mu_i, Sigma_i) }_(i = 1)^I.
$

This is exactly what PRISM tries to avoid during training: it wants to reconstruct local posteriors on the fly from a shared global basis.

=== Why Gauss-Hermite or generic stochastic VI does not solve it
If we keep a non-Gaussian likelihood and approximate
$bb(E)_(q(f_i)) [ log p(y_i mid f_i) ]$
by Gauss-Hermite quadrature (or other generic stochastic VI techniques), we are still in the non-collapsed setting:

- we need $q_i (u_i)$ (or $q_i (epsilon_i)$) to define $q(f_i)$,
- therefore we still need per-example Gaussian variational parameters.

So the fundamental issue is not the quadrature method.
The issue is the need to represent per-example posterior uncertainty during training, unless collapse removes those degrees of freedom analytically.

=== What collapse changes
Collapse replaces `"store and optimize (m_i, S_i)"` by:

- evaluate $cal(L)_i^("coll"))(Z, theta, sigma^2)$ for each $i$,
- by solving linear algebra problems that depend only on $(Z, theta, sigma^2)$ and the observed $(t_i, y_i)$.

Thus global basis learning becomes feasible:

- $Z$ and $theta$ are learned from all waveforms without carrying a large local state,
- after learning, each waveform is projected to $(mu_i, Sigma_i)$.

== Algorithm summary and implementation notes

This section summarizes the full training loop, emphasizing the local–global separation and the points where the t-extension modifies the Gaussian PRISM code path.

Global parameters are $(Z, theta, sigma^2)$.
For each waveform $i$ with time samples $t_i$ and observations $y_i$:

- form $K_(Z Z)$ and its Cholesky $L_(Z Z)$,
- form $K_(Z i)$ and compute
  $
    A_i = L_(Z Z)^(-1) K_(Z i) / sigma,
    quad
    B_i = I + A_i A_i^top,
    quad
    v_i = A_i y_i,
  $
- evaluate $cal(L)_i^("coll")(Z, theta, sigma^2)$.

For a minibatch $cal(B)$, the stochastic objective is:

$
  hat(cal(L))
  =
  I / (|cal(B)|)
  sum_(i in cal(B)) cal(L)_i^("coll") (Z, theta, sigma^2).
$

Then update $(Z, theta, sigma^2)$ by gradient ascent.

== Summary

PRISM learns a shared, global kernel-induced basis from a collection of independent variable-length time series.
Training optimizes only global parameters $(Z, theta, sigma^2)$ by maximizing a collapsed objective,

$
  max_(Z, theta, sigma^2)
  sum_(i = 1)^I cal(L)_i^("coll") (Z, theta, sigma^2),
$

without storing or updating per-example Gaussian variational parameters.

After training, each example $i$ is projected independently to a Gaussian in a fixed-dimensional latent space,

$
  q_i (epsilon_i) = cal(N)(mu_i, Sigma_i),
$

providing a probabilistic representation with uncertainty that is suitable for downstream modeling.

=== Why collapse is the central design choice
Collapse is not an implementation detail but the defining mechanism that enables PRISM to scale.

By analytically eliminating $q_i (u_i)$ during training:

- global basis parameters are learned directly from all waveforms,
- memory usage does not grow with the number of examples,
- minibatching over independent time series becomes natural,
- local posteriors can be reconstructed on demand.

Without collapse, shared basis learning would require maintaining and synchronizing a large collection of per-example Gaussian states, defeating the purpose of a global surrogate representation.
