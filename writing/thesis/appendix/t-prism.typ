= Robust process-induced surrogate modeling (t-PRISM)

We now introduce t-PRISM: a robust extension of PRISM that downweights outliers within a waveform while preserving the collapsed global-basis learning structure.

== Student-t likelihood as a scale mixture

We replace the Gaussian noise by a Student-t likelihood with degrees of freedom $nu$ and scale $sigma^2$:

$
  p(y_(i n) | f_i(t_(i n)), nu, sigma^2) = mono("StudentT")(y_(i n) | f_(i n), nu, sigma^2).
$

Use the standard Normal-Gamma augmentation:

$
  lambda_(i n) ~ mono("Gamma")(nu/2, nu/2),
  quad
  y_(i n) | f_(i n), lambda_(i n) ~ mono("Normal")(f_(i n), sigma^2 / lambda_(i n)).
$

Given $lambda_i = (lambda_(i 1), dots, lambda_(i N_i))$, the likelihood becomes Gaussian with diagonal noise:

$
  p(y_i | f_i, lambda_i, sigma^2)
  =
  mono("Normal")(y_i | f_i, sigma^2 Lambda_i^(-1)),
  quad
  Lambda_i = mono("diag")(lambda_(i 1), dots, lambda_(i N_i)).
$

== Full per-example joint model (augmented)

For each example $i$:

$
  p(y_i, f_i, u_i, lambda_i)
  =
  p(y_i | f_i, lambda_i)\, p(f_i | u_i)\, p(u_i)\, p(lambda_i).
$

== Mean-field variational family: local and global structure

t-PRISM uses the mean-field factorization:

$
  q_i (u_i, lambda_i) = q_i (u_i)\, q_i (lambda_i),
  quad
  q_i (lambda_i) = product_(n = 1)^(N_i) q_(i n)(lambda_(i n)).
$

Crucially:

- $q_i (lambda_i)$ are "local variables" per example (and per time point within example).
- global parameters are $(Z, theta, sigma^2, nu)$.
- we do not store $q_i (u_i)$; we will collapse it conditionally on $q_i (lambda_i)$.

This matches the Hoffman et al. (2013) local/global pattern: local latent variables optimized per minibatch, global parameters optimized by stochastic gradients of the ELBO.

== The correct ELBO for t-PRISM (per example)

The per-example ELBO is:

$
  cal(L)_i^t =
  bb(E)_(q_i (u_i) q_i (lambda_i))[log p(y_i | f_i, lambda_i)]
  +
  bb(E)_(q_i (u_i))[log p(u_i) - log q_i (u_i)]
  +
  bb(E)_(q_i (lambda_i))[log p(lambda_i) - log q_i (lambda_i)].
$

We now simplify each term carefully.

=== Term A: expected log likelihood given $q_i (u_i)$ and $q_i (lambda_i)$
Because $p(y_i | f_i, lambda_i)$ is Gaussian with diagonal precision $Lambda_i / sigma^2$:
$
  log p(y_i | f_i, lambda_i)
  =
  -1/2 (
    N_i log(2 pi) + N_i log sigma^2
    - log det(Lambda_i)
    + 1/sigma^2 (y_i - f_i)^top Lambda_i (y_i - f_i)
  ).
$
Take expectation under $q_i (u_i) q_i (lambda_i)$:
- define expected precisions and log precisions:
  $
    w_(i n) = bb(E)_(q_(i n))[lambda_(i n)],
    quad
    ell_(i n) = bb(E)_(q_(i n))[log lambda_(i n)].
  $
  Then $bb(E)[log det(Lambda_i)] = sum_n ell_(i n)$.
- for the quadratic term we need $bb(E)[(y_i - f_i)^top Lambda_i (y_i - f_i)]$.
  Since $Lambda_i$ is diagonal and independent of $f_i$ under mean-field:
  $
    bb(E)[(y_i - f_i)^top Lambda_i (y_i - f_i)]
    =
    sum_(n = 1)^(N_i) w_(i n)\, bb(E)[(y_(i n) - f_(i n))^2].
  $
  And
  $
    bb(E)[(y_(i n) - f_(i n))^2]
    =
    (y_(i n) - m_(i n))^2 + v_(i n),
  $
  where
  $
    m_(i n) = bb(E)[f_(i n)],
    quad
    v_(i n) = mono("Var")(f_(i n))
  $
  under the marginal induced by $q_i (u_i)$ and the sparse conditional.

So Term A becomes:

$
  bb(E)[log p(y_i | f_i, lambda_i)]
  =
  -1/2 (
    N_i log(2 pi) + N_i log sigma^2
    - sum_n ell_(i n)
    + 1/sigma^2 sum_n w_(i n) ((y_(i n) - m_(i n))^2 + v_(i n))
  ).
$

=== Term B: KL for inducing variables (collapsed later)
The inducing KL contribution is:

$
  bb(E)_(q_i (u_i))[log p(u_i) - log q_i (u_i)]
  =
  - mono("KL")(q_i (u_i) || p(u_i)).
$

=== Term C: KL for Gamma latent scales (local term)
Because $p(lambda_(i n)) = mono("Gamma")(nu/2, nu/2)$ and we will choose Gamma $q_(i n)$, this KL is closed form:

$
  bb(E)_(q_i (lambda_i))[log p(lambda_i) - log q_i (lambda_i)]
  =
  - sum_(n = 1)^(N_i) mono("KL")(q_(i n)(lambda_(i n)) || p(lambda_(i n))).
$

So the complete per-example ELBO is:

$
  cal(L)_i^t
  =
  -1/2 (
    N_i log(2 pi) + N_i log sigma^2
    - sum_n ell_(i n)
    + 1/sigma^2 sum_n w_(i n) ((y_(i n) - m_(i n))^2 + v_(i n))
  )
  - mono("KL")(q_i (u_i) || p(u_i))
  - sum_n mono("KL")(q_(i n)(lambda_(i n)) || p(lambda_(i n))).
$

This is the starting point for both the local CAVI updates and the global optimization.

== Collapsing $q_i (u_i)$ conditional on $q_i (lambda_i)$

Conditional on fixed weights $w_(i n)$ (equivalently a diagonal noise covariance $sigma^2 Lambda_i^(-1)$), the likelihood is Gaussian and the Titsias collapse applies exactly as in Section 3, but with per-point weights.

Define the weighted design:

$
  W_i = mono("diag")(w_(i 1), dots, w_(i N_i)),
  quad
  tilde(y)_i = W_i^(1/2) y_i,
  quad
  tilde(Psi)_i = W_i^(1/2) Psi_i.
$

Then the Gaussian collapsed term for that example is:

$
  cal(L)_i^("coll")(Z, theta, sigma^2; W_i)
  =
  log mono("Normal")(tilde(y)_i | 0, tilde(Q)_(i i) + sigma^2 I)
  - 1/(2 sigma^2) mono("Tr")(W_i (K_(i i) - Q_(i i))),
$

where $tilde(Q)_(i i) = tilde(Psi)_i tilde(Psi)_i^top$.

This matches the weighted linear algebra pattern already present in your robust sketch (weights multiply the columns of $K_(Z i)$, or equivalently rows of $Psi_i$) and matches how one must implement the Student-t augmentation in the collapsed setting.

After collapsing $q_i (u_i)$, the per-example ELBO becomes:

$
  cal(L)_i^t(Z, theta, sigma^2, nu; q_i (lambda_i))
  =
  cal(L)_i^("coll")(Z, theta, sigma^2; W_i)
  + 1/2 sum_(n = 1)^(N_i) ell_(i n)
  - sum_(n = 1)^(N_i) mono("KL")(q_(i n)(lambda_(i n)) || p(lambda_(i n))).
$

Explanation of the additional $1/2 sum ell_(i n)$ term:

- in $cal(L)_i^("coll")(*; W_i)$ we used the weighted Gaussian log likelihood with covariance $sigma^2 W_i^(-1)$, which contributes $+ 1/2 log det(W_i)$ inside the Gaussian normalization.
- the expected Gaussian normalization requires $bb(E)[log det(Lambda_i)] = sum ell_(i n)$.
  So we must explicitly keep $sum ell_(i n)$ in the ELBO, and then the remaining Gamma KL term ensures correctness.

This is exactly where robust implementations often accidentally drop terms if they only reweight residuals without including the latent-scale KL.

== Local CAVI updates for $q_(i n)(lambda_(i n))$ (closed form)

Choose

$
  q_(i n)(lambda_(i n)) = mono("Gamma")(alpha_(i n), beta_(i n))
$

(shape-rate parameterization).

To derive the coordinate update, write the terms in the ELBO that depend on $lambda_(i n)$.
From Term A and Term C, the relevant part is:

$
  bb(E)_(q_(i n))[log p(lambda_(i n))] - bb(E)_(q_(i n))[log q_(i n)(lambda_(i n))]
  + 1/2 bb(E)_(q_(i n))[log lambda_(i n)]
  - 1/(2 sigma^2) bb(E)_(q_(i n))[lambda_(i n)]\, bb(E)[(y_(i n) - f_(i n))^2],
$

where $bb(E)[(y_(i n) - f_(i n))^2] = (y_(i n) - m_(i n))^2 + v_(i n)$.

Using the Gamma prior $p(lambda_(i n)) = mono("Gamma")(nu/2, nu/2)$ and conjugacy, the optimal $q_(i n)$ is Gamma with:

$
  alpha_(i n) = (nu + 1)/2,
  quad
  beta_(i n) = 1/2 (nu + ((y_(i n) - m_(i n))^2 + v_(i n)) / sigma^2).
$

Then the moments needed in the ELBO are:

$
  w_(i n) = bb(E)[lambda_(i n)] = alpha_(i n) / beta_(i n),
  quad
  ell_(i n) = bb(E)[log lambda_(i n)] = psi(alpha_(i n)) - log beta_(i n).
$

These are exactly the closed-form local updates you were aiming for: no gradient steps, no Gauss-Hermite, just conjugate CAVI.

== Global stochastic optimization for t-PRISM

For a minibatch $cal(B)$ of waveforms, define:

$
  hat(cal(L))^("t-PRISM")
  =
  I / (|cal(B)|)
  sum_(i in cal(B))
  cal(L)_i^t (Z, theta, sigma^2, nu; q_i (lambda_i)).
$

Algorithmically, for each waveform $i$ in the minibatch:

1. initialize local Gamma parameters (or weights) for that waveform (e.g. $w_(i n) = 1$),
2. iterate a fixed number of CAVI steps:
  - compute the collapsed Gaussian quantities given $W_i$ (hence compute $m_(i n), v_(i n)$),
  - update $(alpha_(i n), beta_(i n))$ and moments $(w_(i n), ell_(i n))$,
3. evaluate $cal(L)_i^t$.

Then take a gradient step in the global parameters $(Z, theta, sigma^2, nu)$.

This is the local/global variational structure emphasized by Hoffman et al. (2013): local variational updates inside minibatches, global stochastic gradients on the ELBO.

== Algorithm summary and implementation notes

Global parameters are $(Z, theta, sigma^2, nu)$.
Local variables are the per-example Gamma factors:

$
  q_i (lambda_i) = product_(n = 1)^(N_i) q_(i n) (lambda_(i n)),
  quad
  q_(i n) (lambda_(i n)) = mono("Gamma")(alpha_(i n), beta_(i n)).
$

For each waveform $i$ in a minibatch:

1. initialize weights
  $
    w_(i n) = 1, quad ell_(i n) = 0,
  $
  equivalently $W_i = I$.

2. repeat for a small fixed number of CAVI iterations:

  (a) run the *weighted* collapsed Gaussian computations with
  $
    W_i = mono("diag")(w_(i 1), dots, w_(i N_i)),
  $
  producing the marginal moments
  $
    m_(i n) = bb(E)[f_(i n)],
    quad
    v_(i n) = mono("Var")(f_(i n)).
  $

  (b) update each local Gamma factor via

  $
    alpha_(i n) = (nu + 1)/2,
    quad
    beta_(i n) = 1/2 (nu + ((y_(i n) - m_(i n))^2 + v_(i n)) / sigma^2),
  $

  and refresh the required moments

  $
    w_(i n) = bb(E)[lambda_(i n)] = alpha_(i n) / beta_(i n),
    quad
    ell_(i n) = bb(E)[log lambda_(i n)] = psi(alpha_(i n)) - log beta_(i n).
  $

3. evaluate the per-example objective

$
  cal(L)_i^t (Z, theta, sigma^2, nu; q_i (lambda_i))
  =
  cal(L)_i^("coll") (Z, theta, sigma^2; W_i)
  + 1/2 sum_(n = 1)^(N_i) ell_(i n)
  - sum_(n = 1)^(N_i) mono("K L")(q_(i n) (lambda_(i n)) || mono("Gamma")(nu/2, nu/2)).
$

The minibatch estimator is then:

$
  hat(cal(L))^("t-PRISM")
  =
  I / (|cal(B)|)
  sum_(i in cal(B))
  cal(L)_i^t (Z, theta, sigma^2, nu; q_i (lambda_i)).
$

Then take a gradient step in the global parameters $(Z, theta, sigma^2, nu)$.

== Test-set scoring via importance-sampled marginal likelihood

We evaluate models using an approximation of the marginal likelihood

$
  p(y)
  =
  integral p(y | f, lambda)
  p(f)
  p(lambda)
  dif f
  dif lambda,
$

where $f$ denotes the latent GP function and $lambda$ are the per-sample scale variables arising from the Student-$t$ likelihood formulation. In our model,

- $f$ is represented via a collapsed sparse GP (PRISM),
- $lambda_n$ are latent precision variables such that
$
  y_n | f_n, lambda_n
  ~
  cal(N)(f_n, sigma^2 / lambda_n),
  quad
  lambda_n
  ~
  mono("Gamma")(nu / 2, nu / 2).
$

=== Collapsing the latent function

Conditioned on $lambda$, the model becomes Gaussian and the latent function $f$
(or equivalently the latent amplitudes $epsilon$)
can be analytically marginalized.
This yields a closed-form expression

$
  p(y | lambda)
  =
  cal(N)(y; 0, C(lambda)),
$

with covariance determined by the learned basis functions and the scaled noise term.
Thus only the integral over $lambda$ remains:

$
  p(y)
  =
  integral p(y | lambda)
  p(lambda)
  dif lambda.
$

=== Importance sampling using the variational posterior

This remaining integral is intractable.
During training we obtain a factorized variational posterior

$
  q(lambda)
  =
  product_n mono("Gamma")(lambda_n; alpha_n, beta_n)
$

via local variational updates.
We reuse this distribution as a proposal for importance sampling:

1. Draw samples
$
  lambda^(s)
  ~
  q(lambda),
  quad
  s = 1, dots, S.
$

2. Compute importance weights
$
  w^(s)
  =
  p(y | lambda^(s))
  p(lambda^(s))
  /
  q(lambda^(s)).
$

3. Estimate the log marginal likelihood using log-sum-exp:

$
  log hat(p)(y)
  =
  log
  (
    1 / S
    sum_(s = 1)^S w^(s)
  ).
$

This estimator can be viewed as an importance-corrected ELBO
(or IWELBO),
providing a tighter approximation to the true evidence than the variational bound alone.

=== Length normalization and null comparison

Waveforms have varying numbers of valid samples $n_("eff", i)$,
making the log evidence an *extensive* quantity:

$
  log p(y_i)
  ~
  cal(O)(n_("eff", i)).
$

To remove shared extensive terms, we compare against a null model consisting of iid Student-$t$ noise:

$
  Delta_i
  =
  log p_("model")(y_i)
  -
  log p_("null")(y_i).
$

However, even the difference remains extensive in sequence length.
Therefore we normalize per waveform by

$
  s_i
  =
  Delta_i / n_("eff", i),
$

yielding an average log-evidence improvement per datapoint.
This normalization effectively stratifies across the empirical distribution of waveform lengths,
preventing longer sequences from dominating the comparison.

The reported test score is the mean of $s_i$ across the test set,
along with its standard error.


=== Practical notes
- The only change from Gaussian PRISM to t-PRISM at the linear algebra level is the diagonal weight matrix $W_i$, which reweights residuals and effective noise per time sample.
- The local CAVI loop is deterministic and cheap: it requires only the current marginal moments $(m_(i n), v_(i n))$ and a few scalar Gamma updates.
- All per-example Gaussian state remains collapsed. Only the local Gamma parameters (or their moments $w_(i n), ell_(i n)$) exist transiently inside a minibatch.


=== t-PRISM
t-PRISM extends PRISM by replacing the Gaussian likelihood with a Student-t likelihood implemented through Gamma latent scale variables.

This yields a local–global variational structure:

- local variables $q_(i n) (lambda_(i n))$ handle outliers and heavy-tailed noise within each waveform,
- global parameters $(Z, theta, sigma^2, nu)$ are optimized by stochastic gradients of the ELBO,
- the inducing-variable posterior remains analytically collapsed.

The per-example objective takes the form

$
  cal(L)_i^t
  =
  cal(L)_i^("coll") (Z, theta, sigma^2; W_i)
  + 1/2 sum_(n = 1)^(N_i) ell_(i n)
  - sum_(n = 1)^(N_i) mono("K L")(q_(i n) (lambda_(i n)) || mono("Gamma")(nu/2, nu/2)),
$

preserving PRISM’s core advantage: scalable global basis learning with probabilistic latent projections, now robust to deviations from Gaussian noise.

=== Conceptual position
PRISM and t-PRISM sit between classical parametric models and fully nonparametric approaches.

They retain interpretability and computational control through a shared low-rank structure, while allowing uncertainty-aware, data-driven adaptation at the level of individual examples.
Collapse ensures that this balance is maintained even at scale, and the t-extension ensures that it remains stable under realistic, non-ideal data conditions.
