#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page

= t-PRISM
<algo:t-prism>

Glottal flow recordings are not clean.
Isolated artefacts — a creak, a microphone pop, a brief period of turbulent noise — can corrupt individual samples within an otherwise well-behaved waveform, and Gaussian noise, being light-tailed, is poorly equipped to handle them.
A single outlying sample inflates the residual quadratically and pulls the posterior over amplitudes away from the clean portion of the signal.

t-PRISM replaces the Gaussian likelihood with a Student-t, whose heavier tails allow the model to assign large residuals a modest probability rather than treating them as catastrophic evidence against the current fit.
The key is to do this without dismantling the collapsed structure of @algo:prism.
The approach is standard: represent the Student-t as a scale mixture of Gaussians via auxiliary precision variables $lambda_(i n)$, derive the ELBO over both the GP amplitudes and the precisions, and then observe that the GP part collapses exactly as before — now with a per-sample weighted noise covariance determined by the current $lambda_(i n)$ estimates.
The $lambda$ variables themselves have conjugate closed-form updates.
The result is an alternating scheme where each minibatch runs a small inner loop of cheap precision updates, then takes a gradient step on the global parameters, with no large per-waveform state persisting between minibatches.

== Student-t as a scale mixture

The Student-t likelihood with degrees of freedom $nu$ and scale $sigma^2$ is recovered from the Normal-Gamma augmentation
$
  lambda_(i n) ~ mono("Gamma")(nu\/2, thin nu\/2),
  quad
  y_(i n) | f_(i n), lambda_(i n) ~ mono("Normal")(f_(i n), thin sigma^2 \/ lambda_(i n)).
$
Marginalizing over $lambda_(i n)$ recovers $mono("StudentT")(y_(i n) | f_(i n), nu, sigma^2)$ exactly.
Stacking the per-sample precisions into $bm(Lambda)_i = "diag"(lambda_(i 1), dots, lambda_(i N_i))$, the augmented likelihood for waveform $i$ is Gaussian with a diagonal but heteroscedastic noise covariance:
$
  bm(y)_i | f_i, bm(Lambda)_i ~ mono("Normal")(f_i(bm(t)_i), thin sigma^2 bm(Lambda)_i^(-1)).
$ <eq:augmented-likelihood>
High-precision samples ($lambda_(i n)$ large) are trusted; low-precision ones ($lambda_(i n)$ small) are downweighted.
The prior on each $lambda_(i n)$ is $mono("Gamma")(nu\/2, nu\/2)$ with mean one, so in the absence of data the weights default to unity and the model reduces to PRISM.

== t-PRISM ELBO

We introduce a mean-field variational family over the latent precision variables,
$
  q_i(bm(Lambda)_i) = product_(n=1)^(N_i) q_(i n)(lambda_(i n)),
  quad
  q_(i n)(lambda_(i n)) = mono("Gamma")(alpha_(i n), beta_(i n)),
$ <eq:q-lambda>
while the GP amplitudes $bm(a)_i$ remain collapsed exactly as in @algo:prism.
The per-waveform ELBO, before collapsing, is
$
  cal(L)_i^t =
  bb(E)_(q_i)[log p(bm(y)_i | f_i, bm(Lambda)_i)]
  - D_"KL" (q_i(bm(u)_i) || p(bm(u)_i))
  - sum_(n=1)^(N_i) D_"KL" (q_(i n)(lambda_(i n)) || p(lambda_(i n))).
$ <eq:t-elbo-pre-collapse>
Define the expected precision and expected log-precision under $q_(i n)$:
$
  w_(i n) = bb(E)[lambda_(i n)] = alpha_(i n) \/ beta_(i n),
  quad
  ell_(i n) = bb(E)[log lambda_(i n)] = psi(alpha_(i n)) - log beta_(i n),
$ <eq:lambda-moments>
where $psi$ is the digamma function.
The expected log-likelihood expands as
$
  bb(E)[log p(bm(y)_i | f_i, bm(Lambda)_i)]
  = -1/2 sum_(n=1)^(N_i) (
    log(2pi) + log sigma^2
    - ell_(i n)
    + w_(i n) / sigma^2 ((y_(i n) - m_(i n))^2 + v_(i n))
  ),
$ <eq:expected-ll>
where $m_(i n) = bb(E)[f_i(t_(i n))]$ and $v_(i n) = "Var"(f_i(t_(i n)))$ are the marginal moments of the sparse GP, to be computed from the collapsed GP with the current weights.

=== Collapsing the GP conditional on $bm(W)_i$

Given fixed weights $bm(W)_i = "diag"(w_(i 1), dots, w_(i N_i))$, the augmented likelihood @eq:augmented-likelihood is Gaussian with noise covariance $sigma^2 bm(W)_i^(-1)$.
This is exactly the setting of @algo:prism with a heteroscedastic but still diagonal noise, and the Titsias collapse applies without modification.
Define the weighted design matrix and its Gram matrix,
$
  bm(A)_i^W = bm(W)_i^(1\/2) bm(Phi)_i \/ sigma in bb(R)^(N_i times M),
  quad quad
  bm(B)_i^W = bm(I)_M + (bm(A)_i^W)^top bm(A)_i^W in bb(R)^(M times M),
$ <eq:weighted-AB>
and the weighted sufficient statistic $bm(v)_i^W = (bm(A)_i^W)^top bm(W)_i^(1\/2) bm(y)_i in bb(R)^M$.#footnote[
  In the unweighted case $bm(W)_i = bm(I)$ these reduce to $bm(A)_i, bm(B)_i, bm(v)_i$ from @algo:prism exactly.
]
The collapsed Gaussian contribution to the ELBO is
$
  cal(L)_i^("coll")(bm(W)_i)
  = log mono("Normal")(bm(y)_i | bm(0), bm(Q)_i^W + sigma^2 bm(W)_i^(-1))
  - 1/(2sigma^2) "Tr"(bm(W)_i (bm(K)_(i i) - bm(Q)_(i i))),
$ <eq:weighted-collapsed>
where $bm(Q)_i^W = bm(Phi)_i (bm(B)_i^W)^(-1) bm(Phi)_i^top$ and the Woodbury and determinant-lemma identities from @algo:prism apply with $bm(A)_i^W$ in place of $bm(A)_i$.
The marginal moments needed in @eq:expected-ll are recovered as
$
  m_(i n) = phi(t_(i n))^top (bm(B)_i^W)^(-1) bm(v)_i^W,
  quad quad
  v_(i n) = k(t_(i n), t_(i n)) - phi(t_(i n))^top (bm(I) - (bm(B)_i^W)^(-1)) phi(t_(i n)).
$ <eq:marginal-moments>
Substituting everything, the per-waveform t-PRISM ELBO is
$
  cal(L)_i^t (bm(W)_i, bm(ell)_i)
  = cal(L)_i^("coll")(bm(W)_i)
  + 1/2 sum_(n=1)^(N_i) ell_(i n)
  - sum_(n=1)^(N_i) D_"KL" (q_(i n)(lambda_(i n)) || mono("Gamma")(nu\/2, nu\/2)).
$ <eq:t-elbo-final>
The middle term arises because the collapsed Gaussian uses $sigma^2 bm(W)_i^(-1)$ as the noise covariance while the full ELBO requires $bb(E)[log det(bm(Lambda)_i)] = sum_n ell_(i n)$; keeping it ensures the bound is tight with respect to the expected Gaussian normalization.
Dropping it would produce a subtly incorrect objective, which is where implementations often silently go wrong.

=== Local CAVI updates for the precision variables

Differentiating @eq:t-elbo-final with respect to $alpha_(i n)$ and $beta_(i n)$, or equivalently using the conjugacy of Gamma with the Normal-Gamma model, the optimal $q_(i n)$ is
$
  alpha_(i n)^* = (nu + 1) \/ 2,
  quad
  beta_(i n)^* = 1/2 (nu + ((y_(i n) - m_(i n))^2 + v_(i n)) \/ sigma^2).
$ <eq:cavi-update>
These are the coordinate ascent updates: the shape is constant and the rate inflates exactly when the current residual $(y_(i n) - m_(i n))^2 + v_(i n)$ is large relative to $sigma^2$.
Large residuals drive $beta_(i n)$ up, which drives $w_(i n) = alpha_(i n)^* \/ beta_(i n)^*$ down, which downweights that sample in the next collapse.
This is the robust mechanism: the model learns to distrust samples it cannot explain, without any manual flagging of outliers.

== Training

For a minibatch $cal(B)$ of waveforms, the stochastic objective is
$
  hat(cal(L))^("t-PRISM") = I \/ |cal(B)| sum_(i in cal(B)) cal(L)_i^t.
$
For each waveform $i$ in the minibatch, the updates are:

+ initialize $w_(i n) = 1$ for all $n$ (unit weights, equivalent to PRISM),
+ repeat for a small fixed number of steps:
  - run the weighted collapsed GP computation @eq:weighted-AB with current $bm(W)_i$, obtaining $(m_(i n), v_(i n))$ via @eq:marginal-moments,
  - update $(alpha_(i n)^*, beta_(i n)^*)$ via @eq:cavi-update and refresh $(w_(i n), ell_(i n))$ via @eq:lambda-moments,
+ evaluate $cal(L)_i^t$ @eq:t-elbo-final.

Then take a gradient step in the global parameters $(bm(Z), theta, sigma^2, nu)$.
The per-example Gamma parameters $(alpha_(i n), beta_(i n))$ are local to each minibatch step and discarded afterwards; no per-waveform state accumulates across iterations.

== Projection

After training, each waveform is projected to a Gaussian over whitened amplitudes exactly as in @algo:prism, but using the final converged weights $bm(W)_i$ from the inner loop.
The posterior is
$
  bm(a)_i | bm(y)_i ~ mono("Normal")(bm(m)_i, bm(S)_i),
  quad
  bm(S)_i = (bm(B)_i^W)^(-1),
  quad
  bm(m)_i = (bm(B)_i^W)^(-1) bm(v)_i^W,
$ <eq:t-prism-posterior>
which again reuses the quantities already computed during the inner loop.
The PRISM representation ${ (bm(m)_i, bm(S)_i) }_(i=1)^I$ is therefore the same fixed-dimensional probabilistic object as before, now obtained from a robust fit.

== Summary

t-PRISM extends PRISM by replacing the Gaussian observation model with a Student-t, implemented through per-sample Gamma precision variables $lambda_(i n)$.
Given these variables, the GP collapse of @algo:prism applies without change, with a weighted noise covariance $sigma^2 bm(W)_i^(-1)$ in place of $sigma^2 bm(I)$.
The $lambda_(i n)$ variables have closed-form conjugate updates @eq:cavi-update that automatically downweight samples with large residuals, requiring no outer iteration beyond a small fixed-count inner loop per minibatch.
Global parameters $(bm(Z), theta, sigma^2, nu)$ are then updated by a standard gradient step, and per-example states are discarded.

The PRISM-RFF basis of @chapter:prism-rff is compatible with t-PRISM without modification.
The only part of the algorithm that touches the definition of the basis is the computation of $bm(Phi)_i$ and the kernel diagonal $k(t_(i n), t_(i n))$; both are provided equally well by time-domain inducing points and by spectral features.
The two extensions — robustness and spectral basis — are therefore orthogonal, and all four combinations (PRISM, t-PRISM, PRISM-RFF, t-PRISM-RFF) are valid models sharing the same training loop.