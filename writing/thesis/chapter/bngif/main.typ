#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/equate:0.3.2": equate, share-align
#show: equate.with()

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

= Bayesian nonparametric glottal inverse filtering
<chapter:bngif>

== Recap of all the ingredients

- IKLP
- AR prior
- Quasiperiodic arc cosine kernel
  - Learnt from synthetic Paule data
- Latent variable models for refining

== Learning priors from synthetic data

=== Paule

=== Finding high quality voiced groups

We use t-PRISM for that, and our LBGID algorithm.
The main conceptual idea here is that we use _an explicit pitch track model_ to do the extraction rather than ad hoc heuristics.

/*
IMPORTANT

Scoring on test set when learning pitch track models is written down in writing/thesis/scrap/test-set-scoring.typ
*/

Tricks we use:
- Normalize data to (0,1): this means we freeze mean to 0 *and* kernel variance to 1.
- Lengthscale in ballpark order: 10. This is simply to cut number of iterations, the model does find lengthscales ~ 15 if initialized from 1, but takes much longer.
- Inverse-ECDF initialization of inducing points. This is important because trajectory lengths are long tailed, and we need to place predictive power where it matters. In 1D we can just sample from the empirical distribution, with interpolation.
- Set $nu$ based on event statistics. Optimizing $nu$ is possible, but more stable optimization. Results depend only very slightly on exact value of $nu$ as long as $<10$.

For the rest standard Adam optimizer with standard settings.

==== Heuristic for setting $nu$
We set $nu$ using a simple tail-probability heuristic derived from the expected frequency of spike events. After normalization, the GP prior has unit variance and the learned observation noise standard deviation is approximately $sigma approx 0.15$. Typical spikes have amplitude around $5$, corresponding to residuals of size $r approx 5 / 0.15 approx 30-35$ noise standard deviations. We interpret spikes as events that should lie in the heavy tails of the Student-t likelihood rather than being explained by the Gaussian core. Therefore we choose $nu$ such that the tail probability of a Student-t distribution satisfies $P(|X| > r) approx p_"spike"$, where $p_"spike"$ is the empirical spike rate (here about $1%$). Using the large-$r$ asymptotic tail behaviour $P(|X| > r) approx C * r^(-nu)$ and ignoring constants gives the practical rule $nu approx - log(p_"spike") / log(r)$. For $r approx 35$ and $p_"spike" approx 0.01$, this yields $nu approx 1-2$, which corresponds to a Cauchy-like heavy-tailed likelihood. This choice ensures that extreme residuals are downweighted rather than forcing the GP to explain them through excessive curvature.

=== Time alignment

==== Instantaneous phase and pitch
Voiced speech can be described by a monotone _phase function_. Let

- $t$ be physical time (in seconds),
- $tau(t)$ be the cycle index or phase, dimensionless,
- $f_0(t)$ be the instantaneous pitch (Hz).

By definition,
$
  (dif tau) / (dif t) = f_0(t)
$ <eq:phase-function>
This says that phase advances at a rate given by instantaneous pitch.
The key is now that we will describe time as a function of phase, not the other way around; this is known as _time warping_: the local pitch warps time.

In the simplest case, that of constant pitch, we have $f_0(t) := 1\/T$ such that, integrating @eq:phase-function, we have
$
  tau(t) = (t-t_0)/T
$
where $t_0 in bb(R)$ is an arbitrary offset. Thus
$
  t(tau) = t_0 + T tau.
$
This is simply rescaling and re-offsetting when going from $tau$ space from $t$, as people would naievely do.

The most general solution of @eq:phase-function however, is:
$
  (dif t)/(dif tau) = 1/(f_0(t(tau))) := T(tau),
$
such that
$
  t(tau) = t_0 + integral_0^tau T(u) dif u.
$
This defines physical time $t$ as time-warped by local curvature $T(tau)$, which is the instantaneous period given by the reciprocal of the instantaneous fundamental frequency $f_0(tau) = 1\/T(tau)$.
These can be indexed either in phase $tau$ or physical time $t$.

/*
Phase warping necessitates having a model with continuous time like GPs.
This is not as evident in straight signal-processing approaches.

Note: this construction is like a GP with a single hidden layer.
*/

=== Period-to-period jitter as a function of ell

// https://chatgpt.com/c/695cfddb-4798-832f-b07c-0ef09fbc37fc

/* TODO: should probably be an appendix */

Classical local jitter is defined as the expected absolute change in consecutive periods, normalized by the mean period,
$
  "jitter" := expval(|T_(k+1) - T_k|) / expval(T_k).
$

In the present framework, jitter is not introduced as an independent noise source.
Instead, it emerges deterministically from a smooth stochastic model of phase warping.
All variability originates from curvature of the time warp $t(tau)$, induced by a Gaussian process prior on the log-period
$
  g(tau) = log T(tau).
$

Sampling the process at integer phases $tau = k$ yields a stationary Gaussian sequence
$
  g_k := g(k),
$
with mean $mu$, marginal variance $sigma^2$, and lag-one correlation
$
  rho(ell) := k_ell(1) / k_ell(0),
$
which depends only on the kernel family and the lengthscale $ell$ measured in cycles.

For the kernels considered here, the lag-one correlation is given by#footnote[
  Matérn 1/2: $rho(ell) = exp(-1 / ell)$.
  Matérn 3/2: $rho(ell) = (1 + sqrt(3)/ell) exp(-sqrt(3)/ell)$.
  Matérn 5/2: $rho(ell) =
  (1 + sqrt(5)/ell + 5 / (3 ell^2)) exp(-sqrt(5)/ell)$.
  Squared exponential: $rho(ell) = exp(-1 / (2 ell^2))$.
]

The observed periods are lognormal variables,
$
  T_k = exp(g_k),
$
and the pair $(T_k, T_(k+1))$ is bivariate lognormal.
Because $g_k$ and $g_(k+1)$ have equal variance, the sum and difference variables
$
  S = (g_(k+1) + g_k) / 2,
  quad
  D = g_(k+1) - g_k
$
are independent Gaussians, with
$
  S ~ "Normal"(mu, sigma^2 (1 + rho) / 2),
  quad
  D ~ "Normal"(0, 2 sigma^2 (1 - rho)).
$

Using the identity
$
  |exp(a) - exp(b)| = 2 exp((a+b)/2) |sinh((a-b)/2)|,
$
the expected absolute period difference factorizes as
$
  expval(|T_(k+1) - T_k|) = 2 expval(exp(S)) expval(|sinh(D/2)|).
$

The first factor evaluates to
$
  expval(exp(S)) = exp(mu + sigma^2 (1 + rho) / 4),
$
while the second admits a closed-form expression in terms of the Gaussian error function.
After normalization by $expval(T_k) = exp(mu + sigma^2 / 2)$, all dependence on $mu$ cancels, yielding the exact prediction
$
  expval("jitter" | ell) = 4 Phi(sigma sqrt((1 - rho(ell)) / 2)) - 2,
$
where $Phi$ denotes the standard normal cumulative distribution function.

This expression constitutes a quantitative theory of jitter.
Given the marginal log-period variability $sigma$ and the kernel-dependent correlation $rho(ell)$, it predicts the expected period-to-period jitter without further assumptions.

For small arguments, the Gaussian CDF admits the expansion
$
  Phi(x) approx 1/2 + x / sqrt(2 pi),
$
which yields the asymptotic scaling law
$
  expval("jitter" | ell)
  approx
  (2 sigma / sqrt(pi)) sqrt(1 - rho(ell)),
$
recovering the intuitive result that jitter vanishes as $rho(ell) -> 1$, i.e. as the phase warp becomes locally affine.

To make the prediction concrete, we fix a representative adult pitch distribution with median period $T_0 approx 6.7$ ms (150 Hz) and log-period standard deviation $sigma = 0.25$.
The table below reports the predicted mean jitter for several kernel families and lengthscales.

#figure(
  tablem(
    //columns: (0.9fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr),
    fill: (_, y) => if calc.odd(y) { rgb("#eeeeee89") },
    stroke: frame(rgb("21222C")),
  )[
    | $ell$ (cycles) | $nu = 1\/2$ | $nu = 3\/2$ | $nu = 5\/2$ | $nu = oo$ |
    | ------------- | ---------- | ---------- | ---------- | ------ |
    | 5  | 12.0% | 6.2% | 5.0% | 4.0% |
    | 10 | 8.7%  | 3.3% | 2.6% | 2.0% |
    | 20 | 6.2%  | 1.7% | 1.3% | 1.0% |
    | 40 | 4.4%  | 0.85% | 0.64% | 0.50% |
    | 80 | 3.1%  | 0.43% | 0.32% | 0.25% |
  ],
  placement: auto,
  caption: [
    *Predicted mean period-to-period jitter* as a function of lengthscale $ell$
    for Matérn kernels with $nu in {1/2, 3/2, 5/2}$ and the squared exponential
    kernel ($nu = infinity$).
    Predictions use $sigma = 0.25$ and are obtained from the exact closed-form
    expression
    $
      expval("jitter" | ell)
      = 4 Phi(sigma sqrt((1 - rho(ell)) / 2)) - 2.
    $
  ],
) <table:jitter-vs-ell>

#figure(
  gnuplot(read("./fig/f0.gp")),
  placement: auto,
  caption: [
    *Analytic marginal fundamental frequency distribution* implied by the log-period prior.
    The distribution is lognormal with median 150 Hz, 5-95% range approximately 100-224 Hz (2.5–97.5%: 92–244 Hz).
  ],
) <fig:f0>

Normal adult jitter values around 0.5% are only attained for lengthscales of several tens of cycles under smooth kernels.
Rougher kernels require substantially larger $ell$ to suppress cycle-to-cycle variation.
In this sense, $ell$ admits a direct physiological interpretation as a cycle-to-cycle smoothness parameter governing the temporal regularity of voicing.

This framework yields falsifiable predictions linking observed jitter statistics to latent smoothness of the underlying pitch trajectory, which can subsequently be tested by fitting the model to data.

=== On derivative time scales (0.1 ms considerations)

We are given $u(t)$ but need $u'(t)$.
But hard differentiation yields huge spikes way over O(1) which defeats kernel modeling.
Luckily we are saved.

Glottal closure is not an instantaneous discontinuity but a short transition extending over a finite time interval. High-speed imaging and synchronized EGG measurements show that opening and closing events occupy O(0.1) ms durations and may exhibit small temporal offsets depending on how closure propagates along the folds (e.g. zipper-like anterior-posterior contact) @Herbst2014 @Orlikoff2012. Consequently, a derivative operator that reacts to arbitrarily small sample-scale fluctuations does not reflect the underlying physiology; it instead amplifies numerical artefacts or simulator-specific microstructure. A meaningful derivative therefore requires an explicit temporal scale reflecting the duration over which closure dynamics remain physically interpretable.

Since we already model $u(t)$ by Gaussian processes, we are implicitly asserting that first and second moments alone carry most of the important information.
Therefore, we propose to use a _Gaussian-smoothed derivative_ to measure the slope of a _locally averaged signal_
$
  partial_t (G_sigma * u(t))
$
rather than differentiating raw samples directly.
#footnote[
  Gaussian smoothing also turns out to be the unique linear kernel that preserves causality in scale-space, meaning that increasing scale removes fine structure without introducing spurious extrema @Lindeberg2013.
]
This type of differentiation results in another GP, since differentiation and convolution (smearing) are closed operators in GP space.

Selecting $sigma$ such that the effective smoothing width corresponds to approximately $0.1$ ms (about four to five samples at $44.1$ kHz) aligns the operator with the expected duration of glottal closure transitions. At smaller scales the derivative is dominated by high-frequency noise and discretization effects; at larger scales the transition itself becomes blurred. The resulting derivative can therefore be interpreted as a slope measured at the physically relevant temporal resolution rather than as a raw numerical difference.


== EGIFA

/*
The MAE score used here is very nondiscriminate and VERY sensitive to delays in the time domain
Its SD (stddev) is also insane, and makes ranking basically obsolete (even though they use ranking tests)
Worse: when using the --oracle method (ie use ground truth as the test solution), the oracle scores EQUALLY to most methods, even slightly worse than a few
This is because the orace is off by 14 samples; when using --oracle-delayed the oracle has zero error, as it should
They also use per-cycle metrics but this is bullocks: there is an affine shift equivalence class for scoring metrics THAT MUST OPERATE ON THE FRAME LEVEL (not cycle level) BECAUSE LPC IS APPLIED AT THE FRAME LEVEL
The authors (Chien+) also did not do null model (--whitenoise) calibration and go over this fixed delay of 14 samples (0.65 msec) very lightly
The other metrics (H1H2, NAQ) also don't really fix any of these problems; many of them are sensitive to this shift too

Therefore: we test our method with their eval suite, and expect to score basically ~in the middle => really basic test. Plus we also reproduce their numbers so all implementations work (already confirmed)
- We only do this with our final implementation; it can choose its own window and has access to GCI information, both estimated from waveform or ground truth
- We can also do this with our own whitekernel/periodickernel implementations for baselines

The REAL test is bringing these algos to our testing env, which is far superior and actually implements the correct equivalence class (affine shifts)
AND evaluates the inferred spectrum via formants, which is a metric that ALSO IMPLEMENTS EQUIVALENCE CLASSES
- We only do this with our final implementation
- Each GIF implemented here has its own params etc (and windows)

*/

=== About the dataset

=== Algorithms used

=== Problems

See @chapter:gauge for the gauge discussion

== Evaluation on EGIFA

== Evaluation on TIMIT-Voiced

== Summary