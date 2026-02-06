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

= The quasiperiodic arc cosine kernel
<chapter:qpack>



The quasiperiodic arc cosine kernel (QPACK)

We implement two types of quasiperiodicity:
- time warping
- direct AM and DC modulation


Reason for DC baseline: pathological voices, as observed by
#quote[
  /*
  Although the original LF model is well-suited to
  synthesizing normal voice quality, some constraints and
  modifications were needed to fit the model to experimental data derived from pathological voices. The present version of the LF model differs from the original in
  several respects in the modeling of the return phase.
  First, point tc is not constrained to equal point t0 for the
  following cycle (see Figure 1), so the closed phase is formally modeled in this implementation. Second, */
  _in many
    cases in our data, returning flow derivatives to 0 at the
    end of the cycle conflicted with the need to match the
    experimental data and conflicted with the requirement
    for equal areas under positive and negative curves in the
    flow derivative (the equal area constraint). Empirically,
    this constraint means that the ending flow level should
    be the same as the beginning flow level (although that
    value need not equal 0; Ní Chasaide & Gobl, 1997). This
    is probably not true for highly variable pathological voices.
    In many cases, the combination of this constraint with
    fitting the experimental pulses resulted in a flow derivative that stepped sharply to 0. This introduced significant high-frequency artifacts into the synthetic signals.
    These conflicts between model features, constraints, and
    the empirical data were handled by abandoning the
    equal area constraint_
  @Kreiman2007
]

== Making the PACK quasiperiodic

Need for this:
- Experiment `experiments/iklp/openglot-II` shows that IKLP will combine kernels of different pitch $T$ to make quasiperiodic $u'(t)$ out of periodic SqExpKernel samples

/*
This section can be quite short => dont have to explain Hilbert GP completely

note that we can add quasiperiodicity immediately by working in kernel domain (not basisfunctions domain) and take the SVD

so stress why we take this faster, stabler route
*/

Hilbert expansion of:
- AM term => lengthscale param
- DC term => lengthscale and scale param

== Alignment, or fixing the gauge

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

== Learning with t-PRISM

Tricks we use:
- Normalize data to (0,1): this means we freeze mean to 0 *and* kernel variance to 1.
- Lengthscale in ballpark order: 10. This is simply to cut number of iterations, the model does find lengthscales ~ 15 if initialized from 1, but takes much longer.
- Inverse-ECDF initialization of inducing points. This is important because trajectory lengths are long tailed, and we need to place predictive power where it matters. In 1D we can just sample from the empirical distribution, with interpolation.
- Set $nu$ based on event statistics. Optimizing $nu$ is possible, but more stable optimization. Results depend only very slightly on exact value of $nu$ as long as $<10$.

For the rest standard Adam optimizer with standard settings.

==== Heuristic for setting $nu$
We set $nu$ using a simple tail-probability heuristic derived from the expected frequency of spike events. After normalization, the GP prior has unit variance and the learned observation noise standard deviation is approximately $sigma approx 0.15$. Typical spikes have amplitude around $5$, corresponding to residuals of size $r approx 5 / 0.15 approx 30-35$ noise standard deviations. We interpret spikes as events that should lie in the heavy tails of the Student-t likelihood rather than being explained by the Gaussian core. Therefore we choose $nu$ such that the tail probability of a Student-t distribution satisfies $P(|X| > r) approx p_"spike"$, where $p_"spike"$ is the empirical spike rate (here about $1%$). Using the large-$r$ asymptotic tail behaviour $P(|X| > r) approx C * r^(-nu)$ and ignoring constants gives the practical rule $nu approx - log(p_"spike") / log(r)$. For $r approx 35$ and $p_"spike" approx 0.01$, this yields $nu approx 1-2$, which corresponds to a Cauchy-like heavy-tailed likelihood. This choice ensures that extreme residuals are downweighted rather than forcing the GP to explain them through excessive curvature.

== Evaluation on `OPENGLOT-II`

== Summary