
#import "../lib/prelude.typ": bm

= PRISM-VFF
<chapter:prism-vff>

PRISM, in its original form, learns a _shared_ low-rank representation by placing inducing variables at locations $bm(Z)$ in the input domain and applying a collapsed variational bound.
This works well when the kernel is nonstationary or when the relevant input range is naturally bounded.
With stationary kernels the story is less pleasant: placing $bm(Z)$ on a finite interval commits the approximation to that interval, and the resulting basis is not naturally "about time differences" anymore.
In glottal inverse filtering (GIF), the nuisance of unknown offsets is pervasive; two derivative of glottal flow (DGF) waveforms can be physiologically similar and still arrive shifted by a fraction of a cycle.
A basis tied to absolute time locations is then asked to solve the wrong problem.

Stationarity is attractive precisely because it makes the prior indifferent to translations.
The annoyance is that the classical stationary spectral representation lives on the frequency axis, not on the time axis.
PRISM-RFF is the move that takes the PRISM machinery _as is_ and simply relocates the inducing variables from time to frequency, making the learned shared basis a Fourier-style basis.
Formally, this is an instance of an _interdomain sparse GP_: we choose inducing variables as linear functionals of the latent function, and the entire variational framework remains unchanged as long as we can compute $K_(u u)$ and $K_(u f)$.
This is the organizing principle behind interdomain GP approximations @Lazaro-Gredilla2009.

What makes the spectral move nontrivial is an old trap: the Fourier transform of a stationary GP does not define finite-variance random variables, so one cannot simply set inducing variables to be "the Fourier coefficients of $f$ on $bb(R)$."
_Variational Fourier Features_ (VFF) @Hensman2017 is an explicit treatment of this issue and motivates the key fix: _window the transform_.

In this chapter we specialize PRISM to stationary kernels with a known Bochner spectrum, and we keep the observation model Gaussian throughout.

== Data model and the normalized cycle coordinate

We work in a normalized cycle coordinate $tau$, so that a single period has length one.
The DGF waveform in a frame is observed at sample locations $tau_1, dots, tau_N$, not necessarily uniform in principle, though usually uniform in implementation.
The latent function is
$
  f(tau) ~ mono("GaussianProcess")(0, k(tau - tau')),
$
and observations are
$
  y_i = f(tau_i) + epsilon_i, quad quad epsilon_i ~ mono("Normal")(0, sigma^2).
$ <eq:obs-model>
The kernel $k$ is stationary.
By Bochner's theorem it admits a spectral density $S(xi)$ such that, using an angular-frequency convention,
$
  k(r) = 1/(2 pi) integral_(bb(R)) S(xi) e^(i xi r) dif xi.
$ <eq:bochner>
In PRISM-RFF we assume we can evaluate $S$ and we choose a parametric family for it that is convenient for learning and for closed forms.

=== Spectral Gaussian mixture model

We model the spectrum as a Gaussian mixture
$
  S(xi) = sum_(q=1)^Q A_q thin mono("Normal")(xi | mu_q, v_q),
$ <eq:sgm>
where $mono("Normal")(dot | mu, v)$ denotes a unit-area Gaussian density with mean $mu$ and variance $v$, and $A_q >= 0$ carries the spectral mass of component $q$.
Real-valued kernels correspond to even spectra, $S(xi) = S(-xi)$.
In practice one can enforce this either by parameterizing symmetric $plus.minus mu_q$ pairs explicitly, or by building symmetry into the formulas.

This spectral Gaussian mixture (SGM) view is not new in GP modeling; it is the same general idea that underlies spectral mixture modeling of stationary kernels, under different notational conventions.
The point here is not to invent a better kernel, but to give PRISM a spectral handle that is both learnable and algebraically tractable.

== Inducing variables in the spectral domain

=== Why we window

A naive attempt would define inducing variables as Fourier transforms
$
  u(omega) = integral_(bb(R)) f(tau) e^(-i omega tau) dif tau.
$
For stationary GPs this does not yield finite-variance variables: the integral is over the whole line with nondecaying weights, and $K_(u u)$ is not defined.
VFF explains this failure and motivates windowing as the remedy @Hensman2017.

We therefore define _windowed Fourier inducing variables_ at a set of inducing frequencies $omega_1, dots, omega_M$:
$
  u_m equiv u(omega_m) = integral_(bb(R)) f(tau) thin w(tau) thin e^(-i omega_m tau) dif tau,
$ <eq:windowed-inducing>
with a Gaussian window
$
  w(tau) = mono("Normal")(tau | 0, sigma_w^2).
$ <eq:window>
This choice aligns well with the speech framing intuition: within a window, a DGF frame is meaningful, while outside it we are content to let the prior dominate.
This is an interdomain GP construction, $u_m = L_m f$ with linear functionals $L_m$.
Interdomain theory guarantees that the sparse variational machinery needs only the covariances induced by these functionals, and otherwise proceeds exactly as in the inducing-point case @Lazaro-Gredilla2009.

=== Cross-covariances and prior covariance

Define the inducing vector $bm(u) = (u_1, dots, u_M)^top$.
We need $K_(u u) in bb(C)^(M times M)$ with $(K_(u u))_(m n) = "cov"(u_m, u_n)$, and $K_(u f)(tau) in bb(C)^M$ with $(K_(u f)(tau))_m = "cov"(u_m, f(tau))$.
These determine the conditional $p(f | bm(u))$ and hence the entire variational bound.

==== The window as a Gaussian factor in frequency
enters through the Fourier transform of $w$:
$
  hat(w)(nu) = exp(-1/2 sigma_w^2 nu^2),
$
up to the chosen Fourier convention @eq:bochner.
With this convention fixed, the window enters $K_(u f)$ and $K_(u u)$ as a Gaussian factor that couples the inducing frequency $omega_m$ to spectral frequencies $xi$.

==== $K_(u f)$ in spectral form
follows from
$
  (K_(u f)(tau))_m = bb(E)[u_m f(tau)] = integral w(t) e^(-i omega_m t) k(t - tau) dif t.
$
Inserting Bochner's representation @eq:bochner and exchanging the order of integration,
$
  (K_(u f)(tau))_m = 1/(2pi) integral_(bb(R)) S(xi) e^(-i xi tau)
  underbrace((integral_(bb(R)) w(t) e^(i(xi - omega_m) t) dif t), exp(-1/2 sigma_w^2 (xi - omega_m)^2))
  dif xi,
$
which gives the clean spectral form
$
  #rect(inset: 0.7em)[
    $ display(
      (K_(u f)(tau))_m = 1/(2pi) integral_(bb(R)) S(xi) thin
      exp(-1/2 sigma_w^2 (xi - omega_m)^2) thin
      e^(-i xi tau) dif xi.
    ) $
  ]
$ <eq:kuf>
The "blur" is now explicit: the window turns each inducing frequency into a local average of the spectrum around $omega_m$, weighted by a Gaussian of width $1 \/ sigma_w$.

==== $K_(u u)$ in spectral form
is derived analogously.
Starting from
$
  (K_(u u))_(m n) = bb(E)[u_m overline(u_n)]
  = integral w(t) w(t') e^(-i omega_m t) e^(+i omega_n t') k(t - t') dif t dif t',
$
each inner integral again produces a Gaussian factor, and we arrive at
$
  #rect(inset: 0.7em)[
    $ display(
      (K_(u u))_(m n) = 1/(2pi) integral_(bb(R)) S(xi) thin
      exp(-1/2 sigma_w^2 (xi - omega_m)^2) thin
      exp(-1/2 sigma_w^2 (xi - omega_n)^2) dif xi.
    ) $
  ]
$ <eq:kuu>
These two boxed formulas are the only kernel-specific ingredients PRISM-RFF needs.

== Closed forms for an SGM spectrum

Substituting the mixture @eq:sgm, both $K_(u u)$ and $K_(u f)$ become sums over $q$ of one-dimensional Gaussian integrals.
The general pattern is a Gaussian density in $xi$ from the mixture component, multiplied by one or two Gaussian window factors in $xi$, and in the case of @eq:kuf also by the complex exponential $e^(-i xi tau)$.
All of these remain Gaussian integrals.

=== $K_(u f)$ for one component

Fix a component $(A, mu, v)$.
The contribution to @eq:kuf is
$
  K_(u f)^((q))(tau; omega) = A/(2pi) integral mono("Normal")(xi | mu, v) thin
  exp(-1/2 sigma_w^2 (xi - omega)^2) thin e^(-i xi tau) dif xi.
$
Multiplying $mono("Normal")(xi | mu, v)$ by a Gaussian factor in $xi$ produces an unnormalized Gaussian; multiplying further by $e^(-i xi tau)$ shifts the mean into the complex plane but remains analytically integrable.
The result takes the form
$
  K_(u f)^((q))(tau; omega) = A/(2pi) thin C_q(omega) thin
  exp(-1/2 D_q tau^2) thin exp(-i E_q(omega) tau),
$ <eq:kuf-closed>
with explicit scalar expressions $C_q, D_q, E_q$ determined by $v, mu, sigma_w$, and $omega$.#footnote[
  Completing the square in the exponent of the product of the two Gaussians gives a new Gaussian with updated mean $tilde(mu) = (mu \/ v + omega \/ sigma_w^2)(1\/v + 1\/sigma_w^2)^(-1)$ and precision $(1\/v + 1\/sigma_w^2)$, and a Gaussian normalizing constant $C_q$.
  The subsequent integral over the remaining exponential $e^(-i xi tau)$ is then a standard Gaussian characteristic function evaluation.
]

Two observations matter more than the exact algebra.
First, the dependence on $tau$ is a complex sinusoid modulated by a Gaussian envelope, so the cross-covariances that become our basis functions are Gaussian-windowed complex exponentials; this is the mild sense in which Gabor atoms appear.
Second, the envelope width is controlled jointly by the window scale $sigma_w$ and the spectral component variance $v$.
We do not need to romanticize this: it is simply what Gaussians do under Fourier transforms.

=== $K_(u u)$ for one component

For the same component,
$
  K_(u u)^((q))(omega_m, omega_n) = A/(2pi) integral mono("Normal")(xi | mu, v) thin
  exp(-1/2 sigma_w^2 (xi - omega_m)^2) thin
  exp(-1/2 sigma_w^2 (xi - omega_n)^2) dif xi.
$ <eq:kuu-component>
This is purely real and positive.
It again reduces to a Gaussian integral and yields a closed-form scalar expression in $(omega_m, omega_n)$, $mu$, $v$, and $sigma_w$.

==== Non-diagonality of $K_(u u)$
is worth noting explicitly.
Even though $S$ is diagonalizing in the infinite-domain Fourier basis, the _window_ couples frequencies, so $K_(u u)$ is generally not diagonal in $(omega_m)$.
One should not expect independent Fourier coefficients under this prior.
VFF emphasizes that special structure such as diagonal plus low rank requires carefully chosen features and, in their work, particular kernel families and boundary constructions @Hensman2017.

=== Even spectrum and real features

For real stationary kernels we enforce $S(xi) = S(-xi)$.
With an even window $w(tau)$, the complex inducing variables come in conjugate pairs, and the model can equivalently be represented using a real feature vector of a DC term plus cosine and sine pairs,
$
  Phi(tau) = mat(1, cos(omega_1 tau), dots, cos(omega_M tau), sin(omega_1 tau), dots, sin(omega_M tau))^top.
$
These are the shape prototypes only.
In PRISM-RFF the actual design vector used by the sparse GP is the cross-covariance $k_u(tau) equiv K_(u f)(tau)$, which is a frequency-weighted, window-shaped version of these trigonometric functions determined by $S$ and $sigma_w$.

== Collapsed variational inference: PRISM machinery unchanged

The original appeal of PRISM was that the expensive part does not scale with the number of examples.
In the Gaussian likelihood case, Titsias-style variational inference gives a bound expressible in terms of low-rank objects involving $K_(u u)$ and $K_(u f)$, and the variational distribution over $bm(u)$ can be optimized in closed form, leaving only kernel hyperparameters and inducing locations to learn.

Interdomain inducing variables do not change that story: the bound and the conditional formulas remain the same, and only the meaning of $K_(u u)$ and $K_(u f)$ changes.
This is precisely the modularity emphasized in interdomain GP treatments @Lazaro-Gredilla2009.
In PRISM-RFF we exploit this modularity: the inducing "locations" are now inducing _frequencies_, $K_(u u)$ and $K_(u f)$ are computed by the SGM closed forms above, and everything else in the collapsed objective remains as in ordinary PRISM.
From a modeling perspective this is a change in _parameterization_ of the shared basis, not a change in the learning principle.

== Two levels of learning

PRISM-RFF produces two useful products, corresponding to two different notions of "what is learned."

=== Level 1: kernel learning

At the top level we optimize the SGM spectrum parameters $\{A_q, mu_q, v_q\}$, possibly constrained by symmetry, along with the inducing frequencies $omega_m$ and the window scale $sigma_w$, the latter often treated as fixed from framing considerations.

This level alone defines a stationary kernel on $tau$ and hence a full-rank GP prior usable independently of PRISM.
In other words, PRISM-RFF can be treated as a _kernel learning method_ that happens to use a sparse variational surrogate during learning.

==== The DC question
lives here.
Including an explicit $omega = 0$ basis element does not force a DC component in the data; it simply makes one representable.
If the learned spectrum places negligible mass near zero, then $K_(u f)(tau)$ communicates that to the model by making the corresponding coefficient irrelevant, and the posterior over that component remains concentrated near zero.

=== Level 2: amplitude learning

Once the shared spectral basis is learned, each waveform induces a Gaussian posterior over the inducing variables, and equivalently over a whitened coefficient vector.
The BLR interpretation follows from whitening the inducing variables:
$
  bm(u) = bm(L) bm(a), quad quad bm(L) bm(L)^dagger = K_(u u),
$
so that $bm(a) ~ mono("Normal")(bm(0), bm(I))$ under the prior.
The projected component of the GP is then
$
  g(tau) = k_u(tau)^dagger K_(u u)^(-1) bm(u)
  = underbrace((bm(L)^(-1) k_u(tau))^dagger, phi(tau)^dagger) bm(a),
$ <eq:projection>
where
$
  phi(tau) equiv bm(L)^(-1) k_u(tau)
$ <eq:feature-map>
is the learned _shared_ feature map: the cross-covariance to the inducing variables, whitened by $K_(u u)$.

For a given waveform with observations $bm(y) in bb(R)^N$, define the design matrix $bm(Phi) in bb(R)^(N times M)$ by $[bm(Phi)]_(i dot) = phi(tau_i)^top$.
We are then in Bayesian linear regression,
$
  bm(y) = bm(Phi) bm(a) + bm(epsilon), quad quad bm(a) ~ mono("Normal")(bm(0), bm(I)), quad quad bm(epsilon) ~ mono("Normal")(bm(0), sigma^2 bm(I)),
$ <eq:blr>
with posterior
$
  bm(a) | bm(y) ~ mono("Normal")(bm(m)_a, bm(S)_a),
  quad bm(S)_a = (bm(I) + sigma^(-2) bm(Phi)^top bm(Phi))^(-1),
  quad bm(m)_a = sigma^(-2) bm(S)_a bm(Phi)^top bm(y).
$ <eq:posterior>

Each waveform thus becomes a Gaussian in coefficient space, and those Gaussians are precisely what we can cluster, embed via a GP-LVM, or otherwise postprocess.

==== Phase information
is encoded at this second level.
Spectral _power_ alone does not distinguish waveform families that differ by time shift or asymmetry within the cycle.
The posterior over $(cos, sin)$ coefficients, or equivalently over complex coefficients, does contain relative phase information.
This is why it is plausible that modal, breathy, whispery, and creaky DGF families can separate in the learned coefficient space even when the discontinuity-like behavior of the closed phase looks challenging in a purely magnitude spectrum view.
The representation is not a bare periodogram; it is a Bayesian projection of the waveform onto a learned set of windowed Fourier atoms.

== Ordinary PRISM versus PRISM-RFF for DGF

It is worth being explicit about what changes and what does not.

=== What does not change
is the learning objective, the collapsed variational logic, and the interpretation of the projection as a Bayesian linear regression with an $mono("Normal")(bm(0), bm(I))$ prior after whitening.
Interdomain inducing variables are still inducing variables; they simply require different covariances @Lazaro-Gredilla2009.

=== What changes
is the _shape_ of the shared basis.
In ordinary PRISM the basis is built from kernels centered at inducing times $bm(Z)$.
In PRISM-RFF it is built from windowed spectral projections, so the shared basis elements behave like localized sinusoids in the Gaussian-windowed sense above.
VFF gives the cleanest explanation for why windowing is structurally necessary in the stationary case @Hensman2017.

For DGF this matters because offsets are endemic and expensive to correct perfectly.
A stationary prior reduces sensitivity to absolute phase, and a Fourier-like basis can represent shifts largely by rotating mass between cosine and sine components.
This does not eliminate alignment issues, but it means the representation can be stable under the small sub-cycle shifts that often dominate scoring variability in GIF.

=== A note on harmonizable connections
<sec:harmonizable>

One can tell a broader story in which windowed Fourier features bridge stationary and certain nonstationary kernels.
@Shen2019 frame variational Fourier features in a generalized spectral representation language and show how to compute covariances between a GP and its Fourier transform when the kernel is harmonizable and integrable; they also discuss windowed transforms explicitly.

PRISM-RFF does not need the full harmonizable apparatus: our target kernels are stationary in $tau$.
The only role of the window here is to produce valid inducing variables and to match the framing logic used in speech processing.

== Practical capacity: how many inducing frequencies

In normalized cycle time $tau$, the relevant highest frequency is determined by the finest time structure one expects the DGF to meaningfully carry.
If the smallest meaningful time scale in physical time is $Delta t_min$, then in normalized time it becomes $Delta tau_min = Delta t_min \/ T_0$, with $T_0$ the period for that frame.
The corresponding rough upper harmonic index is on the order of
$
  H_max ~ 1/(2 Delta tau_min).
$
Since $T_0$ varies widely for voiced speech, $H_max$ in $tau$ is not constant across speakers and pitch, but the normalization reduces this variability substantially.
In practice, $M$ in the range $16$ to $128$ inducing frequencies is often enough to capture the stable low- and mid-frequency structure, while leaving the sharpest closed-phase edges to the residual process $h(tau) = f(tau) - g(tau)$ in the standard sparse GP decomposition.
This decomposition perspective is emphasized in VFF: the approximation captures a low-frequency subspace and leaves the rest as an orthogonal complement @Hensman2017.

== Summary

PRISM-RFF is PRISM with a spectral choice of inducing variables.
We choose a stationary kernel through its Bochner spectrum $S(xi)$, here parameterized as a Gaussian mixture @eq:sgm.
Inducing variables are defined as windowed Fourier projections @eq:windowed-inducing to ensure finite variance and tractable covariances, as motivated by VFF @Hensman2017.
$K_(u u)$ and $K_(u f)$ follow from closed-form Gaussian integrals @eq:kuu and @eq:kuf, and plug into the standard sparse variational machinery; interdomain theory guarantees nothing else needs to change @Lazaro-Gredilla2009.
The result is interpreted at two levels: learned kernel hyperparameters and inducing frequencies define a stationary prior usable on their own, and per-example posteriors over whitened coefficients @eq:posterior give a compact representation suitable for clustering and downstream latent modeling.

The main conceptual gain for DGF is that the shared basis is no longer tied to absolute time locations.
It is tied to spectral content under a window, which makes the representation naturally tolerant of the small time offsets that dominate practical GIF variability.
