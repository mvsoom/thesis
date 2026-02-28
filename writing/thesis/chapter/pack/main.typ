#import "../../lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "../../lib/gnuplot.typ": gnuplot

#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/equate:0.3.2": equate, share-align
#show: equate.with()

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

/*
Writeup
- Why PRISM
- Scaling and batching aspect
- Comparison to kernel PCA
- MoPPCA
- Objective to be optimized
- What kind of distance between waveforms is induced?
*/

= The periodic and quasiperiodic arc cosine kernel
<chapter:pack>

@chapter:gfm established parametric and nonparametric glottal flow models for a _single period_ of the glottal cycle.
In this chapter and the next, our goal is to extend this basic building block to _multiple periods_.
Recordings of voiced speech often visually exhibit self-similarity on timescales of $cal(O)(10 "msec")$ because in normal speech the glottal cycle often reaches steady-state, and a clear pitch is perceived as a consequence.#footnote[
  Results from clinical trials @Little2007 indicate that normal (non-pathological) voices are usually something called Type I @Titze1995; that is, phonation results in nearly periodic sounds, which supports the notion that uttered vowels /*(which make up the majority of voiced speech)*/ in normal speech typically reach a steady-state before "moving on".
  Thus when a vowel is perceived with a clear and constant pitch, it is reasonable to assume that the vowel attained steady-state at some point (though perceptual effects forbid a one-to-one correspondence).
  In practice, steady-state vowels are identified simply by looking for such quasiperiodic strings, which typically consist of about 3 to 5 pitch periods @Rabiner2007.
]
An illustrative example for the case of steady-state vowels is given in @fig:quasi-periodicity.

/*
What is meant by steady-state vowels is the steady-state portion (which may never be attained in some cases) of a vowel utterance.
This is the time interval in which (a) the VT configuration can be taken to be approximately fixed, leading to essentially constant formants, and (b) the vocal folds are sustained in a reasonably steady vibration called the glottal cycle, during which the vocal folds open and close periodically.
Because of (a) and (b) the vowel waveform appears as a quasiperiodic string of elementary units called pitch periods.
@Peterson1966 // p. 75
*/

To capture this steady-state behavior of the glottis, we idealize for now and assume perfect periodicity in this chapter, which gives rise a to periodic-yet-nonstationary Gaussian process governed by the _periodic arc cosine kernel_ (PACK).
Looking ahead, @chapter:pack then relaxes this assumption and models quasiperiodic glottal flows with the _quasiperiodic arc cosine kernel_ (QPACK), which is a complete nonparametric model for source signals driving steady-state voiced speech.
Coupling back to @fig:quasi-periodicity, when paired with vocal tract filters, the PACK can model only synthetic waveforms as in (a), while the QPACK has full support for real speech waveforms as in (b).

/*
TODO: improve paragraph below. talk about the PACK model then learning from surrogate flow
*/
We "finetune" the PACK on example generations of the LF model.
Finally, we validate the PACK on `OPENGLOT-I` and `OPENGLOT-III`, as these both employ synthetic and perfectly periodic LF signals as their source signals. // TODO: we will?

#figure(
  grid(
    columns: 2,
    column-gutter: { 2em },
    row-gutter: { 1em },
    align: center + bottom,
    image("./fig/bdl-gamma.svg", height: 17%), image("./fig/bdl-ae.svg", height: 17%),
    [(a)], [(b)],
  ),
  placement: auto,
  kind: image,
  caption: [
    *Examples of steady-state vowels*
    during three pitch periods.
    Here (a) is perfectly periodic because it was synthesized with @eq:lf; and (b) is quasiperiodic because it is taken from a real recording.
    #footnote[
      Taken from a recorded steady-state instance of the vowel /æ/ at a fundamental frequency of 138 Hz.
    Source: `bdl/arctic\_a0017.wav:0.51-0.54sec` @Kominek2004.
    ]
    The timescale is indicated by the horizontal bars
    #box(baseline: 0pt)[
      #stack(dir: ltr,
        h(2pt),
        line(start: (0pt, -3pt), angle: 90deg, length: 6pt),
        line(length: 2.5em, stroke: 1.3pt),
        line(start: (0pt, -3pt), angle: 90deg, length: 6pt),
        h(2pt)
      )
    ]
    below the waveforms, each bar having a length of 5 msec.
  ],
  gap: 1em,
) <fig:quasi-periodicity>

== The periodic arc cosine kernel

// from this writing session: https://chatgpt.com/g/g-p-68f9d6b4a46c81919b645b342ba50e41-ongoing/c/6915a9bc-afa8-8327-98d8-f7c99c085e83

Recall from the previous chapter that a single glottal cycle consists of three phases: the open phase (O), an optional return phase (R), and the closed phase (C).
In @chapter:gfm we identified the temporal arc cosine kernel (TACK) as a good candidate glottal flow model during the open phase.

#share-align[
Assuming the glottal flow $u(t)$ is zero outside that phase, such that $u'(t)$ is identically zero too, our full model during _a single pitch period_ $[0, T]$ is:
$
  u'_"cycle" (t) = cases(
    u'_"NN" (t) quad quad & 0 < t <= t_c quad quad quad quad & "(O)",
    0 quad quad & t_c < t <= T quad quad quad quad & "(C)",
  )&
$
and zero outside that pitch period.
We thus have a single bump $u(t)$.
// TODO picture of single bump and periodized with 0, T, t_c annotated

This induces the _full-cycle_ kernel
$
  k_"cycle" (t, t') = cases(
    k^((d))_bm(Sigma) (t, t') quad quad & t\, t' in [0, t_c] quad quad & "(O)",
    0 quad quad & "otherwise on" [0, T] quad quad  & "(C)",
  )&
$
]

We cannot make use of classic periodization formulas because these are defined classically only for _stationary kernels_, @Rasmussen2006 #cite(<Scholkopf2002>, supplement: [Eq. 4.42]).
For that reason the usual periodic-kernel construction
$k_"per" (t,t') = sum_j k(t - t' + j T)$ is not appropriate here.

We instead periodize the _function_ itself by summation:
$
  u'_T (t) = sum_(j in bb(Z)) u'_"cycle" (t + j T).
$
Since the single bumps are non-overlapping, this boils down to gluing copies end-to-end.
Here it is assumed $t_c < T$, as otherwise the covariance for any $(t, t')$ will diverge due to non-decaying covariance in $|t-t'|$, unlike stationary kernels (implied by their spectrum being integrable).

==== Fourier series
Any well behaved $T$-periodic function admits a discrete Fourier representation
$
  u'_T (t) = sum_(k in bb(Z)) c_k exp(i 2 pi k t \/ T),
$ <eq:periodic-fourier-series>
with Fourier coefficients $bm(c) = (..., c_(-1), c_0, c_1, ...)^top$
$
  c_k = integral_0^T u'_"cycle" (t) exp(-i 2 pi k t \/ T) dif t.
$
Because each $c_k$ is a linear functional of the Gaussian process $u'$(t), the
vector $bm(c)$ is jointly Gaussian.  Therefore @eq:periodic-fourier-series is a
Karhunen-Loève expansion of the periodic process $u'_T (t)$.
Nothing has been
approximated: periodicity alone fixes the basisfunctions $exp(i 2 pi k t / T)$.

In this representation the PACK is specified entirely by the covariance
$
  expval(c_k overline(c_(k'))) =
    integral_0^T integral_0^T
      k_"cycle" (t, t')
      exp(-i 2 pi k t / T)
      exp(i 2 pi k' t' / T)
    dif t dif t'.
$
Inference is then linear in the number of retained harmonics, rather than
cubic in the number of time points.

It thus remains to evaluate these integrals, i.e. to compute the Fourier
bitransform of the kernel.

/*
==== Sampling frequencies
Our sampling of freqs in the Nyquist band reproduces the Sinc kernel-property that we automatically band limit the smoothness => good prior information

Nyquist reconstruction essentially expresses that by bandlimiting frequency information we limit smoothness, thereby making perfect reconstruction possible in principle.

Multiplying with SqExp with low order Hilbert amplitude envelope doesn't add high freq information, neither does summing with slowly varying DC background, so principle is still preserved. However, white noise delta reintroduces full spectrum!

Matern kernels and others do not take into account Nyquist information, whereas RFF (random fourier features) / SM (spectral mixture) kernels do.
*/

/*
==== Periodic but nonstationary
Note that kernels can be periodic but nonstationary, like ours.
Typically we think in terms of periodic stationary kernels, but this does not have to be always so.
See example in Marocco notebook.
*/

/*
==== Complex GPs
For any complex random variable $z = x + i y$, the second-order information
(the shape of its density if Gaussian) lives entirely in the second moments of
$x$ and $y$. Those are captured by the $2 times 2$ real covariance matrix:
$
  bm(Sigma) = mat(
    E[x^2], E[x y];
    E[x y], E[y^2]
  )
$
But it's convenient and natural in complex analysis to rewrite this in
complex coordinates. There, the two invariant quadratic forms you can make
from $z$ are $E[z z^*]$ and $E[z z]$. The first is the usual covariance (energy
or total power), the second measures the departure from circular symmetry
(orientation or eccentricity).

They're _complementary_ because together they completely reconstruct the real
$2 times 2$ covariance:
$
  bm(Sigma) = 1/2 mat(
    Re(&bb(E)[z z^* + z z]), Im(&bb(E)[z z]);
    Im(&bb(E)[z z]), Re(&bb(E)[z z^* - z z])
  ).
$
They're _independent_ (in the representational sense) because $E[z z^*]$ is
invariant under rotation $z -> e^(i theta) z$ (it sets the overall scale), while $E[z z]$
transforms like $e^(i 2 theta)$: $E[z z] -> e^(i 2 theta) E[z z]$ — it tracks the anisotropy's orientation and
ellipticity.
*/

==== Interdomain view
It is instructive to position this within GP interdomain-feature literature.

In sparse GP methods (e.g. @Titsias2009), one introduces linear
functionals of the process
$
  L_j(u') = integral phi_j(t) u'(t) dif t,
$
and treats the vector $bm(L)$ as latent Gaussian variables with covariance
matrix
$
  bb(E)[L_j L_k] = integral.double k(t,t') phi_j(t) phi_k(t') dif t dif t'.
$

The Fourier coefficients $c_k$ are exactly such interdomain features with
$phi_k(t) = e^{-i 2 pi k t/T}$.  
The difference is that here:
we are not approximating the GP,  
we are exploiting periodicity to identify its exact KL basis.

This eliminates the entire variational machinery usually required for
spectral features.  
The “interdomain’’ structure is present, but nothing is approximate or learned.

==== Symmetry view
It is seen that translation invariance over $T$-intervals give rise to the Fourier decomposition of the kernel, which is exact, and provides for a considerable computational speedup.
This particular instance (1D input, Fourier invariance) can be generalized to any symmetry of the input space @Sun2021, including the translational invariance of the familiar stationary kernels and the rotational invariance of dot product kernels.

==== Taking advantage of $f_s$
The finite Nyquist sampling frequency is exploited here, as in #pcite(<Tobar2019>)'s sinc kernel which takes into account the band-limited nature of sampled time series.
Given noiseless observations, the signal can be predicted everywhere perfectly via the Nyquist reconstruction theorem,
the precise nature of which described by the slowly-decaying tails of the sinc kernel.

== The quasiperiodic arc cosine kernel



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

== Learning Fourier features

Eigenfunctions do not depend on hyperparameters, only the expansion coefficients, as in Hilbert-GP
- In fact we can do $cal(O)(N log N)$ learning due to regular grid

==== Steps of inference producedure
TLDR; first ALIGN inductive bias (on the hyperparam level), then REFINE inductive bias (on the coefficient/GP level).

- Infer $p(theta | "examplars")$ via Nested Sampling: same as previous thesis
- Each $theta$ is a "key" that brings posterior in $N_k$ space (data length of examplar $k$) to common coefficient space $bm(a)$
  - For each $theta$: compute posterior $p(a_k | D_k)$ via normal GP inference
  - Then "envelope" these posteriors via $D_"KL"$ inference
  - This can be repeated for as many samples of $theta$ as wanted, thereby increasing support

After that we toss $theta$ and end up with a single posterior $p(a) = "Normal"(mu^*, Sigma^*)$ which encodes the inductive bias to GFMs, without collapsing to a single learned $hat(bm(a))$.

Therefore the prior GP model and its hyperparameters $theta$ act like a "decoding key" and determine posterior density in coefficient $bm(a)$-space.

These are Fourier features directly in the Bayesian linear regression form due to our periodic expansion into Fourier series.
When we add quasiperiodicity, the basis changes, but the method stays exactly the same, and we get quasi-Fourier features encoded in the $bm(a)$.
Here "quasi" implies that AM and DC modulation are both modest.
In all cases are the basisfunctions fixed for efficiency, but this need not be the case: the $theta$ can also influence the basisfunctions, and these can be learned in the first step;
in this case we can't _sample_ $theta$ anymore, since the (features) basisfunctions depend on it, but can just use an argmax $theta$.


==== Why not do PCA?
PCA learns the basis functions from a SVD of the empirical covariance matrix.
You can turn it into a generative model by fitting a Gaussian to the score vector.

We do something more elaborate, which has following advantages:

1. Model functions, not vectors:
Irregular grids, different amount of samples, missing data all heandled.
PCA needs rectangular data.
This also means that we can condition on stuff: eg, when varying $T$ we can still condition on previous or future data with QPACK

2. Uncertainty handling
Uncertainty is retained in a principled way at each step, especially if we sample the $theta$ rather than optimizing.
(We effectively integrate out the $theta$.)

3. Prior
We refine the prior into a smaller support, but not too small.
We want to tame it but not make it tame; we still need to be prepared for what's out there in the wild.
This has advantages:
- Attempts to quantify the physics/geometry underlying the signals.
- Can do with very little examplars (PCA needs many samples if $N$ is large as the empirical covariance matrix can take time to converge) which is handy for expensive simulations.
- Predicting out-of-data-distribution handled consistently; fall back to predictions based on underlying assumed model OR inflate uncertainty accordingly.

In short: PCA: empirically-oriented and data exploration phase; BNGIF: try to turn what we know quantitatively in a uncertainty-calibrated model.


// == Evaluation on `OPENGLOT-I`

== Summary

We extended the TACK to multiple pitch periods.
Our idealization of perfect periodicity allowed us to derive an efficient Karhunen-Loève representation of the corresponding PACK.

We now have a kernel that checks following properties of theoretical GFMs:
- periodicity
- differentiability and spectral case
/*
@Rosenberg1971, p. 587: Decays of order 12 dB/oct are typical
*/
- closure constraint
- hard/soft GCI support

To add more realism, we relax these hard constraints and learn expand support again using simulations.

/*

==== How good of a GFM is this still?
After all this surgical changes to basic GFMs we run through our checklist to see if this still can a priori represent what we need: the glottal cycle! @fig:glottal-cycle

Before doing so, we take a look at how much of viable candidates @eq:udH still are as GFMs.

Differentiable: yes

Domain: yes

No return phase unless very lucky, tiny support

Closure constraint: we could restrict this analytically

Putting priors will enable us to trace out a family of GFMs


Polished rewrite of the above:

==== How good of a GFM is this still?
It remains a valid generative model of the glottal cycle: differentiable, time-localized, and periodic by construction.
The closure constraint can be imposed analytically in the finite case and later in functional form through interdomain features.
Return-phase behaviour and sharp glottal closures are naturally expressed through steep RePU transitions; genuine discontinuities would require Lévy-style processes, but the present formulation already approximates them closely in practice.
*/
