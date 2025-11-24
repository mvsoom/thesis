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

/*
TODO: figures (search for TODO below)

- Samples from neural net RePu model with d = 0, 1, 2, 3 H = small, large, infinite, with and without closure constraint. Parameters sampled from their priors except for (S)TACK.
- Model comparison figure with NS: a table
*/

= From parametric to nonparametric glottal flow models
<chapter:gfm>

Glottal flow models (GFMs) describe the source signal $u(t)$ as it drives the vocal tract during voiced speech.
These models operate in the time domain because the delicate phase characteristics of the _glottal cycle_ are an integral part of vocal communication.#footnote[
  For example, plosives: glottal stops are micro-events at the millisecond level that underlie semantic information two orders of magnitude higher up the scale.
  The timing of the vocal fold movement is so precisely controlled by our brain that the slightest deviations are studied as biomarkers for neurodegenerative diseases like Parkinson's @Ma2020.
]

Decades of empirical work have over time produced a "model zoo" of GFMs: handcrafted waveforms of $u(t)$ with handfuls of carefully chosen parameters.
Because these are _parametric_ models, they all share the same basic trait: /*parsimonious, */ interpretable, but inevitably inflexible @Schleusing2012. Their parametric nature limits the range of time domain information they can encode.//, which is sometimes called the parametric bottleneck.
This motivates the construction of a family of _nonparametric_ models, which are more expressive as they are able to grow "parametric capacity" depending on the data at hand.
//while encoding structure at a higher level.
//reach for greater expressivity with _less_ parameters.

With the many equations to follow, it is all too easy to lose connection to the actual physical phenomenon being modeled.
After a short look at the glottal cycle, we begin with a review of the classic Liljencrants-Fant model.
Then we revisit the parametric piecewise polynomial models and generalize them to the nonparametric regime, where they are identified with the _arc cosine kernel_ of #pcite(<Cho2009>).
The latter is argued to combine the advantages of both worlds: the interpretability of parametric models with the flexibility of nonparametric ones.
This then sets the stage for @chapter:pack, which derives the _periodic arc cosine kernel_ as a learnable surrogate model of synthetic glottal flow data.

#figure(
  image("./fig/glottal-cycle.png", height: 28%),
  placement: auto,
  caption: [*A look at the glottal cycle* via laryngeal stroboscopy of the vocal folds. Each frame here is a picture taken at intervals spanning different cycles to give the illusion of steady-state behavior. The glottis is the opening between the folds. After #pcite(<Jopling2009>) #pcite(<Chen2016>).],
) <fig:glottal-cycle>
 
==== The glottal cycle
is the periodic opening and closing of the vocal folds, depicted in @fig:glottal-cycle.
During one _pitch period_, a single cycle is completed, in this case spanning $T = "6 ms"$.
GFMs describe the rate of airflow $u(t)$ [typically expressed in cm³/s] through the opening between the vocal folds called the _glottis_, visible as a black slit in @fig:glottal-cycle.
Importantly, it is not at maximum glottis aperture that the acoustic output is greatest; it is at the sudden _glottal closure instant_ (GCI), as first shown by #pcite(<Miller1959>) in modern literature.
At this point, the moving air column in the vocal tract is abruptly interrupted, and kinetic energy is converted efficiently into acoustic energy which then excites the vocal tract much like plucking a harp or clapping in a resonant room @Chen2016[Section~4.6].
The sharpness of this transition governs much of the clarity and perceived strength of voiced speech @Fant1979.
Thus, efficiency demands that the glottal cycle be strongly asymmetric: a slow buildup of flow during the open phase, followed by a rapid, almost impulsive closure to the closed phase.

== The Liljencrants-Fant model

A model that has been very useful in describing how $u(t)$ varies during the glottal cycle is the _Liljencrants-Fant (LF) model_ proposed by #pcite(<Fant1985>).
It has been the GFM of choice for many joint-inverse filtering approaches#footnote[
  See #section-title-page(<sec:joint-source-filter-methods>).
] but also been studied in its own right.#footnote[
  See #pcite(<Degottex2010>, supplement: [Section~2.4]) for a discussion. Research into the LF model falls mainly into study of its spectral characteristics @Doval2006 and (re)parametrization, notably #pcite(<Fant1995>) #pcite(<Perrotin2021>).
]

The model states that $u'(t)$ in a single pitch period of length $T$ consists of three parts, or phases. The _open phase_ (O) is modeled as a rising and falling exponential sinusoid part, and transitions to the _closed phase_ (C) via an exponential return during the _return phase_ (R). The latter is thus part of (C). Mathematically, this is defined piecewise as
$
  u'_"LF" (t) = cases(
    -E_e e^(a (t - t_e)) sin(pi t \/ t_e)/sin(pi t_e \/ t_p) & 0 <= & t <= t_e & "(O)" &,
    -E_e / (epsilon T_a) ( e^(-epsilon (t - t_e)) - e^(-epsilon (t_c - t_e)) ) quad quad &t_e < &t <= t_c quad quad &"(C, R)"&,
    0 & t_c <= & t <= T & "(C)" &,
  )
$ <eq:lf>
where $bold(theta)_"LF" = {E_e, t_p, t_e, t_c, T_a, T}$ are the six model parameters explained below in @eq:lf-parameters and ${a, epsilon}$ are determined implicitly from the _closure constraint_
$
  integral_0^T u'(t) dif t = 0 quad "such that" quad u(0) = u(T),
$ <eq:closure-constraint>
which holds for any GFM, and the continuity constraint
$
  lim_(t ->_< t_e) u'(t) = lim_(t ->_> t_e) u'(t)
$ <eq:lf-continuity>
which is peculiar to GFMs with smooth return phases rather than hard, discontinuous GCI events @Veldhuis1998.
Note that like most GFMs, the LF model is defined in $u'(t)$ not $u(t)$ due to the radiation effect @eq:suh, though $u(t)$ can be recovered easily via integration @eq:udu.
@fig:lf shows the waveforms for $u'_"LF" (t)$ and $u_"LF" (t)$ for a typical setting of $bold(theta)_"LF"$.
When varying $bold(theta)_"LF"$ the LF waveforms trace a family of broad to narrow asymmetric pulses with smooth to abrupt closure encoding positive ($u(t) >= 0$) glottal flow.

#figure(
  gnuplot(read("./fig/lf.gp")),
  placement: auto,
  caption: [
    *The Liljencrants-Fant model* for #box(baseline: -3pt)[#line(length: 1.5em, stroke: 1.3pt + blue)] $u'_"LF" (t)$ and #box(baseline: -3pt)[#line(length: 1.5em, stroke: 1.3pt + red)] $u_"LF" (t)$ during a single period of length $T$. The open phase (O), closed phase (C) and return phase (R) are marked at the top and are determined by the changepoints ${t_p, t_e, t_c, T}$. The amplitude at peak excitation $t_e$ is $E_e$. Not shown here is time constant of the return phase $T_a$, which during (R) determines the steepness of exponential decay towards zero in $u'(t)$.
  ],
) <fig:lf>

==== Parameters of the LF model
The LF model achieved broad adoption because it strikes a balance between simplicity and empirical accuracy.
After several iterations, #pcite(<Fant1985>) arrived at this parametrization:
$
  bold(theta)_"LF" = cases(
    E_e & = min_t u'(t) = u'(t_e) quad quad & "peak excitation amplitude",
    t_p & = argmax_t u(t) & "instant of peak glottal flow",
    t_e & = argmin_t u'(t) & "instant of peak excitation",
    t_c && "instant of glottal closure (GCI)",
    T_a && "time constant of the return phase",
    T && "fundamental period",
  )
$ <eq:lf-parameters>
which turned out to be effective in capturing the glottal cycle in both modal and more extreme phonations.
#footnote[
  Actually, #cite(<Fant1985>, form: "author") present LF as a "four-parameter model".
  They ignore $E_e$ and set $T := t_c$ for convenience, thus arriving at a count of four not six as in @eq:lf-parameters.
  In their unifying theory of common GFMs, #pcite(<Doval2006>) also collapse $T$ to $t_c$.
  It is common too in applications (for example, #ncite(<OCinneide2012>)).
  Of course, nothing is really lost because a closed phase of any length can be appended to any GFM by piecewise concatenation.
  We choose to keep $t_c$ distinct from $T$ explicitly because this becomes significant for short return phases and is experimentally observed to be important, especially in clinical scenarios @Rosenberg1971 @Kreiman2007.
]
Note that, apart from $E_e$, which sets the vertical scale, these are all time domain parameters which determine _changepoints_ in the glottal cycle in @fig:glottal-cycle.
These tend to covary significantly and can in fact be regressed into a single convenient parameter $R_d$ @Fant1995.

From the modeling perspective, the most important changepoints are $t_e$ and $t_c$: when these lie close together and return is swift ($T_a$ small and vanishing return phase), the glottal cycle is at peak efficiency and $tilde(u)(t)$ has broadband spectral energy.
This results in a clear voice and sharp features in the speech waveform data $bm(d)$.
Conversely, a smooth return ($T_a$ medium to large) results in a longer return phase with $t_c$ tending to $T$ and the GCI becoming ill-defined.
The result here is a more mellow voice and more smooth features in $bm(d)$.
Both modes are common in voiced speech @Maurer2016 and GIF algorithms must be able to handle both cases.

==== Implementation
In practice, the LF model is unwieldy to compute @Veldhuis1998 @Gobl2017. The closure and continuity constraints in @eq:closure-constraint and @eq:lf-continuity lack an analytical solution, meaning that for each evaluation of $u'(t)$ one must numerically solve for the auxiliary constants ${a, epsilon}$ by a bisection or root-finding routine. This seems like a high price to pay for an analytical model, and worse, this optimization is brittle.
As noted by #pcite(<Perrotin2021>), simplified LF models exist that are perceptually equivalent to LF and easier to work with.
//Perhaps authors should retire LF altogether.

Nevertheless, it remains the de facto standard in parametric source-filter modeling and we therefore use it here. Our code in @chapter:jaxlf implements the LF equations in `JAX` @Bradbury2020, so the bisection routines are compiled to native code for a speedup and the entire model is differentiable and easily batcheable.
The code also implements four LF parametrizations, including the previously mentioned $R_d$ formulation of #pcite(<Fant1995>).

=== One of many: the glottal flow “model zoo”
<sec:model-zoo>

#figure(
  image("./fig/gf-models.png", width: 100%),
  placement: bottom,
  caption: [
    *A lineup of GF models* back from 1986 (handdrawn?), together with their derivatives (DGFs). These models are visually very similar. From #cite(<Fujisaki1986>, form: "author").
  ],
) <fig:gf-lineup>

Over the years, a large number of glottal flow models have been proposed in acoustic phonetics. A sample of these is shown in @fig:gf-lineup, already an impressive lineup by 1986. Despite their visual differences, these models are, essentially, variations on a single theme: they all describe a glottal pulse that tracks the periodic opening and closing of the glottis.

#pcite(<Doval2006>) demonstrated that the most common GFMs#footnote[
  These are LF, KLGLOTT88 @Klatt1990, Rosenberg C @Rosenberg1971, R++ @Veldhuis1998.
]
share a common mathematical structure and can be unified under a single framework;
the $bold(theta)_"LF"$ parameters in @eq:lf-parameters can in fact express all of them.
These GFMs all model a glottal flow $u(t)$ that is positive or null, continuous, and typically differentiable except at the glottal closure instant (GCI). Within each period, $u(t)$ rises during the opening phase, falls during the closing phase, and returns to zero during the closed phase, possibly through a short return phase that smooths the closure. The derivative $u'(t)$ alternates positive, zero, and negative in the expected order and integrates to zero over one period, satisfying the closure constraint @eq:closure-constraint.
It is continuous in case of a nonempty return phase and discontinuous otherwise (meaning a jump at $t_c$ as in @fig:alku), known as hard closure.

==== Allow partial closure?
The strict closure constraint in @eq:closure-constraint is of course just a convenient modeling choice and must be expected to be violated to varying degrees in reality, as real folds often leak, especially in breathy or pathological voices.
On the one hand, it clearly is a defining feature
#footnote[
  The pioneering study on GIF by #pcite(<Miller1959>) had to deconvolve taped speech waveforms with analogue resistor networks and manual optimization.
  A review article on GIF methods notes the part played by the defining quality of the closure constraint in this search for $u(t)$: "the fine-tuning of the resistors of the inverse network was conducted by searching for such values that yielded zero flow after the instant of glottal closure" @Alku2011[p.~626].
]
of the periodic nature of the glottal cycle.
On the other, empirical data show the limits of this hard constraint.
For example, #pcite(<Kreiman2007>, supplement: [p.~601]) observe that
#quote(block: true)[
  in many cases in our data, returning flow derivatives to 0 at the end of the cycle conflicted with the need to match the experimental data and conflicted with the requirement for equal areas under positive and negative curves in the flow derivative. [...] These conflicts between model features, constraints, and the empirical data were handled by abandoning the equal area constraint.#footnote[That "equal area constraint" is just another name for the closure constraint @eq:closure-constraint.]
]
The approach we take in @chapter:qpack is to learn a softened version of the closure constraint @eq:closure-constraint from data rather than set it a priori.
This means we learn both whether the flow typically returns to zero and, if not, to what extent this constraint is plausibly violated.

==== Allow negative flow?
Most GFMs enforce a positive glottal flow and assert $u(t) >= 0$. But there are exceptions such as #pcite(<Fujisaki1986>)/*, also noted in #pcite(<Degottex2010>, supplement: [p.~35])*/. They allow for
transient negative flow which could represent a lowering of the vocal folds after GCI (presumably when air is sucked back into the lungs).
As with the closure constraint, we will learn this effect from data in @chapter:pack and do not forbid $u(t) < 0$ a priori.

== Classic piecewise polynomial models
<sec:classic-polynomial-models>

#figure(
  tablem(
    columns: (0.3fr, 0.8fr, 1.2fr),
    align: left,
    fill: (_, y) => if calc.odd(y) { rgb("#eeeeee89") },
    stroke: frame(rgb("21222C")),
  )[
    | *Degree $d$* | *Name* | *Reference* |
    | ------------- | ------- | ----------- |
    | 0 | Triangular pulse model | #pcite(<Alku2002>) |
    | 1 | — | #pcite(<Titze2000>) #pcite(<Verdolini1995>) |
    | 2 | KLGLOTT88 | #pcite(<Klatt1990>) \ #pcite(<Yang1998>) |
    | 3 | R++ | #pcite(<Veldhuis1998>) |
    | 3 | FL model | #pcite(<Fujisaki1986>) |
    | 6 | — | #pcite(<Childers1994>) |
  ],
  placement: auto,
  caption: [
    *Classic piecewise polynomial models* from the literature.
    Of these, KLGLOTT88 and R++ remain popular today @Doval2006.
  ],
) <table:polys>

/*
derivative degree $d$

0th order: @Alku2002
1th order: @Titze2000 @Verdolini1995
2nd order: KLGLOTT88 @Klatt1990 @Yang1998
3rd order: R++ @Veldhuis1998
3rd order: @Fujisaki1986
6th order: @Childers1994

non-polynomials:
Rosenberg-C: trig (sine) model
LF-model: trig + exp model
Hedelin (1984, ICASSP): not polynomial. It’s a glottal LPC-vocoder using a Rosenberg-type, two-segment trigonometric (cosine) pulse; slope discontinuity at closure, akin to Rosenberg C.
*/

Polynomial GFMs occupy an interesting corner of the "model zoo."
They fit into the same piecewise framework we have been using for the LF model, but with each part modeled as a polynomial of degree $d$.
We list several GFMs proposed in the literature in @table:polys.

This class of GFMs mainly appeal to simplicity and spectral strength.
Piecewise polynomial models admit an analytical solution to the closure constraint while remaining computationally cheap.
More importantly, the sharp transitions generated by low-order polynomials have bright spectra with slow decay, which is exactly what it takes to model hard GCIs @Childers1994 and score well on perceptual tests of synthetic speech @Rosenberg1971.

To get a feel for these models, we now turn to the simplest case: $d = 0$.

/*

We make a case for the old "forgotten" family of polynomial GF models such as @Alku2002 @Verdolini1995 @Doval2006:
- Computationally fast, analytical null flow condition
- Many exist in literature guised in orders $n = 0,1,2,3$
- Capable of very sharp events
- "Bright spectrum": very slow decay, so are excellently placed to excite GF


The modern-day revival of piecewise functions (linear, quadratic, ...) puts these ancient models in a new light. Changepoint modeling ("hard ifs") in the guise of decision surfaces is what drives deep architectures today, and it is exactly the same kind we need for GFs. Plus, these models are already embedded in zero DC line (ie, a polynomial of order 0) as they model only open phase.

There are conventially several changepoints in the glottal cycle to be modeled: the primary changepoints are opening onset and closure instant, with optional landmarks like max flow (maximum of $u(t)$) and closing phase onset (minimum of $u'(t)$, start of return phase) used to quantify shape.

We now look at the simplest of the polynomial models in more detail. We will use this model below as a starting point for our generalization to general polynomials of arbitrary degree $n$ and precision $H$, and finally take the limit $H -> oo$.

*/

=== The triangular pulse model
<sec:triangular-pulse-model>

#figure(
  gnuplot(read("./fig/alku2002.gp")),
  placement: auto,
  caption: [
    *The triangular pulse model* for #box(baseline: -3pt)[#line(length: 1.5em, stroke: 1.3pt + blue)] $u'_Delta (t)$ and #box(baseline: -3pt)[#line(length: 1.5em, stroke: 1.3pt + red)] $u_Delta (t)$ during a single period of length $T$.
    The parameters used in this model are identical to @fig:lf, except for the lack of a return phase and that the 'instant' of maximum excitation $t_e$ now spans the interval $[t_p, t_c]$.
  ],
) <fig:alku>

The simplest polynomial model is the _triangular pulse model_ proposed by #pcite(<Alku2002>).
It takes $u(t)$ piecewise linear and $u'(t)$ piecewise constant, as shown in @fig:alku.
Mathematically, the triangular pulse model is given by:
$
  u'_Delta (t) = cases(
    +E_e (t_c/t_p - 1) quad quad & 0 & < t & <= t_p quad quad & "(O)",
    -E_e quad quad & t_p & < t & <= t_c & "(O)",
    0 quad quad & t_c & < t & <= T & "(C)",
  )
$ <eq:dgf-piece>
The model has no return phase and only four parameters: $bold(theta)_Delta = {E_e, t_p, t_c, T}$.
Note that the amplitude of the first case in @eq:dgf-piece is fixed because the closure constraint @eq:closure-constraint removes one degree of freedom.
#pcite(<Alku2002>) point out that this enables a difficult-to-measure time domain quantity to be expressed as the ratio of two easy-to-measure quantities in the amplitude domain and exploit this fact to measure the open quotient more robustly.


== Parametric piecewise polynomial models
<sec:parametric-piecewise-polynomials>

The triangular pulse model @eq:dgf-piece contains two jumps in the derivative $u'_Delta (t)$, so we can write the latter more compactly as a linear combination of two Heaviside functions
$
  u'_Delta (t) = cases(
    a_1 (t - b_1)_+^0 + a_2 (t - b_2)_+^0 quad quad & 0 & < t & <= t_c quad quad & "(O)",
    0 quad quad & t_c & < t & <= T & "(C)",
  )
$ <eq:alku-lc>
where $(t-b)_+^0 = mono("Heaviside")(t-b)$ and
$
  bm(a) = vec(a_1, a_2) = E_e vec(t_c/t_p - 1, -t_c/t_p), quad quad
  bm(b) = vec(b_1, b_2) = vec(0, t_p).
$
During the open phase, we can interpret the rewritten triangular pulse model as a _parametric piecewise polynomial model_ of _degree_ $d = 0$ and _order_ $H = 2$.
It has parameters $bm(theta)_2 = {bm(a), bm(b)}$ which describe $H = 2$ jumps with amplitudes $bm(a)$ at locations $bm(b)$.
Note that in this formulation the amplitudes $bm(a)$ differ from the piecewise amplitudes in @eq:dgf-piece but are still linearly dependent given the changepoints $bm(b)$ and $t_c$.

Continuing, nothing stops us from generalizing both $H$ and $d$.
The open phase part of @eq:alku-lc becomes:
$
  u'_H (t) = sum_(h=1)^H a_h (t-b_h)_+^d quad quad & 0 & < t & <= t_c quad quad & "(O)"
$ <eq:udH>
This is the general parametric piecewise polynomial model of degree $d$ and order $H$.//, supported on $[0,t_c]$ (the open phase). // Technically the support operation takes the closure of the interval 0 < t <= t_c so this is correct
Again, its parameters $bm(theta)_H = {bm(a), bm(b)}$ are the amplitudes $bm(a) = (a_1, dots, a_H)^top in bb(R)^H$ and the changepoints $bm(b) = (b_1, dots, b_H)^top in bb(R)^H$.
These scale and shift $H$ piecewise monomials of degree $d$:
$
  (t-b)_+^d = (t-b)^d mono("Heaviside")(t-b) = cases(
    0 quad & t < b,
    (t-b)^d quad & t >= b,
  )
$ <eq:repu>
We recognize @eq:repu as the family of activation functions called _rectified power units_ (RePUs).
These include Heaviside for $d = 0$ and the influential ReLU function for $d = 1$, both shown among higher degree variants in @fig:repu.
It may sound like a stretch to relate polynomial GFMs to neural nets, but this view of things will pay off greatly in @sec:nonparametric-piecewise when we take the limit $H -> oo$.

#figure(
  grid(
    align: center,
    row-gutter: { 10pt },
    column-gutter: { 1pt },
    columns: 4,
    $(t)_+^0$, $(t)_+^1$, $(t)_+^2$, $(t)_+^3$,
    [Heaviside], [ReLU], [ReQU], [ReCU],
    gnuplot(read("./fig/repu0.gp")),
    gnuplot(read("./fig/repu1.gp")),
    gnuplot(read("./fig/repu2.gp")),
    gnuplot(read("./fig/repu3.gp")),
  ),
  placement: auto,
  caption: [*The RePU activation functions* #box(baseline: -3pt)[#line(length: 1.5em, stroke: 1.3pt + blue)] $(t)_+^d$ for $d in {0,1,2,3}$. /*After #pcite(<Cho2009>).*/],
) <fig:repu>

=== Regression view
<sec:regression-view>

GFMs ultimately play the role of regression models in the GIF problem.
From this point of view, the general parametric model in @eq:udH is just a standard _linear model_,
and we will now develop it into a Bayesian regression model following #pcite(<MacKay1998>).

Suppose we observe $N$ time samples $bold(t) = {t_n}_(n=1)^N$ during one glottal cycle, and fix $H$ basis functions of the form
$
  phi.alt_h (t) = /*[0 < t <= t_c] times*/ (t - b_h)_+^d,
$ <eq:phi-iverson>
where /*we used an Iverson bracket $[0 < t <= t_c]$ to bake in the compact support during the open phase, and*/ the changepoints $bm(b)$ and $t_c$ are treated as given.
Collecting these into the design matrix $bm(Phi) in bb(R)^(N times H)$ with
$
  [bm(Phi)]_(n h) = phi.alt_h (t_n),
$
the general parametric model during /*both open and closed phase*/ the open phase becomes simply
$
  bm(u') = bm(Phi) bm(a).
$ <eq:udphia>
This is linear in the amplitudes $bm(a)$ given $bm(Phi)$, so the model is rather constrained at this point.
Indeed, $bm(a)$ is currently the only source of variability and power in this model because we assume all other parameters fixed.
Which prior then for $bm(a)$ should we choose?
If we want to maximize model support while adhering to the known additivity and power constraints, the optimal choice
is the canonical Gaussian prior over amplitudes @Bretthorst1988:
$
  p(bm(a) | bm(b)) = mono("Normal")(bm(a) | bm(0), sigma_a^2 bm(I)).
$ <eq:pab>
Moving on to data space, the induced prior probability of $bm(u')$ is also Gaussian,
$
  p(bm(u') | bm(b)) = integral delta(bm(u') - bm(Phi) bm(a)) thin p(bm(a) | bm(b)) dif bm(a) = mono("Normal")(bm(u') | bm(0), sigma_a^2 bm(Phi) bm(Phi)^top),
$ <eq:udashgauss>
due to closure of the Gaussian under linear combinations, and
$
  bb(E)_(bm(a)|bm(b))[bm(u')] & = bm(Phi) thin bb(E)_(bm(a)|bm(b))[bm(a)] = 0, \
  bb(E)_(bm(a)|bm(b))[bm(u') bm(u')^top] & = bm(Phi) thin bb(E)_(bm(a)|bm(b))[bm(a) bm(a)^top] thin bm(Phi)^top = sigma_a^2 bm(Phi) bm(Phi)^top.
$ <eq:abexpval>
Because $"rank"(bm(Phi) bm(Phi)^top) <= H$, prior samples of $bm(u')$ are intrinsically confined to an $H$-dimensional subspace of $bb(R)^N$.
Assuming $H < N$, we thus see that linear models like @eq:udphia are inherently low-rank.
Their expressivity grows with $H$ and saturates when $H$ reaches $N$.

==== Closure constraint
The closure constraint @eq:closure-constraint becomes
$
  integral_0^T u'_H (t) dif t = sum_(h=1)^H a_h integral_0^t_c phi.alt_h (t) dif t = sum_(h=1)^H a_h r_h = 0,
$ <eq:arh-constraint>
with $r_h = (t_c - b_h)^(d+1)\/(d+1)$.
Observe that the prior @eq:pab already satisfies the closure constraint in expectation:
$
  bb(E)_(bm(a)|bm(b)) [sum_(h=1)^H a_h r_h] = sum_(h=1)^H bb(E)_(bm(a)|bm(b))[a_h] thin r_h = 0.
$
This motivates setting the prior mean of the $bm(a)$ to zero, otherwise than symmetry of ignorance around $bm(a) = bm(0)$.
We can also enforce it:#footnote[
  The general solution is given by $argmin_f D_"KL" [f(bm(a)) || p(bm(a) | bm(b))]$ under the constraint @eq:arh-constraint, but the closure under linear conditioning of Gaussian distributions allow us to take a shortcut here.
]
$
  sum_(h=1)^H a_h r_h = 0 ==> p(bm(a) | mono("closure"), bm(b)) = mono("Normal")(bm(a) | bm(0), sigma_a^2 (bm(I) - bm(q) bm(q)^top))
$ <eq:arh-prior>
where $bm(r) = (r_1, dots, r_H)^top$ and $bm(q) = bm(r)/(||bm(r)||)$.
A convenient way to a sample from this updated prior makes use of the fact that this is a rank-one projection:
$
  bm(a)        &= (bm(q) bm(q)^top) bm(a) + &(bm(I) - bm(q) bm(q)^top) bm(a) quad ~ quad & p(bm(a) | bm(b)) \
  ==> bm(a)_perp &= &(bm(I) - bm(q) bm(q)^top) bm(a)                       quad ~ quad  & p(bm(a) | mono("closure"), bm(b)).
$
How does this look in data space?
By the same calculation @eq:abexpval, marginalizing over $bm(a)$ yields
$
  p(bm(u') | mono("closure"), bm(b)) = mono("Normal")(bm(u') | bm(0), sigma_a^2 bm(Phi) (bm(I) - bm(q) bm(q)^top) bm(Phi)^top).
$ <eq:udashani>
We can then obtain $bm(u)$ from $bm(u')$ via integration using @eq:udu.
Compared to samples from the isotropic prior @eq:udashgauss,
the anisotropic prior @eq:udashani induces glottal flows $u(t)$ that tend to look more pulse-like, as shown in @fig:closure.
Indeed, moving to the bottom right corner, it is seen that increasing $H$ and $d$ tends to produce LF-like pulses out of the box!

#let gpfig(path) = box(width: 80pt, height: 60pt, inset: 0pt)[
  #gnuplot(read(path))
]

// small gutter for typical table cells
// large gutter between closure groups
#let small-col-gap = 4pt
#let big-col-gap = 16pt
#let small-row-gap = 3pt

#figure(
  table(
    columns: (auto, auto, auto, auto, 0pt, auto, auto, auto),
    column-gutter: (
      small-col-gap,
      small-col-gap,
      small-col-gap,
      big-col-gap,
      small-col-gap,
      small-col-gap,
      small-col-gap,
    ),
    row-gutter: small-row-gap,
    align: center,
    stroke: none,
    
    // ---------- HEADER ROW 1 ----------
    [],
    table.cell(colspan: 3)[*Without closure constraint*],
    [],
    table.cell(colspan: 3)[*With closure constraint*],
    
    // ---------- HEADER ROW 2 ----------
    [],
    [$H = 10$], [$H = 100$], [$H = 1000$],
    [],
    [$H = 10$], [$H = 100$], [$H = 1000$],
    
    // ---------- d = 0 ----------
    table.cell(align: center + horizon)[
      #rotate(-90deg, reflow: true)[$d = 0$]
    ],
    gpfig("./fig/closure/closure=0_H=10_d=0.gp"),
    gpfig("./fig/closure/closure=0_H=100_d=0.gp"),
    gpfig("./fig/closure/closure=0_H=1000_d=0.gp"),
    [],
    gpfig("./fig/closure/closure=1_H=10_d=0.gp"),
    gpfig("./fig/closure/closure=1_H=100_d=0.gp"),
    gpfig("./fig/closure/closure=1_H=1000_d=0.gp"),
    
    // ---------- d = 1 ----------
    table.cell(align: center + horizon)[
      #rotate(-90deg, reflow: true)[$d = 1$]
    ],
    gpfig("./fig/closure/closure=0_H=10_d=1.gp"),
    gpfig("./fig/closure/closure=0_H=100_d=1.gp"),
    gpfig("./fig/closure/closure=0_H=1000_d=1.gp"),
    [],
    gpfig("./fig/closure/closure=1_H=10_d=1.gp"),
    gpfig("./fig/closure/closure=1_H=100_d=1.gp"),
    gpfig("./fig/closure/closure=1_H=1000_d=1.gp"),
    
    // ---------- d = 2 ----------
    table.cell(align: center + horizon)[
      #rotate(-90deg, reflow: true)[$d = 2$]
    ],
    gpfig("./fig/closure/closure=0_H=10_d=2.gp"),
    gpfig("./fig/closure/closure=0_H=100_d=2.gp"),
    gpfig("./fig/closure/closure=0_H=1000_d=2.gp"),
    [],
    gpfig("./fig/closure/closure=1_H=10_d=2.gp"),
    gpfig("./fig/closure/closure=1_H=100_d=2.gp"),
    gpfig("./fig/closure/closure=1_H=1000_d=2.gp"),
    
    // ---------- d = 3 ----------
    table.cell(align: center + horizon)[
      #rotate(-90deg, reflow: true)[$d = 3$]
    ],
    gpfig("./fig/closure/closure=0_H=10_d=3.gp"),
    gpfig("./fig/closure/closure=0_H=100_d=3.gp"),
    gpfig("./fig/closure/closure=0_H=1000_d=3.gp"),
    [],
    gpfig("./fig/closure/closure=1_H=10_d=3.gp"),
    gpfig("./fig/closure/closure=1_H=100_d=3.gp"),
    gpfig("./fig/closure/closure=1_H=1000_d=3.gp"),
  ),
  placement: auto,
  caption: [
    *Samples of $u(t)$* from the general parametric piecewise polynomial model @eq:udH
    with or without the closure constraint @eq:arh-constraint.
    For each combination of degree $d$ and order $H$,
    a total of $H$ changepoints $b_h ~ mono("Uniform")(0,t_c)$ were drawn randomly,
    and four instances of $bm(u')$ conditioned on these were sampled either from @eq:udashgauss [*Without closure constraint*] or from @eq:udashani [*With closure constraint*].
    Then $u(t) = integral^t_0 u'(tau) dif tau$ was obtained by quadrature of $bm(u')$.
  ],
) <fig:closure>

Regression with @eq:udashani ensures that posterior mass vanishes at solutions $bm(u)$ that violate the closure constraint.
This illustrates how linear constraints on the amplitudes $bm(a)$ in the linear model @eq:udphia can encode GFM properties at the cost of just a single rank-one downdate per constraint.
Moving on from linear models to Gaussian processes, these linear constraints become linear functionals known as _interdomain features_, which may be used to impose structure directly in function space, without touching rank, but more challenging mathematically.
In @chapter:pack, we will use the interdomain approach to learn spectral features of nonparametric GFMs directly from data rather than hardcoding them as in @eq:arh-constraint.

==== Connection to classic polynomial models
Observe that the analytical solution @eq:arh-prior to the closure constraint is valid for any degree $d >= 0$ and order $H >= 1$, so each classic polynomial model listed in @table:polys can be expressed in our linear model framework given suitable choices of $bm(b)$ and $t_c$, without specialized derivations.

=== Neural network equivalence
<sec:connection-to-neural-networks>

#figure(
  grid(
    columns: 2,
    column-gutter: { 4em },
    row-gutter: { 1em },
    align: center + bottom,
    [#include "./fig/nn-polynomial-a.typ"], [#include "./fig/nn-polynomial-b.typ"],
    [(a)], [(b)],
  ),
  placement: auto,
  kind: image,
  caption: [
    *Neural network view.*
    (a) The triangular pulse model as a tiny neural network with a single hidden layer, linear readout and $t^0_+$ activation.
    (b) The general parametric polynomial model of degree $d$ with order $H$ with hidden weights ${bm(b), bm(c)} in bb(R)^(2 H)$, output weights $bm(a) in bb(R)^H$ and RePU activation function $t_+^d$.
  ],
  gap: 1em,
) <fig:nn-polynomial>

@fig:nn-polynomial illustrates that the triangular pulse model of @sec:triangular-pulse-model can be seen as a tiny neural network.
Likewise, the general parametric piecewise polynomial model @eq:udH can be recast as a single hidden layer of width $H$ with a RePU activation and a linear readout:
$
  u'_"NN" (t) = sum_(h=1)^H a_h (c_h t - b_h)_+^d.
$ <eq:uNN>
Each unit computes $phi.alt_h (t) = (bm(w)_h^top bm(x)_t)_+^d$ with hidden weights $bm(w)_h = (c_h, -b_h)^top$ and $bm(x)_t = (1, t)^top$, exactly like a neuron with a bias input.
The $3H$ parameters of the neural net are thus $bm(a)$ and $bm(b)$ (as before) and the newly acquired degrees of freedom $bm(c) = (c_1, dots, c_H)^top$.

Seen from this point of view, a natural symmetry appears.
We argued in @sec:regression-view that in linear regression context the output weights $bm(a)$ are best assigned an uninformative Gaussian prior, so
$
  p(bm(a)) = mono("Normal")(bm(a) & | bm(0), sigma_a^2 bm(I)).
$ <eq:priora>
But the neural network equivalence makes very clear that the hidden weights ${bm(b), bm(c)}$ actually play similar roles:
they enter linearly inside the activation, so the same argument applies, and we are encouraged to assign Gaussians again:
$
  p(bm(b)) = mono("Normal")(bm(b) | bm(0), sigma_b^2 bm(I)), quad p(bm(c)) = mono("Normal")(bm(c) | bm(0), sigma_c^2 bm(I)).
$ <eq:priorbc>
This step is not obvious from the viewpoint of any of the GFMs we've discussed before, where changepoints $bm(b)$ were nonlinear hyperparameters fixed a priori; here they become "active" and "linear" degrees of freedom.
Assuming $t_c$ is known, the remaining hyperparameters to describe the open phase are thus the three scales $bold(theta)_"NN" = {sigma_a, sigma_b, sigma_c}$ controlling amplitude, typical changepoint location, and horizontal stretch, respectively.

Turning to data space again and having assigned priors to all parameters, we can now do a full marginalization unlike the conditional in @eq:udashgauss.
Updating the design matrix $bm(Phi)$ to
$
  [bm(Phi)]_(n h) = phi.alt_h (t_n) = (c_h t_n - b_h)_+^d,
$ <eq:nnphi>
and assuming the isotropic priors @eq:priora and @eq:priorbc
we have
$
  p(bm(u'))
  &= integral delta(bm(u') - bm(Phi) bm(a)) thin p(bm(a)) thin p(bm(b)) thin p(bm(c)) dif bm(a) dif bm(b) dif bm(c) \
  &= integral mono("Normal")(bm(u') | bm(0), sigma_a^2 bm(Phi) bm(Phi)^top) thin p(bm(b)) thin p(bm(c)) dif bm(b) dif bm(c).
$ <eq:pu-gaussmixture>
This is a Gaussian mixture with varying covariances (not means) as $bm(Phi)$ depends on the hidden weights ${bm(b), bm(c)}$.
//It is easy to sample from @eq:pu-gaussmixture, but
A closed form for this integral exists only for $H -> oo$ as $p(bm(u'))$ converges to a Gaussian process, as we will see in @sec:nonparametric-piecewise.

==== Closure constraint
The closure constraint @eq:closure-constraint given ${bm(b), bm(c)}$ remains a linear condition on $bm(a)$, so we can define
$
  p(bm(a), bm(b), bm(c) | mono("closure")) = p(bm(a) | mono("closure"), bm(b), bm(c)) thin p(bm(b)) thin p(bm(c)),
$ <eq:abc-closure>
where $p(bm(a) | mono("closure"), bm(b), bm(c))$ is identical to @eq:arh-prior but with $r_h = (c_h t_c - b_h)_+^(d+1) \/ c_h (d+1)$.
/* $ r_h = integral_0^T phi.alt_h (t) dif t = integral_0^(t_c) (c_h t - b_h)_+^d dif t = (c_h t_c - b_h)_+^(d+1) \/ c_h (d+1)$. */
This prior yields a mixture in data space that is similar to @eq:pu-gaussmixture:
$
  p(bm(u') | mono("closure")) = integral mono("Normal")(bm(u') | bm(0), sigma_a^2 bm(Phi) (bm(I) - bm(q) bm(q)^top) bm(Phi)^top) thin p(bm(b)) thin p(bm(c)) dif bm(b) dif bm(c).
$ <eq:ccbmu>
Here both $bm(q) = bm(r)/(||bm(r)||)$ and $bm(Phi)$ depend on ${bm(b), bm(c)}$.
This integral has no closed form for $H -> oo$, but we can still impose the closure constraint on @eq:pu-gaussmixture indirectly in the spectral domain via interdomain features (@chapter:pack).

/*
TODO: Insert picture of samples from neural net model with d = 0, 1, 2, 3 H = small and large, with and without closure constraint. Parameters sampled from their priors.
*/

==== Connection to LF and other GFMs
The neural network view of the parametric polynomial model also clarifies expressivity.
It is well known that a single RePU layer of sufficient width can approximate any continuous waveform on a bounded interval @Hornik1993.
Therefore, with its analytical closure constraint @eq:abc-closure and the ability to represent sharp changepoint behavior, the parametric piecewise polynomial model in the form @eq:uNN effectively unifies the entire family of GFMs encountered in the literature, from triangular pulses to LF-type exponentials and sinusoids, given /*a precision tolerance and a*/ sufficiently large $H$.



== Nonparametric piecewise polynomial models
<sec:nonparametric-piecewise>

The correspondence of piecewise polynomials GFMs with a single hidden layer shows that their expressivity depends mostly on the order $H$, as model order is equivalent to layer width in the neural network picture.
Indeed, _parametric_ linear models have a finite representation capacity determined primarily by rank because the modeled function can only explore the subspace spanned by the basisfunctions.
#footnote[
  More precisely: a higher rank naturally enlarges the subspace of attainable functions, but the prior over amplitudes still constrains where probability mass is placed within that subspace.
Geometrically, the model explores an $H$-dimensional ellipsoid like @eq:udashgauss or @eq:udashani rather than the entire hyperplane.
When data accumulate, the prior constraints are eventually overwhelmed and the model can move freely on that plane.
Random kitchen sinks @Rahimi2008 take these two ideas to the extreme: forgo specialized modeling; just use $H$ _random_ basis functions with uninformative priors and rely on the fact that an entire $H$-dimensional hyperplane is reachable with high probability.
]
_Nonparametric_ models grow capacity as the number of datapoints increases;
in the case of Gaussian processes, posterior inference given $N$ datapoints is equivalent to Bayesian linear regression with $N$ linearly independent basisfunctions, hence full-rank @Rasmussen2006.

It thus makes sense to consider the limit $H -> oo$ in our search for more expressive GFMs. //,  for the neural network model @eq:uNN in data space.
Continuing where we left, we already saw in @eq:pu-gaussmixture that $p(bm(u'))$ is a nontrivial Gaussian covariance mixture weighed by the isotropic priors defined in @eq:priorbc.
This can be made explicit:
$
  p(bm(u'))
  &= integral mono("Normal")(bm(u') | bm(0), sigma_a^2 bm(Phi) bm(Phi)^top)
     thin p(bm(b)) thin p(bm(c)) dif bm(b) dif bm(c) \
  &= integral mono("Normal")(bm(u') | bm(0), bm(Q))
     thin p(bm(Q)) dif bm(Q),
$ <eq:uprime-mix>
where we defined the push forward
$
  p(bm(Q)) = integral delta(bm(Q) - sigma_a^2 bm(Phi) bm(Phi)^top) thin p(bm(b)) thin p(bm(c)) dif bm(b) dif bm(c)
$ <eq:Q-push-forward>
which essentially counts the number of ${bm(b), bm(c)}$ configurations that give rise to a given $bm(Q)$.
#share-align[
  Thus each random sample of ${bm(b), bm(c)} in bb(R)^(2H)$ drawn from @eq:priorbc induces in data space a random covariance matrix $bm(Q) in bb(R)^(N times N)$,
  $ 
    {bm(b), bm(c)} mapsto bm(Q) = sigma_a^2 bm(Phi) bm(Phi), quad quad &"rank"(bm(Q)) &<= H&
  $ <eq:define-Q>
  and hence a Gaussian component with weight $p(bm(Q))$ in the mixture @eq:uprime-mix.
  Below we will show that by choosing $sigma_a prop 1\/sqrt(H)$ and letting $H -> oo$, the density of states @eq:Q-push-forward degenerates to
  $ 
   p(bm(Q)) --> delta(bm(Q) - bm(K)), quad quad &"rank"(bm(K)) &= N& "almost surely",
  $
  for some $bm(K) in bb(R)^(N times N)$ derived below in @eq:derive-K, causing the mixture to collapse to a single Gaussian and thus completing the transition to an indexed Gaussian process with covariance matrix $bm(K)$.
]

=== Mean and covariance description

While the mixture @eq:uprime-mix has no closed form for finite $H$, progress can be made by computing its mean and covariance as before in @eq:abexpval:
$
  bb(E)[bm(u')]
  &= bb(E)_(bm(b),bm(c))[bb(E)_(bm(a)|bm(b),bm(c))[bm(u')]]
  = bm(0), \
  bb(E)[bm(u') bm(u')^top]
  &= bb(E)_(bm(b),bm(c))[bb(E)_(bm(a)|bm(b),bm(c))[bm(u') bm(u')^top]]
  = sigma_a^2 thin bb(E)_(bm(b),bm(c))[bm(Phi) bm(Phi)^top],
$ <eq:covexpval>
The mean is zero everywhere due to our zero-mean prior $p(bm(a))$.
The covariance matrix, however, is nontrivial.
#share-align[
To investigate, define
$
  bm(C) &:= bb(E)[bm(u') bm(u')^top] in bb(R)^(N times N) 
$
with elements
$
  [bm(C)]_(n m)
  &= sigma_a^2 thin bb(E)_(bm(b),bm(c)) [sum_(h=1)^H phi.alt_h (t_n) phi.alt_h (t_m)],
$
]
where $phi.alt_h (t) = (c_h t - b_h)_+^d $ for the neural network model $u'_"NN" (t)$ was defined in @eq:nnphi.
Since the $phi.alt_h (t)$ are statistically equivalent under i.i.d. priors like the ones we adopted in @eq:priorbc, define a single "mother wavelet" _without index $h$_ as
$
  phi.alt(t\; b, c) = (c t - b)_+^d, &quad b ~ mono("Normal")(0, sigma_b^2), quad c ~ mono("Normal")(0, sigma_c^2).
$ <eq:motherwavelet>
Then the elements of $bm(Phi) bm(Phi)^top$ become sums of $H$ i.i.d. random variables, and their expectation factorizes across $h$:
$
  [bm(C)]_(n m)
  &= sigma_a^2 thin sum_(h=1)^H bb(E)_(bm(b),bm(c)) [phi.alt_h (t_n) phi.alt_h (t_m)] \
  &= sigma_a^2 thin sum_(h=1)^H bb(E)_(b, c) [phi.alt(t_n\; b, c) phi.alt(t_m\; b, c)] \
  &= sigma_a^2 thin H thin bb(E)_(b,c) [phi.alt(t_n\; b, c) phi.alt(t_m\; b, c)] \
  &= sigma_a^2 thin H thin k^((d))_bm(Sigma) (t_n, t_m).
$ <eq:Ktack>
Here we anticipate the _temporal arc cosine kernel_ of degree $d$, defined below in @eq:tack:
$
  k^((d))_bm(Sigma) (t, t')
  &:= bb(E)_(b,c) [phi.alt(t\; b, c) phi.alt(t'\; b, c)] \
  &equiv integral phi.alt(t\; bm(w)) phi.alt(t'\; bm(w)) mono("Normal")(bm(w) | bm(0), bm(Sigma)) dif bm(w),
$ <eq:tack-e>
where $bm(Sigma) := mat(sigma_b^2, 0; 0, sigma_c^2)$ describes the covariance of the hidden weights $bm(w) = (-b, c)^top$.

To recapitulate, the first and second central moments of the mixture @eq:uprime-mix are known for any $H$:
$
  bb(E)[bm(u')]
  &= bm(0), \
  bb(E)[bm(u') bm(u')^top]
  &equiv bm(C) = sigma_a^2 thin H thin k^((d))_bm(Sigma) (t, t').
$ <eq:firstandsecondmoments>
The third, fourth, ... central moments are in general nonzero and difficult to compute.
They will vanish, however, when $H -> oo$ and we choose $sigma_a prop 1\/sqrt(H)$.
Nevertheless, even before taking any limit, the first two moments of the prior already mirror a kernel regression model.
Note that this result hinges critically on the independence assumption for ${b_h, c_h}$ in @eq:motherwavelet.
The closure-constrained prior of @eq:ccbmu would couple these parameters nonlinearly, destroying that independence and making the derivation intractable, which is why we temporarily set it aside.
// footnote: @Neal1996 suggested non idd priors to effectuate possibly more interesting kernels

=== The temporal arc cosine kernel
<sec:temporal-ack>

Of course, I wouldn't have had the courage to attempt this whole calculation had I not already known that a very similar limit has been evaluated succesfully in the neural-network-as-a-Gaussian-process literature. 
The marginalization we performed above is, in fact, a one-dimensional variant of the classic infinite-width limit of feedforward networks studied by #pcite(<Neal1996>) #pcite(<Williams1998>) and later generalized by #pcite(<Cho2009>). 

In these works, the expectation over random weights $bm(w)$ with $mono("Normal")(bm(w) | bm(0), bm(Sigma))$ priors gives rise to a family of kernels which describe an infinitely wide Bayesian network,
and which differ only with respect to the activation function used and the imposed a priori covariance $bm(Sigma)$.
This same reasoning will now allow us to identify our expected covariance @eq:tack-e with a time-domain specialization of the _arc cosine kernel family_ of degree $d$.

@table:acks organizes the three kernels we will introduce shortly.
Note that to save on mathematical symbols we overload the $k^((d))(dot,dot)$ symbol to change meaning based on the type of its arguments, as we have been doing with the probability density $p(dot)$.

#figure(
  tablem(
    columns: (.9fr, 0.35fr, 0.35fr, 1.1fr, 0.2fr),
    align: left,
    fill: (_, y) => if calc.odd(y) { rgb("#eeeeee89") },
    stroke: frame(rgb("21222C")),
  )[
    | *Name* | | *Form* |  | *Eq.* |
    | ------------- | ------- | ----------- | --------- | --- |
    | arc cosine kernel | ACK | $k^((d)) (bm(x), bm(x'))$ | $= 1/pi ||bm(x)||^d ||bm(x')||^d J_d (theta)$ | @eq:ack |
    | temporal arc cosine kernel | TACK | $k^((d))_bm(Sigma) (t, t')$ | $= 1/(2pi) ||bm(Sigma)^(1/2) bm(x)_t||^d ||bm(Sigma)^(1/2) bm(x)_t'||^d J_d (theta)$ | @eq:tack |
    | standard arc cosine kernel | STACK | $k^((d)) (t, t')$ | $= 1/(2pi) (1+t^2)^(d/2) (1+t'^2)^(d/2) J_d (theta)$ | @eq:stack |
  ],
  placement: auto,
  caption: [
    *Arc cosine kernels* of order $d$ introduced in this section.
    The arc cosine kernel (ACK) defined on arbitrary input dimension $D >= 1$ was originally proposed by #pcite(<Cho2009>).
    The TACK and STACK are simple modifications of ACK for 1D time series.
  ],
) <table:acks>

==== The arc cosine kernel (ACK)
of degree $d$ proposed by #pcite(<Cho2009>) is defined as the expected inner product between infinite-dimensional random RePU features of $D$-dimensional inputs $bm(x), bm(x') in bb(R)^D$:
$
  k^((d)) (bm(x), bm(x')) 
  = 2 integral (bm(w)^top bm(x))_+^d (bm(w)^top bm(x'))_+^d 
    mono("Normal")(bm(w) | bm(0), bm(I)_D) dif bm(w),
$
where the hidden weights $bm(w) in R^D$, and $bm(I)_D$ is the $D times D$ identity matrix.
This integral representation makes it clear that this is indeed a valid positive definite kernel in any input dimension $D >= 1$.
It admits a beautiful closed form:
$
  k^((d)) (bm(x), bm(x')) 
  = 1/pi ||bm(x)||^d ||bm(x')||^d J_d (theta),
$ <eq:ack>
where
$
  theta = arccos ((bm(x)^top bm(x'))/(||bm(x)|| ||bm(x')||)) in [0, pi]
$
is the positive angle between $bm(x)$ and $bm(x')$, and $J_d (theta)$ is given by #pcite(<Cho2009>, supplement: [Eq.~4]) as the generator expression
$
  J_d (theta) = (-1)^d (sin theta)^(2d + 1)
  ( 1 / (sin theta) dif/(dif theta) )^d
  ( (pi - theta) / (sin theta) ).
$
The first few instances of which are
$
  J_0(theta) & = pi - theta, \
  J_1(theta) & = sin theta + (pi - theta) cos theta, \
  J_2(theta) & = 3 sin theta cos theta + (pi - theta)(1 + 2 cos^2 theta), \
  J_3(theta) & = 15 sin theta - 11 sin^3 theta + (pi - theta)(9 cos theta + 6 cos^3 theta).
$
Compared to the familiar Matérn kernels, the arc-cosine kernel is a rather atypical kernel, and probably only sparingly used in machine learning applications.
It is, for one, blatantly nonstationary, and unlike stationary kernels such as the RBF, it can represent asymptotic behavior: its posterior mean need not revert to the prior mean outside the data regime, much like a neural network’s response.
The closed form @eq:ack shows this structure explicitly: the integral representation factors into a polynomial term $||x||^d ||x'||^d$, which controls the overall dynamic range and absorbs asymptotic variance, and an angular term $J_d (theta)$, which captures the nonlinear thresholding of the RePU activation and thus allows for modeling the kind of changepoint structure we know is essential for GFMs.

==== The temporal arc cosine kernel (TACK)
is the kernel which describes $bm(C) equiv bb(E)[bm(u') bm(u')^top]$ in @eq:covexpval.
We can cast it as an affine shift of the bias augmented ACK on the input dimension $D = 2$.
Define the auxiliary vectors
$
  bm(x)_t = vec(1, t), quad
  bm(w) = vec(-b, c).
$
and the covariance matrix for the hidden weights ${b, c}$
$
  bm(Sigma) = mat(sigma_b^2, 0; 0, sigma_c^2).
$
Then, by substituting these into @eq:ack and integrating with respect to the Gaussian weight prior $mono("Normal")(bm(w) | bm(0), bm(Sigma))$, we obtain the TACK:
$
  k^((d))_bm(Sigma) (t, t')
  &= bb(E)_(b,c) [phi.alt(t\; b, c) phi.alt(t'\; b, c)] \
  &= integral (bm(w)^top bm(x)_t)_+^d (bm(w)^top bm(x)_t')_+^d 
     mono("Normal")(bm(w) | bm(0), bm(Sigma)) dif bm(w) \
  &= 1/2 thin k^((d)) (bm(Sigma)^(1/2) bm(x)_t, bm(Sigma)^(1/2) bm(x)_t') \
  &= 1/(2pi) ||bm(Sigma)^(1/2) bm(x)_t||^d ||bm(Sigma)^(1/2) bm(x)_t'||^d J_d (theta),
$ <eq:tack>
where the factor $1\/2$ compensates for the convention used in #pcite(<Cho2009>).
Here,
$
  theta = arccos((bm(x)_t^top bm(Sigma) bm(x)_t')/
          (sqrt(bm(x)_t^top bm(Sigma) bm(x)_t) sqrt(bm(x)_t'^top bm(Sigma) bm(x)_t'))),
$
is the angle between $bm(x)_t$ and $bm(x)_t'$ in the geometry induced by $bm(Sigma)$.
This anisotropic variant was also used in the context of $D $-dimensional feature extraction by #pcite(<Pandey2014>).

==== The standard temporal arc cosine kernel (STACK)
simplifies the TACK with the particular choice
$bm(Sigma) := bm(I)_2$, which yields
$
  k^((d)) (t, t')
  &≡ k^((d))_bm(I)_2 (t, t') \
  &= 1/(2pi) (1+t^2)^(d/2) (1+t'^2)^(d/2) J_d (theta),
$ <eq:stack>
where
$
  theta = arccos ((1 + t t')/(sqrt(1 + t^2) sqrt(1 + t'^2))).
$
/*
Unlike in the original derivation of @Williams1998, we do not require the hidden-unit activation $(c t - b)_+^d$ to be bounded.
It is sufficient that its output has finite variance under the Gaussian weight prior, as shown by @Matthews2018.
For $(c t - b)_+^d$ with $b, c ~ N(0, sigma^2)$, this condition holds for any finite degree $d$ and time $t$, 
so the infinite-width limit indeed yields the degree-$d$ arc-cosine kernel of @Cho2009.
*/

=== Taking the Gaussian-process limit

We turn again to the mixture @eq:uprime-mix.
Recall from @eq:define-Q that each random draw ${bm(b), bm(c)} in bb(R)^(2H)$ from the prior @eq:priorbc produces a random covariance matrix $bm(Q) in bb(R)^(N times N)$,
which elementwise decomposes as a sum of $H$ i.i.d. terms:
$
  [bm(Q)]_(n m) = sigma_a^2 sum_(h=1)^H phi.alt_h (t_n) phi.alt_h (t_m).
$ <eq:bmq83>
This sum tends to grow as $O(H)$, since from @eq:Ktack
$
  bb(E)_(bm(b),bm(c))[bm(Q)] = bm(C) prop H.
$ <eq:scales-as-H>
Such growth of the marginal variance of $p(bm(u'))$ makes the $u'_"NN" (t)$ model @eq:uNN useless as a prior when we want strong inductive bias even for large $H$, which we do.
Therefore, we couple $sigma_a$ and $H$ by choosing $sigma_a prop 1\/sqrt(H)$, in accordance with the variance-preserving initialization principle @Glorot2010.
Thus, substituting $sigma_a -> sigma_a\/sqrt(H)$ in @eq:bmq83, we get
$
  [bm(Q)]_(n m) = sigma_a^2 times [1/H sum_(h=1)^H phi.alt_h (t_n) phi.alt_h (t_m)].
$
Observe that the quantity between brackets is a Monte Carlo estimate of the TACK @eq:tack.
#share-align[
Therefore, by the strong law of large numbers we conclude that
$
  [1/H sum_(h=1)^H phi.alt_h (t_n) phi.alt_h (t_m)] &--> bb(E)_(b,c) [phi.alt(t\; b, c) phi.alt(t'\; b, c)] quad quad &"as" H -> oo, 
$ <eq:dev86>
with deviations vanishing as $O(1\/sqrt(H))$.
Therefore from @eq:tack-e
$
  [bm(Q)]_(n m) &--> sigma_a^2 thin k^((d))_bm(Sigma) (t_n, t_m) quad quad &"as" H -> oo.
$
Equivalently, the induced density $p(bm(Q))$ of covariance matrices collapses to
$
  p(bm(Q)) &--> delta(bm(Q) - bm(K)) quad quad &"as" H -> oo,
$
which in turn collapses the mixture @eq:uprime-mix to
$
  p(bm(u')) &--> mono("Normal")(bm(0), bm(K)) quad quad &"as" H -> oo.
$
]
Here the _kernel matrix_ is the symmetric PSD matrix $bm(K) in bb(R)^(N times N) prop bm(C)$ given by
$
  [bm(K)]_(n m) = sigma_a^2 thin k^((d))_bm(Sigma) (t_n, t_m).
$ <eq:derive-K>

In other words, the marginal $p(bm(u'))$ converges entirely to a Gaussian and the first and second moments in @eq:firstandsecondmoments become _sufficient statistics_ @Jaynes2003.
All higher central moments vanish as $O(1\/sqrt(H))$ because the covariance fluctuations in @eq:dev86 themselves decay at that rate.
Since this argument holds for any finite collection of times $bold(t) = {t_n}_(n=1)^N$, the limiting process is indeed officially a Gaussian process:
$
  u'_"NN" (t) ~ mono("GaussianProcess")(0, sigma_a^2 thin k^((d))_bm(Sigma) (t, t')).
$ <eq:kgfm>
/*
The function family thereby reaches full rank for any $N$ in data space and becomes a nonparametric process.
*/
Except for $H$, it has the same hyperparameters as the neural network model of @sec:connection-to-neural-networks: the scales $bold(theta)_"NN" = {sigma_a, sigma_b, sigma_c}$ describing overall power, typical changepoint location and horizontal stretch, respectively;
and the two higher-level hyperparameters ${d, t_c}$, which determine the polynomial degree and the instant of glottal closure, respectively.

/* TODO: show samples from this GP together with Model comparison figure with NS */

==== Integrating out changepoints
Note that $t_c$ is the only changepoint parameter that survived the Gaussian-process limit.
All other changepoints [each changepoint being defined by a pair of amplitude and location parameters ${a_h, b_h}$] have been analytically integrated out.
That act of marginalization removed the parametric bottleneck carried by finite rank models like @eq:lf, @eq:dgf-piece, @eq:udH, @eq:uNN completely.
This may come as a surprise, because we pointed out before that changepoint parameters as in @eq:lf-parameters constitute the main "control points" of any GFM expressed in the time domain:
what previously required a discrete arrangement of control points is now averaged into a single continuous covariance function capable of learning the
"timing relationships [which] are very important for modelling the glottal flow signal" #cite(<Doval2006>, supplement: [p.~1]).
//Even more surprising, perhaps, is that we can still salvage the closure constraint (@chapter:pack).

==== More aspects of @eq:kgfm
are discussed in greater detail in @chapter:nonparametric-gfm.

/*
This is the exact finite case analogue of the GP: nonparametric, so infinite rank: support everywhere
but priors on the amplitudes determine where on the hyperplane we can reach, and this in GP land is projected as the "kernel character": how smooth, stationary, etc.
Interestingly, when projecting the GP back to the reduced rank case, again a linear model is found and the amplitudes again determine the character to great extent, encoding properties like closure constraint, differentiability, and spectral properties.
*/

/*

Here the closure constraint becomes analytically "intractable" at first sight, but can be done for SqExp model analytically, and Matern models via Matern expansion trick @Tronarp2018

*/

/*
TODO:

Show kernel support for various GF models without closure constraint

Calculate p(D|k) for k in (arccos, matern, PeriodicSqExp, spline) <= nonparametric
Perhaps also a Bayesian network <= parametric

And D is a single open phase of poly (orders <= 3) and nonpoly (order = infty) models

See which kernel or model has most support

Also show # params and compute time for each

*/

== Summary

Great care was taken to argue that any _parametric_ glottal flow model of the open phase of the glottal cycle can be expressed as a RePU network with a single hidden layer of width $H$, given $H$ large enough.

Then we showed that in a Bayesian regression context with noninformative priors, the limit $H -> oo$ corresponds to a _nonparametric_ glottal flow model:
a zero-mean GP with the temporal arc cosine kernel.
This well-known limit, borrowed from classical GP literature, is applied here to unify the two dominant approaches to GIF.
These are [parametric $<=>$ #link(<sec:joint-source-filter-methods>)[joint source-filter estimation]] and [nonparametric $<=>$ #link(<sec:inverse-filtering-methods>)[inverse filtering]], and we attempt here to combine the best of both worlds: parametric interpretability and nonparametric expressivity, respectively.

That is, the proposed probabilistic model has support for any parametric glottal flow model while retaining capacity to "let the data speak for itself" if need be.
In the next chapters we will refine this initially tiny support gradually into a strong inductive bias by learning features from synthetic glottal flow simulations.