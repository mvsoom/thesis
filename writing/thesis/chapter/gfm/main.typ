#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

= From parametric to nonparametric glottal flow models
<chapter:gfm>

Glottal flow models (GFMs) describe the source signal $u(t)$ as it drives the vocal tract during voiced speech.
These models operate in the time domain because the delicate phase characteristics of the _glottal cycle_ are an integral part of vocal communication.#footnote[
  For example, plosives: glottal stops are micro-events at the millisecond level that underlie semantic information two orders of magnitude higher up the scale.
  The timing of the vocal fold movement is so precisely controlled by our brain that the slightest deviations are studied as biomarkers for neurodegenerative diseases like Parkinson's @Ma2020.
]

A large body of work and decades of empirical work @Degottex2010 have produced a "model zoo" of GFMs: handcrafted waveforms of $u(t)$ with handfuls of carefully chosen parameters.
Because these are _parametric_ models, they all share the same basic traits: parsimonious, interpretable, but inevitably stiff @Schleusing2012. Their parametric nature limits the range of time domain information they can encode.
This motivates the construction of a family of _nonparametric_ models, which reach for greater expressivity with _less_ parameters.

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
Importantly, it is not at maximum glottis aperture that the acoustic output is greatest; it is at the sudden _glottal closure instant_ (GCI).
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
  We choose to keep $t_c$ distinct from $T$ explicitly because this becomes important for short return phases and is experimentally observed to be important, especially in clinical scenarios @Kreiman2007.
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
As noted by #pcite(<Perrotin2021>), simplified LF models exist that are perceptually equivalent to LF and easier to work with. Perhaps authors should retire LF altogether.

Nevertheless, it remains the de facto standard in parametric source-filter modeling and we therefore use it here. Our code in @chapter:jaxlf implements the LF equations in JAX @Bradbury2020, so the bisection routines are compiled to native code for a speedup and the entire model is differentiable and easily batcheable.
The code also implements four LF parametrizations, including the widely used $R_d$ formulation of #pcite(<Fant1995>).

=== One of many: the glottal flow “model zoo”

#figure(
  image("./fig/gf-models.png", width: 100%),
  placement: bottom,
  caption: [
    *A lineup of GF models* back from 1986 (handdrawn?), together with their derivatives (DGFs). These models are visually very similar. From #cite(<Fujisaki1986>, form: "author").
  ],
) <fig:gf-lineup>

Over the years, a large number of glottal flow models have been proposed in acoustic phonetics. A sample of these is shown in @fig:gf-lineup, already an impressive lineup by 1986. Despite their visual differences, these models are, at heart, variations on a single theme: they all describe a quasi-periodic glottal pulse that opens, closes, and repeats with subtle asymmetries and smooth transitions.

#pcite(<Doval2006>) demonstrated that the most useful and physically plausible GFMs—such as the Rosenberg C, KLGLOTT88, R++, and LF models—share a common mathematical structure and can be unified under a single framework. All of them model a glottal flow $u(t)$ that is positive or null, continuous, and typically differentiable except at the glottal closure instant (GCI). Within each period, $u(t)$ rises during the opening phase, falls during the closing phase, and returns to zero during the closed phase, possibly through a short return phase that smooths the closure. The derivative $u'(t)$ alternates positive, zero, and negative in the expected order and integrates to zero over one period, satisfying the closure constraint @eq:closure-constraint.

When viewed through this lens, the differences between models are superficial. Each expresses the same physical process using slightly different analytic forms—exponentials, sinusoids, or polynomial pieces—but all encode the same bell-shaped glottal pulse with a characteristic asymmetry and decay. In fact, nearly every GFM can be reparameterized in the terms introduced for the LF model: excitation amplitude $E_e$, open quotient, asymmetry, return-phase time constant, and fundamental period. These few parameters control the timing, smoothness, and efficiency of vocal fold motion and thereby the spectral shape of the resulting voice source.

==== Allow partial closure?
The strict closure constraint in @eq:closure-constraint—that $u(t)$ must return exactly to zero each cycle—is seldom realistic. As #pcite(<Kreiman2007>) observe:
_“in many cases in our data, returning flow derivatives to 0 at the end of the cycle conflicted with the need to match the experimental data and conflicted with the requirement for equal areas under positive and negative curves in the flow derivative (the equal area constraint). [...] These conflicts between model features, constraints, and the empirical data were handled by abandoning the equal area constraint.”_

Perfect closure, then, is mostly a mathematical nicety. Real folds often leak, especially in breathy or pathological voices. We therefore allow $u(t)$ to end at any constant level rather than forcing it to zero.#footnote[
  Early inverse-filtering work also struggled with this: _“After this, the fine-tuning of the resistors of the inverse network was conducted by searching for such values that yielded zero flow after the instant of glottal closure.”_ @Alku2011[after @Miller1959].
]

==== Allow negative flow?
While most GFMs enforce $u(t) ≥ 0$, transient negative flow is sometimes reported @Fujisaki1986 @Degottex2010. It likely reflects a brief suction as the folds recoil—an aerodynamic, not numerical, effect. Its magnitude is small but real, and we do not forbid it: later models learn when such reversals are warranted by data.

/*
=== One of many: the glottal flow "model zoo"

#figure(
  image("./fig/gf-models.png", width: 100%),
  placement: bottom,
  caption: [
    *A lineup of GF models* back from 1986 (handdrawn?), together with their derivatives (DGFs). From #cite(<Fujisaki1986>, form: "author").
  ],
) <fig:gf-lineup>

Many models for the glottal flow and its derivative have been proposed over the course of time in acoustic phonetics -- see @fig:gf-lineup for a catalogue already in 1986. These models differ mainly in the details, and can be unified in a common framework as per #pcite(<Doval2006>).

#pcite(<Doval2006>) note that the most useful GFMs have the following general properties:
- positive: $u(t) >= 0$
- Differentiable ... n times, continuous -- see @Doval2006
- DGF integrates closure constraint @eq:closure-constraint
- definite polarity : Always points up or down \citep{Chen2016}.

/*
and ideally
- interpretable
- Computationally: quick to evaluate
- well defined spectrum: ``voice quality is often described in spectral terms (e.g., voice spectral tilt, brightness, tenseness): spectral parameters are closer to perception than time-domain parameters.'' \citep[p.~1274]{Perrotin2021}
*/

Importantly, to put it simply nearly all GFMs can be recast in the LF parameters we introduced above and the 2 phases, open and closed, in which the closed phase also hosts the return phase.
@Doval2006 go further and unify the parametrizations, but we don't go there as we will introduce GFMs which move from parametric to nonparametric regime.

==== Allow partial closure?
Probably better to allow some slack.
Experimentally -- see this quote:
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

#footnote[
  Interesting historical example of incorporating hard closure constraint directly:
  Note that this connects with ``After this, the fine-tuning of the resistors of the inverse network was conducted by searching for such values that yielded zero flow after the instant of glottal closure.'' \citep[p.~626]{Alku2011} by the first study on glottal inverse filtering by \citep{Miller1959}.
]

With our approach, as we will see, softer constraints are learned by "transfer learning" and incorporated into priors.
We don't always impose perfect zero flow as in @eq:closure-constraint.


==== Allow negative flow?
At first sight, glottal flow should be strictly postive $u(t) >= 0$, and most GF models enforce it. But there are exceptions like @Fujisaki1986, also noted in \citep[p.~35]{Degottex2010}. Motivation: "rounded closure" is often seen; sometimes attributed to residual leakage, but they argue there is also a component due to a period of negative flow caused by lowering of the vocal cords after closure, drawing DC current of air back in.

*/

== Classic piecewise polynomial models
<sec:classic-polynomial-models>

We make a case for the old "forgotten" family of polynomial GF models such as @Alku2002 @Verdolini1995 @Doval2006:
- Computationally fast, analytical null flow condition
- Many exist in literature guised in orders $n = 0,1,2,3$
- Capable of very sharp events
- "Bright spectrum": very slow decay, so are excellently placed to excite GF

/*
Many other models for the glottal flow exist, including the Rosenberg++ model (Veldhuis, 1998), Fant model (Fant, 1979), Hedelin model (Hedelin,
1984), Childers polynomial model (Childers, 1995), etc. However, the models mentioned in this review are the most prevalent in the literature.
@OCinneide2012 p. 37

*/


/*
See images ./fig/:

0th order: @Alku2002
1th order: @Titze2000 @Verdolini1995 (pdf paywalled)
2nd order: KLGLOTT88 (from @Doval2006 A1.1)
3rd order: R++ (from @Doval2006 A1.2)
3rd order: @Fujisaki1986 (FL model, also in @Drugman2019a)
* also allow negative flow segment after closure
* Motivation: “rounded closure” is often seen; sometimes attributed to residual leakage, **but they argue there is also a component due to a period of *negative flow* caused by *lowering of the vocal cords* after closure**

The Rosenberg–Klatt model is a straightforward glottal flow model. It models the shape
of the glottal airflow signal within one fundamental period using a cubic polynomial function @Bleyer2017

non-polynomials:
Rosenberg-C: trig (sine) model
LF-model: trig + exp model
*/

The modern-day revival of piecewise functions (linear, quadratic, ...) puts these ancient models in a new light. Changepoint modeling ("hard ifs") in the guise of decision surfaces is what drives deep architectures today, and it is exactly the same kind we need for GFs. Plus, these models are already embedded in zero DC line (ie, a polynomial of order 0) as they model only open phase.

There are conventially several changepoints in the glottal cycle to be modeled: the primary changepoints are opening onset and closure instant, with optional landmarks like max flow (maximum of $u(t)$) and closing phase onset (minimum of $u'(t)$, start of return phase) used to quantify shape.

We now look at the simplest of the polynomial models in more detail. We will use this model below as a starting point for our generalization to general polynomials of arbitrary degree $n$ and precision $H$, and finally take the limit $H -> oo$.

=== The triangular pulse model

The simplest and arguably most succesful polynomial model is the triangular pulse model proposed in #pcite(<Alku2002>) which is asserts $u(t)$ piecewise linear in GF ($n=1$ degree polynomial) and $u'(t)$ piecewise constant ($n = 0$ degree polynomial). It is used mainly as a more robust way to estimate OQ (a time domain parameter) from the amplitude domain and not as a GF model in itself, but we can use it as a starting point for our generalization from parametric to nonparametric models.

#figure(
  gnuplot(read("./fig/alku2002.gp")),
  placement: top,
  caption: [
    The triangular pulse model proposed in #pcite(<Alku2002>).
  ],
) <fig:alku>

@fig:alku shows the triangular pulse model for the glottal flow and its derivative given a period $T$. Its derivative is a rectangular (piecewise constant) function:
$
  u'(t) = cases(
    0 quad quad & 0 & < t & <= t_o,
    +f_"ac" / (t_e-t_o) quad quad & t_o & < t & <= t_m,
    -f_"ac"/ (t_e-t_m) quad quad & t_m & < t & <= t_e = t_c,
     // 0 quad quad & t_e & > t,
  )
$ <eq:dgf-piece>
This function is parametrized by the time domain parameters ${t_o, t_m, t_e}$, which specifiy the changepoints, and amplitude domain parameters ${f_"ac", d_"peak"}$. Note that $d_"peak"$ is conscipicously absent in @eq:dgf-piece; this is because the closure constraint $integral_(t_o)^(t_e) u'(t) dif t = 0$ removes one degree of freedom, so any single one of these can be expressed in terms of the others. Thus $d_"peak" = f_"ac"/(t_e-t_m)$ or equivalently $t_e - t_m = f_"ac"/d_"peak"$. #pcite(<Alku2002>) point out that this last relation expresses a difficult-to-measure time domain quantity as the ratio of two easy-to-measure quantities in the amplitude domain and exploit this fact to measure the open quotient (OQ) more robustly.


== Parametric piecewise polynomial models
<sec:parametric-piecewise-polynomials>

The rectangular pulse model @eq:dgf-piece contains two jumps, so we can write it more generally as a linear combination of two Heaviside functions during the open phase:
$
  u'(t) = a_1 (t)_+^0 + a_2 (t - t_m)_+^0 quad (t_o <= t <= t_c)
$

where $(t)_+^0 = max (0, t)^0$ is the Heaviside function and
$
  a_1 = f_"ac" 1/T_1, quad a_2 = -f_"ac" (1/T_1+1/T_2).
$
Note that the amplitudes $bm(a) = {a_1, a_2}$ have


This is an instance of a regression problem with $H$ fixed basisfunctions. Following #pcite(<MacKay1998>),

$
  phi_h (t \; bold(theta))
$

But we needn't stop here. We now restate the rectangular pulse model @eq:dgf-piece during the open phase as a probabilistic standard linear model @MacKay1998, in which Gaussian amplitudes modulate fixed basis functions (assume hyperparameters $bold(t)$ fixed). For increased resolution (extra changepoints), we can generalize this to a linear combination of $K$ arbitrarily scaled Heaviside jumps centered at change points $t_(1:K) in [t_o, t_e]$:
$
  u'(t) = sum_(k=1)^K a_k H(t - t_k) quad (t_o <= t, t_k <= t_c)
$
If we assume the $t_(1:K)$ given for now and let $a_k ~ N(0, sigma^2_a)$, then using $integral_(-oo)^t H(tau - c) dif tau = H(t - c)(t - c)$ we get the closure condition for free in expectation:
$
  bb(E)_a [integral_0^T u'(t) dif t] = sum_(k=1)^K bb(E)[a_k] H(T-t_k) (T-t_k) = 0.
$
This motivates setting the mean of the $a$ to zero, otherwise than symmetry of ignorance around $a = 0$.
We can also enforce it: set $b_k = H(T-t_k) (T-t_k) = T - t_k$ then
$
  sum_(k=1)^K a_k b_k = 0 ==> a divides b^top a = 0 ~ cal(N)(0, sigma_a^2 (I - q q^top))
$
where $q = b/(||b||)$. Equivalently,
$
  a = (q q^top) z + (I - q q^top) z, quad z ~ cal(N)(0, sigma_a^2 I). \
  => a = (I - q q^top) z
$
This is a linear constraint on the $a$, so we can update the prior to always respect the constraint. A single constraint projects out the component of $a$ along $b$, so the resulting covariance matrix is degenerate and has rank $K - 1$. A Bayesian way to derive this would make use updating the prior via $D_"KL"$.

This shows how prior of $a$ can encode properties we care about. This is a central theme in what follows: we will find that our priors for the linear amplitude can encode a host of features such as differentiability, closure, ... in other words, the gross features of the expected spectrum of the DGF.

/* picture of triangular pulse model with K=2 ... K = 5 with t_k chosen uniformly and respecting the closure constraint */

In the previous example, we proposed to increase resolution by increasing $K$; we now can add more expressivity by allowing the degree $n$ to be $>= 0$ as well. In this way we also include the higher order classic polynomial models in @sec:classic-polynomial-models.

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
  caption: [Activation functions. After @Cho2009.],
) <fig:repu>

The need to access both $u(t)$ and $u'(t)$ also suggest to write $H(t)$ as a RePU (rectified power unit, aka thresholded monomials) function:
$
  (t)_+^n = H(t) t^n = cases(
    0 quad & t < 0,
    t^n quad & t >= 0,
  )
$
so now integration and differentiation act as convenient ladder operators:
$
  integral_(-oo)^t (tau - c)_+^(n-1) dif tau & = 1/n (t - c)_+^n, \
                     dif/(dif t) (t - c)_+^n & = n (t - c)_+^(n-1) quad quad (n >= 1, c in bb(R))
$
allowing us to quickly go from DGF to GF and vice versa.
#footnote[These relations can be made precise by using distribution theory @Lighthill1958 rather than functions, but they will do for our purpose here.] $t_+^n$ includes Heaviside for $n = 0$ ($t_+^0 = H(t)$); ReLU for $n = 1$ (linear), ReQU for $n = 2$ (quadratic) and ReCU for $n = 3$ (cubic) -- see @fig:repu.

The general parametric polynomial model of degree $n$ and order $K$ is now
$
      u'(t) & = sum_(k=1)^K a_k (t-t_k)_+^n \
  "where" a & ~ cal(N)(0, sigma_a^2 (I - q q^top)", " t_k in [t_o, t_e] "are given," \
  "and" q_k & = 1/(n+1) (T - t_k)_+^(n+1) = 1/(n+1) (T - t_k)^(n+1)
$
which expresses the DGF as a of changepoints $t_k$ followed by changes of direction according to the amplitude $a_k$. All polynomial models of @sec:classic-polynomial-models can be expressed in this way, with the constraint on $a$ taking into account the closure constraint automatically rather than deriving it for each model (as e.g. in @Doval2006).

#figure(
  grid(
    columns: 2,
    column-gutter: { 4em },
    row-gutter: { 1em },
    align: center + bottom,
    [#include "./fig/nn-polynomial-a.typ"], [#include "./fig/nn-polynomial-b.typ"],
    [(a)], [(b)],
  ),
  placement: top,
  kind: image,
  caption: [
    (a) The triangular pulse model as a tiny neural network with a single hidden layer, linear readout and $t^0_+ = H(t)$ activation.
    (b) The general parametric polynomial model of degree $n$ with order $K$ with weights $w in bb(R)^(K times 2), a in bb(R)^K$ and RePU activation function $t_+^n = H(t) t^n$.
  ],
  gap: 1em,
) <fig:nn-polynomial>


*Connection to neural networks.* In a modern lens, these models can be seen as a neural net with a single hidden layer in the regression setting. $H(t) t$ is a ReLU, etc. This is good, because already a single hidden layer has the universality property. Encouraged, this also suggests the prior $t_k ~ N(0, sigma_t^2)$ truncated to $t_k in [t_0, t_e]$: these are biases in the neural net picture, and we truncate to remain in the open phase.

To summarize what we did: we formulated all classic polynomial DGF models as linear polynomial changepoint models, where the $bold(t)$ changepoints were given hyperparameters. Changepoints are encoded as RePU functions; the closure constraint can be imposed analytically. Next we will bring the $bold(t)$ from hyperparameters to ordinary parameters.

The parameters in the polynomial models are always DGF amplitudes (GF slopes) and changepoints, eg. @Fujisaki1986 Fig 2 has 6 parameters. Other models have other parameters, such as LF, which has 5. These can in principle be approximated to arbitrary precision with piecewise polynomial models, such that their parameters are again just amplitudes and changepoints. This is what we mean by parametric models. Next up we will marginalize over them such that they may


/* now we got a prior for t_k: we can show samples */
/* K, n picture */

== Nonparametric piecewise polynomial models

After having generalized $n$, now we generalize $H -> oo$. Let $phi(t) = (t)_+^n$ with $n$ given, then the covariance matrix $K$ is given as

/*
We want to show that inf resolution still goes

But we can't do MacKay completely: he has fixed basisfunctions

We need to write again the expectation <f f'> which he could write as sigma <phi phi'> where phi are fixed, and we can't => this is how we get the expectiation over Gaussian weights and we are done

MacKay integrates first over amplitude weight and THEN integrates over input index h, we integrate BOTH over h (resolution) AND weights (prior) simulteaunisky

*/

$
  K_(i j) & = sigma_a^2 sum_(h=1)^H phi_h (t_i) phi_h (t_j) \
  & = sigma_a^2 sum_(h=1)^H phi(w_(h 2) t_i + w_(h 1)) phi(w_(h 2) t_j + w_(h 1)) \
  &prop sigma_a^2 integral phi(w_(2) t_i + w_(1)) phi(w_(2) t_j + w_(1)) cal(N)(w_1 divides 0, sigma_1^2) cal(N)(w_2 divides 0,sigma_2^2) dif w_1 dif w_2 "as" H -> oo \
  &= arccos_n {bm(Sigma)^(1/2) vec(1, t_i), bm(Sigma)^(1/2) vec(1, t_j) }
$
where $Sigma = "diag"(sigma_1^2, sigma_2^2)$ and the weights are defined in @fig:nn-polynomial. This is the generalized "covariance" arc cosine kernel from @Cho2009 @Pandey2014.

We can absorb $sigma_a$ into $Sigma$, as $arccos$ is homogenous for global rescaling, which is equivalent to rescaling $Sigma -> alpha Sigma$. What is excellent: we managed to push one layer of hyperparameters into the amplitudes! Since we marginalized them away, we end up with a nonparametric polynomial DGF model. We have only three hyperparameters: $sigma$, $sigma_1$, $sigma_2$ instead of $O(H)$ amount. This is the key to fast multiple kernel learning.

But we can allow $bm(Sigma) = mat(sigma_b^2, 0; 0, sigma_t^2)$ as this is important to model behavior of kernel. Introducing a third parameter $rho$ (correlation in $Sigma$) breaks our FT derivation (the $tan "/" arctan$ trick), which assumes no correlation between bias and $t$. So individual rescaling is as far as we can go and probably more than enough. So we can proceed with the $N(0,I)$ case as in @Cho2009. // https://chatgpt.com/s/t_68dfa3181bf88191a3183a8138bf2969



Here the closure constraint becomes analytically "intractable" at first sight, but can be done for SqExp model analytically, and Matern models via Matern expansion trick @Tronarp2018

// @Pandey2014: covariance arccos kernel: to go wide or deep in NNs?
// https://upcommons.upc.edu/server/api/core/bitstreams/bf52946e-0904-4e3d-afb7-d85d8a33c46a/content page 13


/*
Kernel support for various GF models without closure constraint

Calculate p(D|k) for k in (arccos, matern, spline) <= nonparametric
Perhaps also a Bayesian network <= parametric

And D is a single open phase of poly (orders <= 3) and nonpoly (order = infty) models

See which kernel or model has most support

Also show # params and compute time for each

*/

#figure(
  image("/figures/svg/20251008144755528.svg"),
  caption: [Hard changepoints are difficult, best we can do are steep slopes.],
) <fig:steep>

==== Neural networks again.
The marginalization above was first done by #pcite(<Cho2009>) in the context of infinite-width neural networks in the style of #pcite(<Neal1996>) #pcite(<Williams1998>). This viewpoint also allows for going beyond a depth 1 network by iteration of the $arccos$ kernel.#footnote[Kernel composition (reiteration) is indeed a valid kernel operation in general, as recently emphasized by @Dutordoir2020.]

==== 
Why not go infinite depth?
Different character from increasing width; effective depth of deep GPs. If stable limit, it becomes independent of inputs @Diaconis1999. Seen often in DGPs as input independency, "forgetting inputs". Though one might counteract that going to infinite width also has similar "unfortunate" consequences (MacKay's baby with the bath water): "features are not learned", basis functions are fixed. This shows that kernel hyperparameters must encode (most of) features. We do this via sparse multiple kernel learning; ie static kernels on the Yoshii-grid mechanism; our hyperparams are $(T, tau, "OQ")$, ie 3 dim grid.

==== Why not spline models?
Piecewise spline models are well-known and effective in low-dimensional nonparametric regression. Why not use them? Because they depend on the resolution. As Gaussian process priors, spline kernels produce posterior means that are splines with knots (hingepoints in some derivative) fixed at the observed inputs and nowhere else @MacKay1998[p. 6]. In contrast, the $arccos(n)$ kernel will learn the effective number of hingepoints from the data, which may happily remain $O(1)$ while the amount of datapoints grows indefinitely. Translated to our problem, we want the number of effective change points to be resolution-independent (independent of the sampling frequency) and not confined to the observation locations.

==== Why not Lévy processes?
These encode Poisson-style jumps $O(1)$ in number in time. But inference in these is always $O("# of jumps")$. So can't really marginalize out these jump points, and we want to avoid MCMC. We want to stack everything in the amplitude marginalization. But, actual discontinuities require Lévy processes; the arc cosine GP alone can only fake it with steep ramps (see @fig:steep).

/* main point here: YES, we compromised; we got fast inference, but also diminshed support for true O(1) amount of jumps like a Levy process would. This is a practical question -- need to find out by running nested sampling and see how much support for these jumps is really there -- just calculate! */

/* This means we should model u(t), not du(t)!

u(t) ~ arccos: arccos is always continuous -- we cant do real jumps

but if we can get du(t) implicitly -- from spectral domain OR via AR pole -- we can maybe cheat this thing

Maybe we can LEARN the radiation characteristic by priming our AR prior to look for that spectral decay of +6db/oct

So we model u(t) * (h(t) * r(t)) and the latter factor of combined VT + radiation impulse response is learned directly

another option is to integrate data (move diff to data) but this possibly nasty

pre-emphasis just flattens spectral slope as to partition poles better -- has nothing to do with radiation factor.
Radition factor can only be undone by integrating, not further diffing.
Pre-emphasis tilts the spectrum "up" by emphasizing higher freqs and is a fitting trick

On the other hand
LPC just proclaims s(t) = e(t) * h(t) and shoves radiation into the source such that h(t) cna remain all-pole
since differentiation is ~ i2pi x => zero, not a pole
so we are generalizing LPC, so also choose e(t) => u(t) * r(t)
*/