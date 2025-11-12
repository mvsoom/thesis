#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/equate:0.3.2": equate, share-align
#show: equate.with()



= The quasiperiodic arc cosine kernel
<chapter:qpack>



The quasiperiodic arc cosine kernel (QPACK)


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

== Learning Fourier features

Still the case that eigenfunctions do not depend on hyperparameters, only the expansion coefficients!
- But Fourier inference on grid I think not possible anymore, though the Gram matrix can probably be calculated analytically in $O(M^2)$


== Evaluation on `OPENGLOT-II`

== Summary