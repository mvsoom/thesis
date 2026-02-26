#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/equate:0.3.2": equate, share-align
#show: equate.with()

= Bayesian nonparametric glottal inverse filtering
<chapter:bngif>

Bring together IKLP + AR prior + QPACK = BNGIF.

Evaluate on OPENGLOT.

== Recap of all the ingredients

/*
Gridding over OQ also done in @Fu2006

Can check GCI accuracy with that Hilbert transform database with ~100% correct annotated GCIs
*/

==== Online learning
Priors for the expansion coefficients $bm(a)$ can be updated from the posterior from previous frame.
Same for the AR prior $bm(a)$ (nameclash here).
Likewise, values of ${T, t_c}$ are correlated across frames and this can be exploited too.


== Databases

=== Chien database

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