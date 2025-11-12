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


== Evaluation on `OPENGLOT`
