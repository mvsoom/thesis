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

= Latent variable models: going deeper

== QGP-LVM

PRISM-VFF can learn the closed phase better than PRISM could (there were hints, but not strong separation) when doing the qGPLVM trick



== Born expansions

