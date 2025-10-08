= Infinite kernel linear prediction
<chapter:iklp>

== Linear prediction

== Kernel linear prediction

Linear prediction augmented with a more realistic noise source.

== Infinite kernel linear prediction

Multiple kernel learning is a thing, but needn't go there

/*
from Abstract here: https://jmlr.csail.mit.edu/papers/volume12/gonen11a/gonen11a.pdf

"We see that overall, using multiple kernels instead of a
single one is useful and believe that combining kernels in a nonlinear or data-dependent way seems
more promising than linear combination in fusing information provided by simple linear kernels,
whereas linear methods are more reasonable when combining complex Gaussian kernels."

=> We combine the kernels in a data-dependent linear way, so that 's good according to practice

What's more: there is a superposition/blurring principle at play: clusters of "nearby kernels" that are a posteriori active define a single  "interpolated" kernel
*/

Infinite because expected amount of nonzero kernels stays $O(1)$ as $I -> oo$.

== Inference

/* zie papieren */
