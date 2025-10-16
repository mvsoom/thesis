= Infinite kernel linear prediction
<chapter:iklp>

== Linear prediction

== Kernel linear prediction

Linear prediction augmented with a more realistic noise source.

=== Gaussian Processes

In the time–series setting, we consider a stochastic process ${Y(t) | t in T}$ indexed by time $t$.
Here $T subset.eq bb(R)$ represents the time axis, and each $Y(t)$ is a random variable describing the signal value at time $t$.
The process is defined by assigning a consistent joint probability distribution to every finite subset ${Y(t_1), ..., Y(t_N)}$.

A _Gaussian process_ (GP) is a stochastic process that is completely specified by its first two moments:
the mean function
$
  mu(t) = E[Y(t)]
$
and the covariance function
$
  C(t, t') = E[(Y(t) - mu(t))(Y(t') - mu(t'))].
$
Any finite collection ${Y(t_1), ..., Y(t_N)}$ drawn from a GP will have a joint multivariate Gaussian distribution with these mean and covariance functions.
In time–series regression, this provides a nonparametric prior over functions $Y(t)$, allowing uncertainty quantification and smooth interpolation between observed samples.


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
