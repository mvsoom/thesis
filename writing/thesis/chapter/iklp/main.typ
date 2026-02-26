#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

= Infinite kernel linear prediction
<chapter:iklp>

== Linear prediction

== Kernel linear prediction

Linear prediction augmented with a more realistic noise source.

We explain and generalize IKLP by allowing for more flexible priors:
- We use a nonzero mean GP for $bm(x)$ induced by nonzero mean priors for $bm(w)$ in weight space (hard);
- We use a nonzero mean and a general covariance matrix for $bm(a)$ (trivial).
- We rewrite everything in terms of $bm(Phi)_i$ rather than $bm(K)_i$ for stability and speed.
- Use a modern JAX implementation for GPU batch acceleration; could also use derivatives of VI objective if wanted

==== Nonzero mean for $p(bm(w))$
Equation (6) in @Yoshii2013:
$
  epsilon.alt(t) & = sum_(j=1)^J w_j phi.alt_j (t) + eta(t) \
                 & = bm(phi.alt)(t)^top bm(w) + eta(t)
$
When sampled at times $bold(t) = {t_n}_(n=1)^N$:
$
  bm(epsilon.alt) = bm(Phi) bm(w) + bm(eta).
$
Assume
$
    bm(w) & ~ mono("Normal")(bm(mu)_w, nu_w bm(Sigma)_w) \
  bm(eta) & ~ mono("Normal")(bm(0), nu_e bm(I))
$
Then the excitation $epsilon.alt(t)$ is a GP with marginal#footnote[
  We marginalized both over $bm(w)$ and over all other values of $t$ here.
]
$
  bm(epsilon.alt) ~ mono("Normal")(bm(Phi) bm(mu)_w, nu_w bm(Phi) bm(Sigma)_w bm(Phi)^top + nu_e bm(I)).
$
The data likelihood becomes via Eq. (5) in @Yoshii2013:
$
  bm(x) ~ mono("Normal")(bm(Psi)^(-1) bm(Phi) bm(mu)_w, bm(Psi)^(-1) (nu_w bm(Phi) bm(Sigma)_w bm(Phi)^top + nu_e bm(I)) bm(Psi)^(- top))
$
For _multiple kernel learning_, we have for each $i$th GP the linear regression model
$
  {bm(Phi)_i; bm(mu)_w^((i)), nu_w bm(Sigma)_w^((i))}.
$
We assume that the unit power of each $i$th GP is comparable, so $nu_w$ plays the role of overall scale to match the excitation to the data.
#footnote[
  The AR($P$) process also has a nontrivial influence on gain, because gain, phase and frequency are all entangled, so $nu_w$ also has a beneficial untangling effect on these three -- we want the AR($P$) process to focus on phase and frequency, not gain.
]
The marginal of the total GP (summed over all indexes $i$) becomes:
$
  bm(x) ~ mono("Normal")(bm(Psi)^(-1) sum_(i=1)^I theta_i bm(m)_i, bm(Psi)^(-1) (nu_w sum_(i=1)^I theta_i bm(K)_i + nu_e bm(I)) bm(Psi)^(- top))
$ <eq:likelihood-x>
where $bm(m)_i = bm(Phi)_i bm(mu)_w^((i))$ and $bm(K)_i = bm(Phi)_i bm(Sigma)_w^((i)) bm(Phi)_i^top$.

We now have two variables controlling scale of the signal part:
1. $nu_w$: controls _overall_ covariance (power).
2. $theta_i$: controls scale of means and covariances.

So in terms of normalization:
- $nu_w$ can learn the overall scale needed for the signal, so don't need to worry here about the effects of AR($P$) models on output scale.
- Just need to make sure the individual excitation GPs $epsilon.alt_i (t)$ are comparable in terms of power, because then $theta_i$ can play the role of relative importance faithfully.
  (If say $epsilon.alt_3(t)$ and $epsilon.alt_7(t)$ have comparable importance but the latter has much lower power, than $theta_7$ will be much larger than $theta_3$ to make up for this power difference.)
  Therefore when we normalize (gauge) all $u'(t)$ examplars to unit power, the learned GPs will have means and covariances with similar power in turn, and we are good.

==== Derivation
A nonzero mean for $bm(w)_i$ changes the VI algorithm.
The mean of the likelihood for $bm(x)$ is, from @eq:likelihood-x:
$
  bm(mu) = bm(Psi)^(-1) sum_(i=1)^I theta_i bm(m)_i
$
In Eq. (23) in @Yoshii2013 we need to do
$
  bm(x) --> (bm(x) - bm(mu)) = bm(Psi) bm(x) - sum_(i=1)^I theta_i bm(m)_i
$
Here $bm(m)_i$ are constants in the optimization, but ${bm(Psi), theta_i}$ are not.
We thus get another $bb(E)[theta_i]$ in the VI steps and need to make some changes in the downstream derivations.
// we can do that derivation here

[TODO]

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

Infinite because expected amount of nonzero kernels stays $cal(O)(1)$ as $I -> oo$.

== Inference

/* zie papieren */
