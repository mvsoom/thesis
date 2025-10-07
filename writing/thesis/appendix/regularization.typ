= Regularization and priors
<chapter:regularization-and-priors>

As its name suggests, GIF belongs to the class of #emph[inverse
  problems] @LeBesnerais2010. Inverse problems confront us when we have
observations of some effects \[noisy oservations of $s (t)$ in our
case\] of which we want to infer the causes $u (t)$ and $h (t)$
within the context of some model $s (t) = u (t) * h (t) * r (t)$.

In the applied sciences it is very often the case for inverse problems
that the observed data are quite consistent with a broad and sometimes
colorful class of possible solutions rather than a unique and logically
deductible solution. From the viewpoint of pure mathematics therefore
such inverse problems are often deemed 'ill-posed', but perhaps a more
productive viewpoint#footnote[And closer to the historical origin of the
  epithet "mal posée" @Jaynes1984a.] is that such problems are simply
underdetermined.

Underdetermined problems are ubiquitous in a wide range of applications,
from medical diagnosis (what could be causing the patient's persistent
cough) to astronomy (what is perturbing the orbit of Uranus). Bayesian
theory by itself does not require any particular structure on the
"feasible set" of plausible solutions and as such underdetermined
problems have no special status within it, nor pose any special
challenge to it @Jaynes2003. In practice however, we would like to have
a more well determined solution to underdetermined problems as our
computing budgets (and patience) are limited. For example, it would most
certainly help if we can already discard those solutions that lack to
some degree a set of required properties we established beforehand. This
is where #emph[regularization] and #emph[priors] come into play.

In general, #emph[regularization] is the process of constraining the
solution space of some problem to obtain a more stable and perhaps
unique solution; in other words, to convert an underdetermined problem
into something more 'well-determined' and, hopefully, more amenable to
computational optimization. For example, ridge regression is a widely
used regularization technique to numerically stabilize linear regression
in high dimensions.#footnote[Ridge regression (also known as $ell^2$
  regularization) can be derived from Bayesian theory as standard linear
  regression with Gaussian priors on the amplitudes
  @Murphy2022[Sec.~11.3]. This principled view allows us to go much
  further than numerical stabilization. We will show in the next chapters
  that ridge regression can (approximately) express intricate pieces of
  prior information such as definite signal polarity, the
  differentiability class to which a function belongs, and power
  constraints on waveforms.]

In the Bayesian framework regularization is imposed simply through
careful assignment of the probability distributions describing a
specific problem – of which the bulk of the effort is traditionally
concentrated on the prior probabilities, or #emph[priors] @Sivia2006.
Assuming our prior information about the problem at hand is correct and
the assignment of the priors accurately reflect that information (and we
got away with the approximations made in our inference methods) we will
have automatically implemented just the right degree of regularization
for our problem; indeed, specially catered to the one data set we
happened to observe, because we express everything in terms of posterior
probabilities that are conditioned on the observed data.

For simple inverse problems where intuition can see the answer clearly,
Bayesian regularization leads to answers that comply with common sense
@MacKay2005. This is simply because Bayesian probability theory is a
formalized theory of plausible reasoning itself @Skilling2005, which
humans undertake every day. But regularization in the Bayesian framework
can go beyond our intuition: it will of course continue to work for
complex problems that require considerable analysis, and this
consistency makes it a valuable tool for attacking inverse problems
@Calvetti2018.
