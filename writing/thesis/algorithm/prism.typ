#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page

= PRISM
<algo:prism>

PRISM stands for *process-induced surrogate modeling*.
The central challenge motivating PRISM is one of representation: glottal flow waveforms arrive as time series of varying length, not as fixed-dimensional vectors, yet everything downstream — clustering, density modeling, latent variable inference — wants fixed-dimensional inputs with calibrated uncertainty.
One could fit a GP posterior to each waveform separately using whatever kernel one happens to like, and store the resulting function.
But storing a posterior function per waveform is expensive, and more importantly there is nothing shared: each waveform lives in its own GP, and you cannot meaningfully cluster or compare "GP posteriors" unless they decompose onto a common basis.

PRISM resolves this by learning a single shared kernel-induced basis from all waveforms jointly, then projecting each waveform into a Gaussian posterior over a fixed-dimensional vector of basis amplitudes.
The result is a collection ${ (bm(m)_i, bm(S)_i) }_(i=1)^I$ of Gaussians in a common $bb(R)^M$, one per waveform, carrying both the best estimate of how that waveform decomposes onto the shared basis and the uncertainty in that decomposition.

The mechanism that makes this tractable is a collapsed variational bound in the style of @Titsias2009.
The essential observation is that for Gaussian likelihoods, the per-waveform variational parameters can be eliminated analytically during training, so only global basis parameters need to be optimized.
Projection to per-waveform Gaussians then becomes a cheap post-training step requiring no gradient computation.
This appendix derives the bound, the projection, and explains why collapse is not merely a computational convenience but the design choice that distinguishes PRISM as a shared-basis learner from a collection of independent GP fits.

== Data and model

We have $I$ independent waveforms.
Waveform $i$ consists of a time grid $bm(t)_i = (t_(i 1), dots, t_(i N_i))$ and observations $bm(y)_i = (y_(i 1), dots, y_(i N_i))$, with $N_i$ varying freely across waveforms.
Each is modeled by an independent latent function
$
  f_i (.) ~ mono("GaussianProcess")(0, k_theta (., .)),
$ <eq:shared-gp>
sharing the same kernel hyperparameters $theta$ across all $i$ but with independent draws of $f_i$.
Observations are conditionally independent given $f_i$:
$
  bm(y)_i | f_i ~ mono("Normal")(f_i(bm(t)_i), sigma^2 bm(I)_(N_i)).
$

The shared kernel $k_theta$ is the central object being learned.
Its hyperparameters $theta$, together with the noise variance $sigma^2$, are the global parameters of the model.
The per-waveform objects — the $f_i$ themselves, or any variational approximation to them — are local.
The tension between learning global parameters efficiently while marginalizing over local ones is precisely what the collapsed bound resolves.

== From kernel to shared basis

Choose $M$ inducing locations $bm(Z) = (z_1, dots, z_M)$ in the time domain, also treated as global learnable parameters.
The inducing variables $bm(u)_i = f_i(bm(Z)) in bb(R)^M$ are the values of example $i$'s latent function at the shared locations.
Since $f_i$ is a GP,
$
  bm(u)_i ~ mono("Normal")(bm(0), bm(K)_(Z Z)),
$
where $bm(K)_(Z Z) in bb(R)^(M times M)$ with $(bm(K)_(Z Z))_(m n) = k_theta(z_m, z_n)$ is the same for all $i$.
The GP conditional on $bm(u)_i$ is
$
  f_i(bm(t)_i) | bm(u)_i ~ mono("Normal")(bm(K)_(i Z) bm(K)_(Z Z)^(-1) bm(u)_i, bm(K)_(i i) - bm(Q)_(i i)),
$ <eq:gp-conditional>
where $bm(K)_(i Z) in bb(R)^(N_i times M)$ has $(n, m)$ entry $k_theta(t_(i n), z_m)$, and $bm(Q)_(i i) = bm(K)_(i Z) bm(K)_(Z Z)^(-1) bm(K)_(Z i)$ is the rank-$M$ approximation to $bm(K)_(i i)$.

The whitened parameterization makes the shared structure explicit.
Let $bm(L)$ be the Cholesky factor with $bm(K)_(Z Z) = bm(L) bm(L)^top$, and define whitened amplitudes $bm(a)_i = bm(L)^(-1) bm(u)_i$, so that $bm(a)_i ~ mono("Normal")(bm(0), bm(I))$ under the prior.
Writing $bm(k)_Z(t) = (k_theta(z_1, t), dots, k_theta(z_M, t))^top in bb(R)^M$ for the cross-covariance vector at time $t$, the whitened feature map is
$
  phi(t) = bm(L)^(-1) bm(k)_Z(t) in bb(R)^M,
$ <eq:feature-map-prism>
and the mean in @eq:gp-conditional can be written as $bm(Phi)_i bm(a)_i$, where
$
  bm(Phi)_i = mat(phi(t_(i 1))^top; dots.v; phi(t_(i N_i))^top) in bb(R)^(N_i times M)
$ <eq:design-matrix>
is the design matrix that maps each observation time to the shared feature space.
This is the "prism": every irregular time grid, no matter how long or short, is mapped to a common $bb(R)^M$ through the shared basis.
Each waveform thus becomes an instance of Bayesian linear regression,
$
  bm(y)_i = bm(Phi)_i bm(a)_i + bm(epsilon)_i, quad quad bm(a)_i ~ mono("Normal")(bm(0), bm(I)), quad quad bm(epsilon)_i ~ mono("Normal")(bm(0), sigma^2 bm(I)).
$ <eq:blr-prism>

== Collapsed variational bound

=== The cost of keeping local parameters
A standard sparse variational GP @Titsias2009 approximates the posterior over $bm(u)_i$ with a Gaussian $q_i(bm(u)_i) = mono("Normal")(bm(m)_i, bm(S)_i)$ and maximizes a per-example ELBO
$
  cal(L)_i^("SVGP") (bm(m)_i, bm(S)_i; bm(Z), theta, sigma^2)
  = bb(E)_(q_i) [log p(bm(y)_i | f_i)] - D_"KL" (q_i (bm(u)_i) || p(bm(u)_i)).
$
In the present setting, examples are independent GPs sharing $(bm(Z), theta)$.
Following this route directly, we must store and optimize variational parameters $(bm(m)_i, bm(S)_i)$ for every waveform $i$ — a storage cost of $cal(O)(I M^2)$ in the covariances alone.
More fundamentally, the local parameters would need to be maintained and warm-started across minibatches during stochastic optimization, requiring a large synchronized state that makes proper minibatching over waveforms awkward.

=== Titsias collapse
For Gaussian likelihoods, @Titsias2009 showed that the optimal $q_i(bm(u)_i)$ can be found analytically as a function of $(bm(Z), theta, sigma^2)$ and the observed data $(bm(t)_i, bm(y)_i)$ for that example alone.
Substituting this optimal form back into the ELBO yields a bound that depends only on the global parameters, and no per-example variational state needs to be stored or updated.

To state the bound compactly, define the scaled feature matrix and its Gram matrix,
$
  bm(A)_i = bm(Phi)_i^top / sigma in bb(R)^(M times N_i),
  quad quad
  bm(B)_i = bm(I)_M + bm(A)_i bm(A)_i^top in bb(R)^(M times M),
$ <eq:AB>
and the sufficient statistic $bm(v)_i = bm(A)_i bm(y)_i = bm(Phi)_i^top bm(y)_i \/ sigma in bb(R)^M$.
The collapsed ELBO for example $i$ is
$
  cal(L)_i (bm(Z), theta, sigma^2)
  = log mono("Normal")(bm(y)_i | bm(0), bm(Q)_(i i) + sigma^2 bm(I))
  - 1/(2 sigma^2) "Tr"(bm(K)_(i i) - bm(Q)_(i i)).
$ <eq:collapsed-elbo>
The first term is the log marginal likelihood under the low-rank approximation; the second penalizes the approximation error.

=== Evaluating the bound
Both terms are computed efficiently from $bm(A)_i$, $bm(B)_i$, and $bm(v)_i$, so the cost per waveform scales as $cal(O)(M^2 N_i)$ and the $N_i times N_i$ matrix $bm(Q)_(i i) + sigma^2 bm(I)$ is never formed.

For the log marginal term, write $bm(Q)_(i i) + sigma^2 bm(I) = sigma^2 (bm(I) + bm(A)_i^top bm(A)_i)$.
By the matrix determinant lemma, $det(bm(I)_(N_i) + bm(A)_i^top bm(A)_i) = det(bm(B)_i)$, so
$
  log det(bm(Q)_(i i) + sigma^2 bm(I)) = N_i log sigma^2 + log det(bm(B)_i).
$
The Woodbury identity gives $(bm(Q)_(i i) + sigma^2 bm(I))^(-1) = sigma^(-2) (bm(I) - bm(A)_i^top bm(B)_i^(-1) bm(A)_i)$, so the quadratic form is
$
  bm(y)_i^top (bm(Q)_(i i) + sigma^2 bm(I))^(-1) bm(y)_i = sigma^(-2) (||bm(y)_i||^2 - bm(v)_i^top bm(B)_i^(-1) bm(v)_i).
$
For the trace correction, $"Tr"(bm(Q)_(i i)) = "Tr"(bm(Phi)_i bm(Phi)_i^top) = ||bm(Phi)_i||_F^2$, so
$
  -1/(2sigma^2) "Tr"(bm(K)_(i i) - bm(Q)_(i i)) = -1/(2sigma^2) ("Tr"(bm(K)_(i i)) - ||bm(Phi)_i||_F^2).
$
The full dataset objective is the sum $cal(L)(bm(Z), theta, sigma^2) = sum_(i=1)^I cal(L)_i$.
Every quantity here is computable from the global parameters and the data for example $i$; no per-example state persists between gradient steps.

== Stochastic optimization
Because examples are independent, an unbiased estimate of $cal(L)$ is available from any minibatch $cal(B) subset {1, dots, I}$,
$
  hat(cal(L)) = I / (|cal(B)|) sum_(i in cal(B)) cal(L)_i (bm(Z), theta, sigma^2).
$
Gradient ascent on $(bm(Z), theta, sigma^2)$ using $hat(cal(L))$ is stochastic variational inference in the sense of @Hoffman2013: minibatching over independent examples, updating global parameters only.
The absence of local parameters to synchronize or warm-start is what makes this natural rather than contrived.

== Projection to the PRISM representation

Once $(bm(Z), theta, sigma^2)$ are trained, the design matrix $bm(Phi)_i$ is determined for any waveform, and the BLR model @eq:blr-prism is fully specified.
The per-example posterior over whitened amplitudes follows immediately from Bayesian linear regression:
$
  bm(a)_i | bm(y)_i ~ mono("Normal")(bm(m)_i, bm(S)_i),
$ <eq:prism-posterior>
$
  bm(S)_i = (bm(I) + sigma^(-2) bm(Phi)_i^top bm(Phi)_i)^(-1) = bm(B)_i^(-1),
  quad quad
  bm(m)_i = sigma^(-2) bm(S)_i bm(Phi)_i^top bm(y)_i = bm(B)_i^(-1) bm(v)_i.
$ <eq:prism-posterior-params>
This reuses $bm(B)_i$ and $bm(v)_i$ from the collapsed bound, so the projection costs nothing beyond what training already computed.
The PRISM representation is the resulting collection
$
  { (bm(m)_i, bm(S)_i) }_(i=1)^I,
$
a set of Gaussians in a common $bb(R)^M$, one per waveform, encoding amplitude estimates and their uncertainty.
Downstream tasks — clustering, density modeling, mixture of factor analyzers — operate entirely in this fixed-dimensional latent space, where the variable-length origin of the data is no longer a complication.

=== Connection to PRISM-RFF
<sec:prism-prism-rff-connection>
The projection @eq:prism-posterior-params is structurally identical to the amplitude posterior in @chapter:prism-rff (@eq:posterior there).
What differs is only the feature map $phi$: in ordinary PRISM, $phi(t) = bm(L)^(-1) bm(k)_Z(t)$ is built from kernel evaluations at the inducing locations $bm(Z)$; in PRISM-RFF, $phi(tau)$ is built from windowed Fourier projections to inducing frequencies.
The collapsed bound, the BLR structure, the Woodbury/determinant-lemma evaluation, and the projection formula are all inherited unchanged.
PRISM-RFF is therefore not a new algorithm but a reparameterization of the basis: from time-domain inducing points to spectral features, with all the translation-invariance benefits that follow for DGF.

== Summary

PRISM learns a shared, global kernel-induced basis from a collection of independent variable-length time series by maximizing the collapsed objective
$
  max_(bm(Z), theta, sigma^2) sum_(i=1)^I cal(L)_i (bm(Z), theta, sigma^2),
$ <eq:prism-objective>
without storing or updating per-waveform variational parameters.
After training, each waveform is projected to a Gaussian in a fixed-dimensional latent space @eq:prism-posterior, and downstream modeling proceeds entirely there.

The collapse step is not an implementation detail.
Without it, shared basis learning would require maintaining and synchronizing per-waveform Gaussian states across minibatches, at a memory cost that grows with $I$.
With it, the learning problem reduces to optimizing $bm(Z)$, $theta$, and $sigma^2$, and the per-example representations are assembled on demand from the trained basis alone.
