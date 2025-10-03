= Gaussian Processes via Span, Weighting, and Woodbury

This is a presentation of GPs specifically geared towards basis function use (low dim case).

Normally you think of Bayesian linear regression as defining "weighted" functions in the span of the basis functions: the MVN prior on the weights induces a prior on functions.

We start humble with linear regression in Jaynes style and can immediately introduce Fourier transform from Bayesian view and the 6 conditions.

Since we should look at data space as much as possible, we use pushforward measure to go from

$ p(w) = cal(N)(0, Sigma) $

to the implied density on $f = Phi w$:

$ p(f) prop exp(-1/2 f^top K^(-1) f) := exp(-1/2 ||f||^2_S) $

where

$ K = Phi Sigma Phi^top $

and we assumed $K$ is full rank.

But you can show that both prior and posterior functions live in the span of the _kernel columns_ (the "Gram span"), which projects out the details of the basis functions -- that $K$ is all you need, and maximum entropy can always reconstruct a basis if you want one.

*We should begin with the simplest nondegenerate GP possible*: analogous to the N(0,I) prior. What is its RKHS?

Nice way of starting because this is the classic LPC starting point and all other GPs can generated from it.

== Why start from spans (not operators)
A Gaussian process is a prior over functions. Nonparametric should not
mean "infinite weights" first; it should mean: the prior is built from a
#emph[span] of functions together with a #emph[weighting] that says
which directions are cheap or costly. Kernels give both.

The correct entry point is the finite case. It contains the whole story
in miniature and, crucially, it shows why the operator viewpoint is
natural only #emph[after] we understand the span. We will start with a
matrix square root, meet the Gram (kernel) span, and only then graduate
to the linear-operator and RKHS picture.

== Finite case: weights $arrow.r$ Gram span, and what Woodbury Moore--Aronszajn
=== General characterization (from matrix square roots)

Pick a set of basis functions collected as a column vector
$ phi.alt \( x \) in bb(R)^d . $ Define the kernel
$ k \( x \, x' \) := angle.l phi.alt \( x \) \, phi.alt \( x' \) angle.r = phi.alt \( x \)^top phi.alt \( x' \) . $
On points $X = { x_1 \, dots.h \, x_n }$ the Gram matrix is
$ K := K \( X \, X \) = Phi Phi^top \, quad upright("with ") Phi_(i :) = phi.alt \( x_i \)^top . $
This is the finite analogue of the universal factorization "kernel $=$
inner product". In matrix language: a symmetric positive semidefinite
matrix $K$ admits a square root $L$ with $K = L L^top$; here $L = Phi$.

#emph[Interpretation.] The family
${ thin phi.alt \( x \)^top w : w in bb(R)^d thin }$ and the Gram family
${ thin k \( x_i \, dot.op \) thin }_(i = 1)^n$ describe the same
geometry on $X$. The first uses basis functions directly; the second
uses their pairwise inner products.

=== Prior and posterior in weight space

Place the Gaussian prior on weights
$ w tilde.op cal(N) \( 0 \, I_d \) \, #h(2em) f \( x \) = phi.alt \( x \)^top w \, $
and observe noisy data
$ y = Phi w + epsilon.alt \, #h(2em) epsilon.alt tilde.op cal(N) \( 0 \, sigma^2 I_n \) . $
Classical algebra gives the posterior mean in weight space
$
  hat(w) = \( Phi^top Phi + sigma^2 I_d \)^(- 1) Phi^top y \, #h(2em) m_(upright("post")) \( x \) = phi.alt \( x \)^top hat(w) .
$

=== Why the #emph[prior] already lives in the Gram span


Perhaps surprisingly, we can also show that the prior lives in the Gram span too. Usually this viewpoint is presented only for the posterior (Rasmussen), but the prior is the same story.

Two rigorous proofs, and a constructive formula for $f = K c$.

#emph[Square-root construction.] Let $K = U_r Lambda_r U_r^top$ be the
eigendecomposition with strictly positive eigenvalues
$Lambda_r = upright(d i a g) \( lambda_1 \, dots.h \, lambda_r \)$. Set
$L := U_r Lambda_r^(1 \/ 2)$ and draw
$z tilde.op cal(N) \( 0 \, I_r \)$. Then
$ f := L z tilde.op cal(N) \( 0 \, L L^top \) = cal(N) \( 0 \, K \) . $
The range of $L$ is $"span" \( U_r \) = "col" \( K \)$, hence
$f in "col" \( K \)$ almost surely.

#emph[Zero-variance directions.] For any $u$ in the null space of $K$,
$ upright(V a r) \( u^top f \) = u^top K u = 0 #h(0em) arrow.r.double #h(0em) u^top f = 0 upright(" a.s.") $
Therefore $f$ is orthogonal to $"null" \( K \)$, i.e.
$f in "null" \( K \)^perp = "col" \( K \)$.

#emph[Constructive coefficients.] With
$f = L z = U_r Lambda_r^(1 \/ 2) z$, define
$ c := U_r Lambda_r^(- 1 \/ 2) z . $ Then
$ K c = U_r Lambda_r U_r^top thin U_r Lambda_r^(- 1 \/ 2) z = U_r Lambda_r^(1 \/ 2) z = f . $
Thus every draw $f tilde.op cal(N) \( 0 \, K \)$ can be #emph[written
  as] $f = K c$ with the explicit $c$ above (and more generally
$c = K^(+) f$ is the minimum-norm solution of $K c = f$).

=== Moving from weights to Gram span: the kernel trick, stated plainly

First, in words:

#emph[Kernel trick (finite, in words).] Any computation that uses only
inner products of the basis vector $phi.alt \( x \)$ with itself and
with training rows $phi.alt \( x_i \)$ can be performed using the kernel
values $k \( x \, x' \)$ alone. In particular, the posterior mean and
all predictive covariances depend on ${ phi.alt \( x_i \) }$ only
through the Gram matrix $K = Phi Phi^top$ and kernel vectors
$k \( x \, X \) = Phi phi.alt \( x \)$. You never need to manipulate $w$
or $d$-dimensional basis coordinates.

Now the same statement in one line:

#emph[Kernel trick (finite, as an equation).]
$ \( Phi^top Phi + sigma^2 I_d \)^(- 1) Phi^top #h(0em) = #h(0em) Phi^top \( Phi Phi^top + sigma^2 I_n \)^(- 1) . $
This is Woodbury. Inserting it into
$m_(upright("post")) \( x \) = phi.alt \( x \)^top hat(w)$ yields
$ m_(upright("post")) \( x \) = k \( x \, X \)^top \( K + sigma^2 I \)^(- 1) y . $

#emph[Why this is the right "dual" viewpoint.] The map
$J : bb(R)^d arrow.r bb(R)^n$, $J w = Phi w$, sends weights to
observable function values. Gaussian priors are invariant under
orthogonal changes of coordinates in the domain; all observable
consequences are governed by $J J^top = Phi Phi^top = K$. Woodbury is
the algebra that exposes this invariance and lets us compute in
$bb(R)^d$ or in $bb(R)^n$, whichever is smaller or more accessible.

=== Posterior also lives in the Gram span

From the previous line,
$
  m_(upright("post")) \( dot.op \) = sum_(i = 1)^n beta_i thin k \( x_i \, dot.op \) \, #h(2em) beta = \( K + sigma^2 I \)^(- 1) y .
$
The posterior mean is a linear combination of the kernel sections at the
observed inputs. This is exactly the "span viewpoint": predictions are
built from the Gram columns.

=== Computational advantage (what is bought)

You can avoid constructing any basis at all:

- If $d$ is large or infinite, the weight equation is ill-posed; the
  Gram system is always $n times n$.

- If $n lt.double d$, Woodbury also gives a fast weight-space route; you
  pick the smaller system.

So for large $d$ the Gram span way of thinking commends itself; we get Gaussian processes when $d arrow.r infinity$.


=== A tiny worked example

Let
$
  phi.alt \( x_1 \) = mat(delim: "[", 1; 0) \, quad phi.alt \( x_2 \) = mat(delim: "[", 0; 1) \, quad phi.alt \( x_3 \) = mat(delim: "[", 1; 1) \, quad sigma^2 = 0.25 \, quad y = mat(delim: "[", 1; - 1; 0) .
$
Then
$ Phi = mat(delim: "[", 1, 0; 0, 1; 1, 1) \, #h(2em) K = Phi Phi^top = mat(delim: "[", 1, 0, 1; 0, 1, 1; 1, 1, 2) . $
Weight-space posterior mean: $hat(w) = \[ 0.8 \, thin - 0.8 \]^top$,
hence
$m_(upright("post")) \( x \) = 0.8 thin \[ phi.alt_1 \( x \) - phi.alt_2 \( x \) \]$.
Gram-space: solve $\( K + sigma^2 I \) z = y$ to get
$z = \[ 0.8 \, thin - 0.8 \, thin 0 \]^top$ and then
$
  m_(upright("post")) \( x \) = k \( x \, X \)^top z = \[ phi.alt_1 \( x \) \, thin phi.alt_2 \( x \) \, thin phi.alt_1 \( x \) + phi.alt_2 \( x \) \] dot.op \[ 0.8 \, - 0.8 \, 0 \] = 0.8 thin \[ phi.alt_1 \( x \) - phi.alt_2 \( x \) \] .
$
Same function; two coordinate systems.

== Infinite case: kernel span $arrow.r$ RKHS $arrow.r$ operators $arrow.r$ spectra

=== Build the function space from the kernel span

Take the span
$ cal(F)_0 := { thin sum_(i = 1)^n alpha_i thin k \( x_i \, dot.op \) #h(0em) : #h(0em) n < oo thin } . $
Declare the inner product on $cal(F)_0$ by

$
  angle.l sum_i alpha_i k \( x_i \, dot.op \) \, #h(0em) sum_j beta_j k \( x_j \, dot.op \) angle.r := sum_(i \, j) alpha_i beta_j thin k \( x_i \, x_j \) .
$

Complete to obtain the Hilbert space $cal(H)_k$ (the RKHS). This is
Moore--Aronszajn in constructive form: we #emph[start] from the kernel
span and #emph[make] a Hilbert space in which
$ k \( x \, x' \) = angle.l k \( x \, dot.op \) \, thin k \( x' \, dot.op \) angle.r_(cal(H)_k) . $
This is the infinite-dimensional version of the square-root identity
$K = L L^top$. It proves existence and gives the basic cost functional
(the norm), but it does not yet diagonalize.

=== Why the operator viewpoint now becomes natural

Once the Hilbert space is in place, consider the linear operator on
$L^2 \( mu \)$
$ \( T f \) \( x \) := integral k \( x \, x' \) thin f \( x' \) thin d mu \( x' \) . $
This operator summarizes the same geometry as the kernel span: it is
self-adjoint and positive. The advantage of moving to $T$ is not
metaphysical; it is computational and conceptual: we can diagonalize.

=== Diagonalizations: Mercer (compact) and spectral (noncompact)

On compact domains with continuous $k$, Mercer yields
$ k \( x \, x' \) = sum_(m = 1)^oo lambda_m thin psi_m \( x \) psi_m \( x' \) \, quad lambda_m gt.eq 0 . $
Write $f = sum_m c_m psi_m$. The RKHS norm is
$ parallel f parallel_(cal(H)_k)^2 = sum_m c_m^2 / lambda_m . $ This is
the weighting: directions with large $lambda_m$ are cheap; small
$lambda_m$ are expensive.

On noncompact domains the spectrum can be continuous and one has
$ k \( x \, x' \) = integral phi.alt \( x \, xi \) thin phi.alt \( x' \, xi \) thin d mu \( xi \) \, $
which is the same diagonalization indexed by a continuum.

=== Bochner as a concrete spectral example

For stationary kernels on $bb(R)^d$,
$k \( x \, x' \) = kappa \( x - x' \)$,
$ kappa \( t \) = integral_(bb(R)^d) e^(i omega^top t) thin d mu \( omega \) \, $
with $mu$ a finite nonnegative measure (the spectral measure). The
"basis" becomes sinusoids $e^(i omega^top x)$; the weighting is the
spectral density $d mu$. Gaussian/RBF corresponds to a Gaussian $mu$.

=== Bayesian reading: prior cost and representer

Informally (Cameron--Martin),
$
  p \( f \) prop exp #h(-1em) #scale(x: 180%, y: 180%)[\(] - 1 / 2 parallel f parallel_(cal(H)_k)^2 #scale(x: 180%, y: 180%)[\)] .
$
The RKHS norm is the negative log prior: it is the cost. With Gaussian
noise the posterior mean solves
$ min_(f in cal(H)_k) #h(0em) parallel f parallel_(cal(H)_k)^2 + 1 / sigma^2 sum_(i = 1)^n \( y_i - f \( x_i \) \)^2 . $
By the representer theorem the minimizer has the Gram-form
$
  f_star.op \( dot.op \) = sum_(i = 1)^n beta_i thin k \( x_i \, dot.op \) \, #h(2em) beta = \( K + sigma^2 I \)^(- 1) y \,
$
exactly as in the finite case. The "dual" sufficient statistics are
$\( K + sigma^2 I \)^(- 1) y$ and $y^top \( K + sigma^2 I \)^(- 1) y$.

=== Computation, again: what diagonalization buys

In practice one needs linear solves with $K + sigma^2 I$, marginal
likelihoods, and predictive variances. The span view tells you that $n$
controls cost; the operator/spectral view tells you which directions are
inexpensive vs expensive:

- Truncating small $lambda_m$ in Mercer space gives principled low-rank
  approximations.

- In stationary problems, Bochner turns kernels into Fourier integrals;
  Monte Carlo yields scalable random quadratures.

- None of this required forming a gigantic SVD of an explicit basis; we
  never even wrote one down. We diagonalized the #emph[induced] operator
  instead.

== The GP prediction equations, with meaning

=== Derivation in the Gram span

From the joint Gaussian of $\( f \( X \) \, f \( x_star.op \) \)$ we
obtain
$
  m_(upright("post")) \( x_star.op \) = m \( x_star.op \) + k_star.op^top \( K + sigma^2 I \)^(- 1) #scale(x: 120%, y: 120%)[\(] y - m \( X \) #scale(x: 120%, y: 120%)[\)] \,
$
$
  "cov"_(upright("post")) \( x_star.op \, x_star.op \) = k_(star.op star.op) - k_star.op^top \( K + sigma^2 I \)^(- 1) k_star.op \,
$
with the usual block notations. These are the central GP equations in
their most economical form: they live entirely in the Gram span.

=== Interpretation from the RKHS

If $K = U Lambda U^top$, then the mean operator is
$
  K \( K + sigma^2 I \)^(- 1) = U thin upright(d i a g) #h(-1em) #scale(x: 180%, y: 180%)[\(] frac(lambda_m, lambda_m + sigma^2) #scale(x: 180%, y: 180%)[\)] U^top .
$
Each direction is shrunk by $lambda \/ \( lambda + sigma^2 \)$.
Large-eigenvalue directions are learned quickly; small ones remain close
to prior. The posterior covariance subtracts exactly what the data
explain along the Gram span.

=== Evidence and the role of the log determinant

The marginal likelihood
$
  log p \( y divides X \, theta \) = - 1 / 2 y^top \( K_theta + sigma^2 I \)^(- 1) y - 1 / 2 log det \( K_theta + sigma^2 I \) - n / 2 log 2 pi
$
balances fit (the quadratic term) against complexity (the log
determinant). In eigenvalues,
$ log det \( K_theta + sigma^2 I \) = sum_(m = 1)^n log \( lambda_m \( theta \) + sigma^2 \) . $
A useful summary is the effective degrees of freedom
$
  gamma \( theta \) := upright(t r) #scale(x: 120%, y: 120%)[\(] K_theta \( K_theta + sigma^2 I \)^(- 1) #scale(x: 120%, y: 120%)[\)] = sum_m frac(lambda_m \( theta \), lambda_m \( theta \) + sigma^2) \,
$
which counts how many Gram directions are effectively used by the
posterior.

== Two crisp summaries of the kernel trick

#emph[Finite, in words.] If you can write a calculation using only inner
products of basis vectors, then you can do it using the kernel values
$k \( x \, x' \)$ alone. Posterior means and covariances depend on
${ phi.alt \( x_i \) }$ only through $K$ and $k \( x \, X \)$.

#emph[Finite, as a formula.]
$
  m_(upright("post")) \( x \) = phi.alt \( x \)^top \( Phi^top Phi + sigma^2 I \)^(- 1) Phi^top y = k \( x \, X \)^top \( K + sigma^2 I \)^(- 1) y .
$

#emph[Infinite, in words.] Build the function space from the kernel
span, equip it with the RKHS norm, and compute entirely in the Gram span
at the observed points. The operator/spectral view explains the
weighting; the Gram view does the work.

#emph[Infinite, as a formula.] For Gaussian likelihood,
$
  f_star.op \( dot.op \) = sum_(i = 1)^n beta_i thin k \( x_i \, dot.op \) \, #h(2em) beta = \( K + sigma^2 I \)^(- 1) y \,
$
regardless of whether the basis is finite, countable, or continuous.

== Closing

The path is: prior over functions $arrow.r.double$ kernel span
$arrow.r.double$ RKHS norm (cost) $arrow.r.double$ operator
diagonalization (bias) $arrow.r.double$ Gram computations (practice).
This is the nonparametric story in one line. We did not assume an
infinite basis and then bolt on kernels; we started from what the finite
case makes unavoidable: predictions and priors live in the span of
kernel sections, and Woodbury is the bridge that turns this fact into
working algebra.
