#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

/*
CODEX PROMPT:

I have a big one for you. I want you to write a comprehensive technical report (basically Chapter 2 in my thesis main.typ or in compiled pdf form which is attached -- main.pdf) which will be the detailed basis for my own writings to build on top of. You can output the report at writing/thesis/chapter/arprior/comprehensive_technical_report_of_iklp.md. Re-use the mathematical symbols of the paper as-is, they are mostly mirrored in the code as well.

The technical report will about how I implemented Yoshii+ (2013) (cited as @Yoshii2013 in Markdown; pdf attached) as done in the Python JAX tree at src/iklp. The equations in the pdf are quite dense, so it is necassary to carefully unwrap them, and (the most important part):

>>> HOW I REWROTE THESE EQUATIONS IN A STABLE WOODBURY FORM WHICH IS READY TO BE USED VERY EFFICIENTLY BY LOW-RANK GPs LIKE THE VARIATION FOURIER FEATURES GPS <<<

and how this is a new angle not present in the paper. You must track and study the code to see how I did it -- and the hard math in woodbury.py must be symbolified and simplified, because it is HIGHLY OPTIMIZED, while the math itself (rewriting everything in terms of stable Choleskys) is really quite elegant and smartly concise.

Important here is that in the technical report you don't refer at all to the particular code as this is a mathematical and technical exposition immaculate from actual implementation. You can however mention that this is a modern and highly optimized JAX implementation capable of running on CPU and GPU and capable of autodiff, even though we use CAVI updates. The goal of the implementation (ie the Cholesky rewrite of the math) is SPEED and STABILITY, and both are obtained. Another advantage is that large batches of data (stacked speech frames) can be processed at high throughput. And that x32 can be used for extra speedups.

In the code you can ignore all the helpers like random_periodic_kernel_hyperparams() etc; they are just there for testing. (Btw you can find tests at tests/iklp for reference if you need more context).

I'll mention more important themes in the context of my thesis that must also make it into the report:
- How IKLP (infinite kernel linear predicton) is a generalization of standard LP methods in acoustic speech science, where now the kernel isn't just white noise but actually can be modeled by a GP or reduced rank GP (this is explained in the paper)
- How IKLP used the number of kernels, I, to operationalize pitch tracking by gridding possible fundamental periods on a grid, but how this is a much more general principle: we will use it in a variety of ways, from pitch tracking to tracking different hypotheses of the type ("modality") of glottal flow derivative (called "excitation") at the same time
- How IKLP also works with P=0, ie no AR component at all, and how this becomes a powerful inference method to do inference with multiple kernels (hypotheses) (this can be a footnote actually)
- How I also rewrote the priors for IKLP into more interpretable parametrizations: these explanations, derivations and experiments are resp. in alpha.ipynb and pi-kappa-prior.ipynb . These must also make it into the report without any loss of detail!
- How VI (variational inference) allows for combining uncertainty with fast inference: we still get the benefit of a Bayesian treatment, but without the slow performance of MCMC (there have been some glottal inverse filtering (GIF) based on MCMC methods, but these are slow -- DSP people are used to speed).
The major motivation for using this is to do away with all the heuristics in traditional (MATLAB-based) DSP approaches to GIF, and rather present them via IKLP as testable hypotheses, where the hypotheses, we shall see in the coming chapters, are motivated theoretically as nonparametric priors and then refined on synthesized data.

I remind you again that the goal is a comprehensive report, such that a reader could implement it all on their own from scratch. Your prose therefore mustn't be too terse. Take your time as this could be quite lengthy. Make sure to pay attention to themes I highlighted. If you have any questions, shoot.
*/

/*
... CODEX ASKS QUESTIONS AND COMES UP WITH A PLAN
*/

/*
AFTER SEEING CODEX'S PLAN:

Excellent! Indeed you may add complexity analysis (mem and time) of the two ways (naive vs Cholesky), and you may also add that we indeed generalized the prior for the AR coefficients to arbitrary MVN( mean, covariance ) as we will address that in the next chapter. It's a good plan but make sure to refer every once in a while to my original (first) prompt to "resniff the spirit" of what I'm after here. And you may also add that this is a highly obscure paper which somehow has only a few citations, and that it is perfect for a Bayesian-though-efficient attempt at GIF. Finally take care of wording wrt. excitation e(t) = s(t) + neta(t) (neta being noise) and that s(t) can be identified with u'(t), though in this thesis we will always use u'(t) as the "true"/"theoretical" DGF and e(t) as the inferred one. Indeed during voiced regions we USE e(t) NOT s(t) AS OUR ESTIMATE OF THE DGF as ordinary LPC (ie s(t) == 0) already recovers u'(t) estimates quite well from neta(t) alone given sufficiently long inference frames. An important observation here is that with this strategy the neta(t) component can "jump in" to provide sharp excitation spikes (ie broadband energy) which are necessary to resolve the resonances (formants) with the AR coefficients and to model the broadband energy injections due to glottal closure instants (GCI) in the vocal apparatus.
*/

/*
from Abstract here: https://jmlr.csail.mit.edu/papers/volume12/gonen11a/gonen11a.pdf

"We see that overall, using multiple kernels instead of a
single one is useful and believe that combining kernels in a nonlinear or data-dependent way seems
more promising than linear combination in fusing information provided by simple linear kernels,
whereas linear methods are more reasonable when combining complex Gaussian kernels."

=> We combine the kernels in a data-dependent linear way, so that 's good according to practice

What's more: there is a superposition/blurring principle at play: clusters of "nearby kernels" that are a posteriori active define a single  "interpolated" kernel
*/

/* zie papieren */

= Infinite kernel linear prediction
<chapter:iklp>

The inference engine at the heart of BNGIF is a nonparametric Bayesian extension of linear prediction due to #pcite(<Yoshii2013>).
The paper, presented over a decade ago at ICASSP 2013, has attracted fewer citations than its conceptual depth warrants — which, for our purposes, is fortunate, because it provides exactly the principled Bayesian foundation we need to replace the DSP heuristics of traditional glottal inverse filtering with a framework of testable probabilistic hypotheses.
This chapter develops the model in full.

We proceed in three steps that mirror the paper's own structure: a classical probabilistic formulation of LP, a kernelized generalization in which the excitation is replaced by a Gaussian process, and the full Bayesian nonparametric limit in which the kernel is itself inferred from the data.
To these we add two contributions of our own: a Woodbury-Cholesky reformulation of the inference equations that avoids all explicit $M times M$ matrix inversions, and two reparameterizations of the prior hyperparameters into quantities with direct physical interpretations.

== Linear prediction
<sec:lp>

Let $bm(x) = (x_1, dots, x_M)^top in bb(R)^M$ be $M$ consecutive speech samples drawn from a short analysis frame.
The autoregressive (AR) model of order $P$ asserts that each sample is a linear combination of the $P$ most recent ones plus an excitation residual:
$
  x_m = sum_(p=1)^P a_p x_(m-p) + epsilon_m.
$ <eq:ar>
The vector $bm(a) = (a_1, dots, a_P)^top in bb(R)^P$ collects the _predictor coefficients_ and $bm(epsilon) = (epsilon_1, dots, epsilon_M)^top$ is the _excitation_.
In source-filter terms these play distinct roles: $bm(epsilon)$ represents the glottal excitation and $bm(a)$ encodes the resonance structure of the vocal tract.

Two matrices appear repeatedly.
The _AR operator_ $bm(Psi)(bm(a)) in bb(R)^(M times M)$ is approximately lower-triangular with unit diagonal and sub-diagonal entries given by $-a_1, dots, -a_P$; it satisfies $bm(epsilon) = bm(Psi) bm(x)$, or equivalently $bm(x) = bm(Psi)^(-1) bm(epsilon)$.
The _lag matrix_ $bm(X) in bb(R)^(M times P)$ has $m$-th row $(x_(m-1), dots, x_(m-P))$ and is the design matrix for the normal equations.

The classical probabilistic LP model assumes white Gaussian excitation,
$
  bm(epsilon) ~ mono("Normal")(bm(0), nu bm(I)),
$ <eq:white-excitation>
which yields the likelihood
$
  bm(x) ~ mono("Normal")(bm(0), nu bm(Psi)^(-1) bm(Psi)^(-top)).
$ <eq:lp-likelihood>
Maximum-likelihood estimation under @eq:lp-likelihood yields the familiar normal equation $bm(X)^top bm(X) bm(a) = bm(X)^top bm(x)$, which is what every standard LPC implementation solves.

The model works as designed for speech with broadband, noise-like excitation.
It fails for voiced speech. // TODO: It nonetheless works quite well for voiced speech
During voiced phonation the excitation is periodic, not white, and fitting @eq:lp-likelihood to a pitched signal forces the estimated envelope to track the harmonic partials of the spectrum: the poles of the all-pole filter chase the pitch harmonics rather than the formants.
This is the central defect that motivates everything that follows.

== Kernel linear prediction
<sec:klp>

The fix is to replace the white-noise excitation model with a richer one.
We model the excitation as a continuous function $epsilon(t)$ over time, approximated by a weighted sum of $J$ basis functions plus a broadband residual:
$
  epsilon(t) = phi(t)^top bm(w) + eta(t),
$ <eq:excitation-regression>
where $phi(t) = (phi_1(t), dots, phi_J(t))^top$ and $eta(t)$ is the residual.
Sampling at times $t_1, dots, t_M$ and writing $bm(Phi) in bb(R)^(M times J)$ for the design matrix with rows $phi(t_m)^top$ gives $bm(epsilon) = bm(Phi) bm(w) + bm(eta)$ in vector form.

With independent Gaussian priors
$
  bm(w) ~ mono("Normal")(bm(0), nu_w bm(I)),
  quad
  bm(eta) ~ mono("Normal")(bm(0), nu_e bm(I)),
$ <eq:klp-priors>
marginalizing over $bm(w)$ yields
$
  bm(epsilon) ~ mono("Normal")(bm(0), nu_w bm(K) + nu_e bm(I)),
  quad bm(K) = bm(Phi) bm(Phi)^top.
$ <eq:klp-excitation>
The matrix $bm(K)$ is a _kernel matrix_ with entries $K_(m m') = phi(t_m)^top phi(t_(m'))$, and the key observation of #pcite(<Kameoka2010>) is that any positive semidefinite matrix can serve as $bm(K)$ directly via the kernel trick — $K_(m m') = k(t_m, t_(m'))$ for some kernel function $k$ — without constructing the basis functions at all.

Substituting @eq:klp-excitation into $bm(x) = bm(Psi)^(-1) bm(epsilon)$ gives the kernelized LP likelihood,
$
  bm(x) ~ mono("Normal")(bm(0), bm(Psi)^(-1)(nu_w bm(K) + nu_e bm(I)) bm(Psi)^(-top)).
$ <eq:klp-likelihood>
Classical LP @eq:lp-likelihood is the special case $bm(K) = bm(I)$ with $nu = nu_w + nu_e$.
A periodic kernel with period $T$ concentrates excitation power at harmonics and thereby _decorrelates_ the envelope estimate from the pitch structure — exactly the fix we needed.

==== Excitation semantics
<sec:excitation-semantics>
It is worth being precise about how the structured and broadband components of $bm(epsilon)$ map onto the GIF picture.
We write the excitation as $epsilon(t) = s(t) + eta(t)$, where $s(t) = phi(t)^top bm(w)$ is the structured GP component and $eta(t)$ is broadband noise, and we distinguish this from $u'(t)$, the true physical DGF.
In this thesis $u'(t)$ is the theoretical DGF in the sense of source-filter theory, and inferred $epsilon(t)$ is our practical DGF estimate.
During voiced regions these are close but not equal, and we always report $epsilon(t)$ rather than $s(t)$ alone as our output.

This is deliberate.
Even ordinary LPC — the $s = 0$ limit where all excitation comes from $eta(t)$ — already recovers recognizable DGF estimates from long enough analysis frames, because $eta(t)$ is free to inject the sharp broadband transients needed to excite all formant resonances simultaneously.
Those sharp transients also model the broadband energy injections associated with glottal closure instants in the vocal apparatus.
Extending from LPC to IKLP enriches what $s(t)$ can model but leaves $eta(t)$ active and free to do this useful work — which is precisely what we want.

== Infinite kernel linear prediction
<sec:iklp>

=== From a fixed kernel to an infinite hypothesis bank

The kernelized model @eq:klp-likelihood requires designing $bm(K)$ to match the excitation structure, but for voiced speech the fundamental period $T$ is itself unknown and must be inferred.
The natural Bayesian response is multiple kernel learning: prepare a bank of $I$ candidate kernels $bm(K)_1, dots, bm(K)_I$ and define the excitation kernel as their weighted sum,
$
  bm(K) = sum_(i=1)^I theta_i bm(K)_i, quad theta_i >= 0.
$ <eq:mkl-kernel>
Each weight $theta_i >= 0$ quantifies the degree to which hypothesis $i$ explains the excitation.
In the original formulation of #pcite(<Yoshii2013>), the bank indexes candidate fundamental periods $T_1, dots, T_I$ on a grid, so the weights $theta_i$ directly encode a posterior over F0 candidates.

This is an important but too-narrow framing.
The index $i$ is a generic _hypothesis axis_: it can index candidate periods, candidate glottal flow morphologies, candidate phonation modalities, or any structured excitation model that can be expressed as a kernel.
We will exploit this generality systematically in later chapters, where the bank is populated with GP priors learned from synthetic DGF data.
The resulting multiple-kernel likelihood is
$
  bm(x) ~ mono("Normal")(bm(0), bm(Psi)^(-1) (nu_w sum_(i=1)^I theta_i bm(K)_i + nu_e bm(I)) bm(Psi)^(-top)).
$ <eq:mklp-likelihood>

MAP estimation of $bm(theta)$ under a regularizing prior does not yield truly sparse weights.
#pcite(<Yoshii2013>) solve this by taking $I -> infinity$ and placing a _gamma process_ (GaP) prior on the infinite weight sequence: for any finite truncation level $I$,
$
  theta_i ~ mono("Gamma")(alpha\/I, alpha),
$ <eq:gap-prior>
and as $I -> infinity$ the vector $bm(theta)$ converges to a draw from a GaP with concentration $alpha$.
It is proven that the effective number of active elements — those with $theta_i > epsilon$ for any $epsilon > 0$ — is almost surely finite, and for $I >> alpha$ only $cal(O)(alpha)$ weights are substantially positive.
Sparsity is a theorem about the prior, not an approximation.

The full IKLP likelihood is
$
  bm(x) ~ mono("Normal")(bm(0), bm(Psi)^(-1) (nu_w sum_(i=1)^infinity theta_i bm(K)_i + nu_e bm(I)) bm(Psi)^(-top)).
$ <eq:iklp-likelihood>
This single model simultaneously infers the spectral envelope through $bm(a)$, the fundamental frequency through the dominant $theta_i$, and voiced/unvoiced status through the ratio $bb(E)[nu_w] \/ bb(E)[nu_w + nu_e]$ — all at once, in a principled Bayesian framework, with none of the separate heuristic steps that characterize classical DSP approaches to GIF.#footnote[
  Setting $P = 0$ is also valid: the AR operator becomes $bm(Psi) = bm(I)$ and IKLP reduces to a pure multi-kernel Bayesian inference engine over the excitation process with no envelope component.
  This is useful as a standalone pitch detector or phonation-type classifier.
]

== Inference
<sec:iklp-inference>

=== Priors and variational family

To complete the model we place priors on the positive scale parameters and on $bm(a)$.
Following #pcite(<Yoshii2013>), the scales receive gamma priors,
$
  nu_w ~ mono("Gamma")(a_w, b_w),
  quad
  nu_e ~ mono("Gamma")(a_e, b_e).
$ <eq:scale-priors>
For the AR coefficients we generalize the isotropic prior $bm(a) ~ mono("Normal")(bm(0), lambda bm(I))$ of the original paper to a full multivariate Gaussian,
$
  bm(a) ~ mono("Normal")(bm(mu)_a, bm(Sigma)_a),
  quad bm(Q) := bm(Sigma)_a^(-1),
$ <eq:ar-prior>
with precision matrix $bm(Q)$.
The isotropic case is $bm(mu)_a = bm(0)$, $bm(Q) = lambda^(-1) bm(I)$.
We need the full form in @chapter:arprior, where informative priors derived from stability theory and spectral constraints replace the isotropic default.

Exact posterior inference is intractable, so we use a mean-field variational family,
$
  q(bm(theta), bm(a), nu_w, nu_e)
  = q(bm(a)) thin q(nu_w) thin q(nu_e) thin product_(i=1)^I q(theta_i),
  quad q(bm(a)) = delta(bm(a) - bm(a)^*),
$ <eq:mean-field>
with $bm(a)$ treated as a MAP point estimate.
Each remaining factor is updated by coordinate ascent on the ELBO $cal(L)$:
$
  log p(bm(x)) >= underbrace(bb(E)[log p(bm(x) | bm(theta), bm(a), nu_w, nu_e)], "expected log-likelihood")
  + sum_z bb(E)[log p(z)] - sum_z bb(E)[log q(z)] =: cal(L).
$ <eq:elbo>

Compared to MCMC-based GIF approaches — where #pcite(<Auvinen2014>) report runtimes on the order of 2.5 hours of single-CPU time per 25-millisecond analysis frame — variational inference keeps the Bayesian treatment while reducing inference to a sequence of closed-form updates.
CAVI is convergence-guaranteed, parallelizable over batches of frames, and compatible with modern autodiff and GPU workflows even when the update steps are computed analytically.

=== Tractable lower bound via matrix inequalities

The expected log-likelihood involves $bb(E)[log |bm(K)|]$ and $bb(E)[bm(x)^top bm(Psi)^top bm(K)^(-1) bm(Psi) bm(x)]$, where $bm(K) = nu_w sum_i theta_i bm(K)_i + nu_e bm(I)$, and neither expectation is tractable in closed form.
Two matrix-variate inequalities provide bounds.

The log-determinant $log |bm(V)|$ is concave, so a first-order Taylor expansion around any PSD matrix $bm(Omega)$ gives
$
  log |bm(V)| <= log |bm(Omega)| + "tr"(bm(Omega)^(-1) bm(V)) - M.
$ <eq:logdet-bound>

The quadratic form $bm(z)^top bm(V)^(-1) bm(z)$ is convex, and an inequality of #pcite(<Sawada2012>) gives
$
  bm(z)^top (sum_i bm(V)_i)^(-1) bm(z)
  <=
  sum_i bm(z)^top bm(Upsilon)_i^top bm(V)_i^(-1) bm(Upsilon)_i bm(z),
$ <eq:quadratic-bound>
where $\{bm(Upsilon)_i\}$ are auxiliary matrices summing to the identity.

Applying both inequalities to the expected log-likelihood and optimizing the auxiliary quantities in closed form gives the tractable lower bound $cal(L)' <= cal(L)$, with optimal values
$
  bm(Omega) = bb(E)[nu_w] sum_i bb(E)[theta_i] bm(K)_i + bb(E)[nu_e] bm(I),
$ <eq:Omega>
$
  bm(Upsilon)_i = tilde(w)_i bm(K)_i cal(S)^(-1),
  quad
  bm(Upsilon)_0 = tilde(nu)_e cal(S)^(-1),
$ <eq:Upsilon>
where the _quadratic operator_ $cal(S)$ and harmonic-mean weights are defined by
$
  cal(S) = sum_i tilde(w)_i bm(K)_i + tilde(nu)_e bm(I),
  quad
  tilde(nu)_e = 1 \/ bb(E)[nu_e^(-1)],
  quad
  tilde(w)_i = 1 \/ (bb(E)[nu_w^(-1)] bb(E)[theta_i^(-1)]).
$ <eq:S-operator>
Both $bm(Omega)$ and $cal(S)$ have the form $nu bm(I) + sum_i w_i bm(K)_i$; they differ only in which moment or harmonic-mean expectations enter as weights.
This shared algebraic structure is what the Woodbury reformulation of @sec:woodbury exploits.

=== GIG posteriors and CAVI updates

The sufficient statistics of a gamma prior are $log z$ and $z$; the bounds above introduce terms in $z$ and $1\/z$.
The optimal variational posteriors for $theta_i$, $nu_w$, $nu_e$ therefore all take the form of a _generalized inverse Gaussian_,
$
  mono("GIG")(z | gamma, rho, tau) prop z^(gamma-1) exp(-1\/2 (rho z + tau\/z)), quad z > 0,
$ <eq:gig>
with moments
$
  bb(E)[z] = sqrt(tau\/rho) thin K_(gamma+1)(sqrt(rho tau)) \/ K_gamma(sqrt(rho tau)),
  quad
  bb(E)[z^(-1)] = sqrt(rho\/tau) thin K_(gamma-1)(sqrt(rho tau)) \/ K_gamma(sqrt(rho tau)),
$ <eq:gig-moments>
where $K_gamma$ is the modified Bessel function of the second kind.

With the LP residual $bm(r) := bm(Psi)(bm(a)) bm(x)$, the weighted residual $bm(w) := cal(S)^(-1) bm(r)$, and the per-kernel quadratic statistics
$
  bm(u)_i := bm(Phi)_i^top bm(w),
  quad q_i := ||bm(u)_i||^2,
  quad q_0 := ||bm(w)||^2,
$ <eq:quad-stats>
the CAVI parameter updates take the compact form.
For the kernel weights:
$
  gamma_(theta_i) = alpha\/I,
  quad
  rho_(theta_i) <- 2alpha + bb(E)[nu_w] "tr"(bm(Omega)^(-1) bm(K)_i),
  quad
  tau_(theta_i) <- (1\/bb(E)[nu_w^(-1)]) (1\/bb(E)[theta_i^(-1)]^2) q_i.
$ <eq:theta-update>
For the structured scale:
$
  gamma_w = a_w,
  quad
  rho_w <- 2b_w + sum_i bb(E)[theta_i] "tr"(bm(Omega)^(-1) bm(K)_i),
  quad
  tau_w <- sum_i (1\/bb(E)[nu_w^(-1)]^2) (1\/bb(E)[theta_i^(-1)]) q_i.
$ <eq:nuw-update>
For the broadband scale:
$
  gamma_e = a_e,
  quad
  rho_e <- 2b_e + "tr"(bm(Omega)^(-1)),
  quad
  tau_e <- (1\/bb(E)[nu_e^(-1)]^2) q_0.
$ <eq:nue-update>
The AR update is MAP under the generalized prior @eq:ar-prior, giving the normal equation
$
  (bm(X)^top cal(S)^(-1) bm(X) + bm(Q)) bm(a)^* = bm(X)^top cal(S)^(-1) bm(x) + bm(Q) bm(mu)_a.
$ <eq:ar-update>
All updates are iterated until the ELBO converges or a maximum iteration count is reached.

The dominant costs per iteration are the $I$ trace terms $"tr"(bm(Omega)^(-1) bm(K)_i)$ and the operator actions $cal(S)^(-1) bm(r)$ and $cal(S)^(-1) bm(X)$, all requiring solutions to linear systems in matrices of the form $nu bm(I) + sum_i w_i bm(K)_i$.
The next section reduces all of these to triangular solves on a single small matrix.

== Woodbury-Cholesky reformulation
<sec:woodbury>

Both $bm(Omega)$ and $cal(S)$ are instances of the abstract operator
$
  bm(S)(nu, bm(w)) := nu bm(I)_M + sum_(i=1)^I w_i bm(K)_i in bb(R)^(M times M).
$ <eq:S-generic>
The direct approach materializes this $M times M$ matrix, factors it by Cholesky ($cal(O)(M^3)$), and solves systems against it ($cal(O)(M^2)$ each).
For $M = 2048$ samples at $I = 400$ kernels this is expensive; for large frames it is prohibitive.

Suppose each kernel admits a low-rank factorization $bm(K)_i = bm(Phi)_i bm(Phi)_i^top$ with $bm(Phi)_i in bb(R)^(M times r_i)$.
In the settings we care about this is exact: the periodic kernels of the original paper are parameterized by Fourier features, and the arc cosine GP priors of later chapters are defined by their feature matrices $bm(Phi)_i$.
Form the weighted stacked factor
$
  tilde(bm(Phi)) := [sqrt(w_1) bm(Phi)_1, dots, sqrt(w_I) bm(Phi)_I] in bb(R)^(M times L),
  quad L = sum_i r_i,
$ <eq:stacked-factor>
so that $bm(S)(nu, bm(w)) = nu bm(I)_M + tilde(bm(Phi)) tilde(bm(Phi))^top$.
The Woodbury matrix identity gives its inverse as
$
  bm(S)^(-1) bm(v)
  = nu^(-1) bm(v) - nu^(-2) tilde(bm(Phi)) bm(A)^(-1) tilde(bm(Phi))^top bm(v),
  quad
  bm(A) := bm(I)_L + nu^(-1) tilde(bm(Phi))^top tilde(bm(Phi)) in bb(R)^(L times L).
$ <eq:woodbury-solve>
The $L times L$ _core matrix_ $bm(A)$ is the only object that needs to be factored.
Its Cholesky decomposition $bm(A) = bm(R)^top bm(R)$ — computed once per operator instance per CAVI iteration — makes the following quantities available through triangular solves alone.

==== Operator solves
To compute $bm(S)^(-1) bm(v)$: form $bm(t) = tilde(bm(Phi))^top bm(v) in bb(R)^L$, solve the triangular systems $bm(R)^top bm(y) = bm(t)$ then $bm(R) bm(u) = bm(y)$ to get $bm(u) = bm(A)^(-1) bm(t)$, then return $nu^(-1) bm(v) - nu^(-2) tilde(bm(Phi)) bm(u)$.
No explicit inverse is formed.

==== Log-determinant
By the matrix determinant lemma, $|bm(S)| = nu^M |bm(A)|$, so
$
  log |bm(S)| = M log nu + 2 sum_(ell=1)^L log R_(ell ell).
$ <eq:woodbury-logdet>

==== Trace of inverse
$
  "tr"(bm(S)^(-1)) = (M - L) \/ nu + nu^(-1) "tr"(bm(A)^(-1)).
$ <eq:woodbury-trace>

==== Per-kernel trace
Define $bm(B)_i := tilde(bm(Phi))^top bm(Phi)_i in bb(R)^(L times r_i)$.
Then
$
  "tr"(bm(S)^(-1) bm(K)_i)
  = nu^(-1) ||bm(Phi)_i||_F^2 - nu^(-2) ||bm(R)^(-top) bm(B)_i||_F^2,
$ <eq:woodbury-pertrace>
where $||bm(R)^(-top) bm(B)_i||_F^2$ is the squared Frobenius norm of the solution to $bm(R)^top bm(Z) = bm(B)_i$.

==== AR normal equation
With $cal(S)^(-1)$ available as a triangular-solve operator, the $P times P$ Gram matrix $bm(G) = bm(X)^top cal(S)^(-1) bm(X)$ is assembled column-by-column, and @eq:ar-update is then a single $P times P$ Cholesky solve of $bm(G) + bm(Q)$.

Every dense operation in the CAVI loop has been reduced to triangular solves on matrices of size at most $max(L, P) << M$.

=== Complexity

Let $M$ be the frame length, $P$ the AR order, $I$ the number of kernels, $r$ the rank per kernel, $L = I r$ the total factor dimension, and $B$ the number of frames in a batch.

In the dense baseline, building $bm(S)$ costs $cal(O)(I M^2 r)$, factoring it costs $cal(O)(M^3)$, each solve costs $cal(O)(M^2)$, and storing it requires $cal(O)(M^2)$ memory.

In the Woodbury-Cholesky regime, the core matrix $bm(A)$ is assembled in $cal(O)(M L^2)$ and factored in $cal(O)(L^3)$.
Each operator solve costs $cal(O)(M L + L^2)$, the log-determinant costs $cal(O)(L)$ after factorization, the per-kernel traces cost $cal(O)(L^2 r)$ each, and memory drops to $cal(O)(M L + L^2)$: the $M times M$ covariance is never stored.
The crossover is favorable whenever $L << M$, precisely the regime of interest.

For $B$ frames the loop vectorizes linearly in $B$.
Since no dense $M times M$ matrix is ever formed, the formulation is insensitive to its conditioning.
Running in 32-bit floating point gives a substantial throughput gain on accelerators while retaining numerical stability through the Cholesky factorization and a small diagonal jitter on $bm(A)$.

== Prior reparameterizations
<sec:iklp-priors>

The priors @eq:gap-prior and @eq:scale-priors are written in shape-rate form, natural for conjugate analysis but opaque for prior elicitation.
We reparameterize both into quantities with direct physical interpretations.

=== Kernel sparsity: $alpha$-calibration

Under @eq:gap-prior, define $T := sum_i theta_i$ and the normalized weights $p_i = theta_i \/ T$.
For any finite truncation $I$, the sum $T ~ mono("Gamma")(alpha, alpha)$ has $bb(E)[T] = 1$ and $"sd"(T) = alpha^(-1/2)$, while the proportions follow a symmetric Dirichlet:
$
  (p_1, dots, p_I) ~ mono("Dirichlet")(alpha\/I, dots, alpha\/I).
$
Small $alpha$ concentrates mass toward the vertices of the probability simplex — one dominant hypothesis — while large $alpha$ spreads it uniformly.

A natural summary of effective sparsity is the exponentiated Shannon entropy,
$
  I_"eff" := exp(bb(E)[H(bm(p))]),
  quad H(bm(p)) = -sum_i p_i log p_i,
$ <eq:ieff>
which counts the number of equally probable hypotheses that would produce the same entropy.
Since the Dirichlet expectation of $H$ is known analytically,
$
  bb(E)[H(bm(p))] = psi(alpha + 1) - psi(1 + alpha\/I),
$ <eq:entropy-exact>
we obtain
$
  I_"eff"(alpha; I) = exp(psi(alpha + 1) - psi(1 + alpha\/I)).
$ <eq:ieff-formula>

As $I -> infinity$ with $alpha$ fixed, $I_"eff"$ saturates at the constant $exp(psi(alpha + 1) + gamma_"EM")$ where $gamma_"EM"$ is the Euler-Mascheroni constant.
At $alpha = 1$ and large $I$ this gives $I_"eff" -> e approx 2.72$, meaning roughly a handful of simultaneously active hypotheses on average.
This matches what #pcite(<Yoshii2013>) observe empirically at $alpha = 1$, $I = 400$.

To calibrate $alpha$ to a desired effective count $I_"eff"^0$, one solves @eq:entropy-exact for $alpha$; the solution $alpha^*(I)$ approaches 1 rapidly for $I_"eff"^0 = e$ as $I$ grows.
The local sensitivity of $I_"eff"$ to $alpha$ near this calibration is described by the log-elasticity
$
  b(alpha; I) := frac(partial log I_"eff", partial log alpha) = alpha [psi_1(alpha + 1) - I^(-1) psi_1(1 + alpha\/I)],
$ <eq:elasticity>
where $psi_1$ is the trigamma function.
As $I -> infinity$ this approaches $b -> psi_1(2) = pi^2\/6 - 1 approx 0.645$, giving the local approximation
$
  I_"eff"(alpha; I) approx e (alpha \/ alpha^*(I))^(0.645).
$ <eq:ieff-powerlaw>
So $alpha$ behaves like a concentration parameter whose effect on sparsity is governed by a fixed exponent near the default calibration.

=== Excitation power decomposition: the $(p, s, kappa)$-reparameterization

When $b_w = b_e =: b$, the total excitation power $nu_w + nu_e ~ mono("Gamma")(a_w + a_e, b)$ and the structured fraction $pi := nu_w \/ (nu_w + nu_e) ~ mono("Beta")(a_w, a_e)$ are independent of each other.
This motivates the reparameterization
$
  a_w = kappa p,
  quad
  a_e = kappa (1 - p),
  quad
  b_w = b_e = kappa \/ s,
$ <eq:reparam-map>
under which the total power has prior $nu_w + nu_e ~ mono("Gamma")(kappa, kappa\/s)$ with
$
  bb(E)[nu_w + nu_e] = s,
  quad
  "Var"(nu_w + nu_e) = s^2 \/ kappa,
$
and the structured fraction has prior $pi ~ mono("Beta")(kappa p, kappa(1-p))$ with
$
  bb(E)[pi] = p,
  quad
  "Var"(pi) = p(1-p) \/ (kappa + 1).
$
The parameter $kappa$ controls concentration only — it tightens both distributions simultaneously without moving their means.
The parameters $(p, s)$ directly control the prior expected values: $s$ is calibrated to the expected dynamic range of the analysis frame, and $p$ reflects prior beliefs about phonation type.
A frame expected to be clearly voiced might receive $p = 0.8$; an unvoiced prior might use $p = 0.1$.

The default of #pcite(<Yoshii2013>), $(a_w, a_e, b_w, b_e) = (1, 1, 1, 1)$, corresponds to $(p, s, kappa) = (1\/2, 2, 2)$: equal prior probability of structured and broadband excitation, total power expected at 2, with moderate concentration.
A unit-power neutral default is $(p, s, kappa) = (1\/2, 1, 1)$.

== Summary

IKLP generalizes classical LP in three directions at once.
The white excitation assumption is replaced by a GP prior whose kernel is a sparse mixture learned from the data.
The single scalar noise variance is decomposed into structured and broadband components whose ratio tracks voiced/unvoiced status.
And the AR prior is generalized to a full multivariate Gaussian, ready to accept the informative spectral priors of @chapter:arprior.

All of this fits inside a single CAVI loop with closed-form GIG updates and a convergence guarantee.
The Woodbury-Cholesky reformulation of @sec:woodbury removes the cubic bottleneck, reducing every inner-loop computation to triangular solves on matrices of size $max(L, P) << M$.
This makes the method both fast and composable: the kernel bank $\{bm(Phi)_i\}$ is the only interface between the inference engine and the prior on the excitation.
Replacing the periodic kernels of the original paper with the arc cosine GP priors of later chapters requires no change to the inference architecture — only the features $bm(Phi)_i$ change.

The GIF problem of @chapter:overview requires finding plausible source-filter pairs consistent with a speech observation.
IKLP, equipped with the right kernel bank, is exactly a hypothesis-testing engine for that problem.