#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

/*
All steps are in:
https://chatgpt.com/c/690cb0c2-0264-8327-a5d2-568620b33956

Next steps:
Derive PACK and use in OPENGLOTI and OPENGLOTII experiments and compare to PeriodicSqExp

Then learn features from LF model and see if evaluation improves with this
*/

= The periodic arc cosine kernel
<chapter:pack>

/* NORMALIZED KERNELS

Expression is always k(x,x') = J(x,x')/J(0)
Samples tend to horizontal asymptotes left and right
Need to look into this
Not a priori better I think cos DGF has non-horizontal asymptotes usually for hard GCI

*/

The periodic arc cosine kernel (PACK)

We will characterize the $arccos$ glottal flow model (AGFM) in the spectral domain, as these models are best described there. Plus, at DC frequency we can already impose the closure constraint.

In principle we can go any depth in the calculation I think.


== Periodizing the TACK

Defined on open phase

Note $t_c != T$

Like @Chen2016, Chapter 5 we make use of periodic property

Thus in the time domain the function can be synthesized as a KL expansion:
$
  u'(t) = sum_k c_k exp(i 2 pi k)
$ // FIXME

It thus remains to find the FT of the kernel


/*
==== Sampling frequencies
Our sampling of freqs in the Nyquist band reproduces the Sinc kernel-property that we automatically band limit the smoothness => good prior information

Nyquist reconstruction essentially expresses that by bandlimiting frequency information we limit smoothness, thereby making perfect reconstruction possible in principle.

Multiplying with SqExp with low order Hilbert amplitude envelope doesn't add high freq information, neither does summing with slowly varying DC background, so principle is still preserved. However, white noise delta reintroduces full spectrum!

Matern kernels and others do not take into account Nyquist information, whereas RFF (random fourier features) / SM (spectral mixture) kernels do.
*/

/*
==== Periodic but nonstationary
Note that kernels can be periodic but nonstationary, like ours.
Typically we think in terms of periodic stationary kernels, but this does not have to be always so.
See example in Marocco notebook.
*/



=== Complex GPs


For any complex random variable $z = x + i y$, the second-order information
(the shape of its density if Gaussian) lives entirely in the second moments of
$x$ and $y$. Those are captured by the $2 times 2$ real covariance matrix:
$
  bm(Sigma) = mat(
    E[x^2], E[x y];
    E[x y], E[y^2]
  )
$
But it's convenient and natural in complex analysis to rewrite this in
complex coordinates. There, the two invariant quadratic forms you can make
from $z$ are $E[z z^*]$ and $E[z z]$. The first is the usual covariance (energy
or total power), the second measures the departure from circular symmetry
(orientation or eccentricity).

They're _complementary_ because together they completely reconstruct the real
$2 times 2$ covariance:
$
  bm(Sigma) = 1/2 mat(
    Re(&bb(E)[z z^* + z z]), Im(&bb(E)[z z]);
    Im(&bb(E)[z z]), Re(&bb(E)[z z^* - z z])
  ).
$
They're _independent_ (in the representational sense) because $E[z z^*]$ is
invariant under rotation $z -> e^(i theta) z$ (it sets the overall scale), while $E[z z]$
transforms like $e^(i 2 theta)$: $E[z z] -> e^(i 2 theta) E[z z]$ — it tracks the anisotropy's orientation and
ellipticity.

== The Fourier bitransform of the STACK

This Section is devoted to calculating the compact Fourier bitransform, also known as the Yaglom transform in the context of kernels, of the standard temporal arc cosine kernel:
$
  tilde(k)^((d)) (f,f'; C) = integral.double_C k^((d)) (t, t') thin exp{- i 2 pi f t} exp{i 2 pi f' t'} dif t dif t'
$ <eq:yaglom>
where $C = (t_1, t_2) times (t_1, t_2)$ and $k^((d))$ is the standard temporal arc cosine kernel (TACK).
$
  k^((d)) (t, t') = 1/(2pi) (1+t^2)^(d/2) (1+t'^2)^(d/2) J_d (theta)
$
We need FT over compact domain $C$ because the open phase (cf. @fig:lf) is defined on $[0, t_c]$.
We generalize this to $[t_1, t_2]$.
"Standard" means that the prior variances $sigma_a = sigma_b = sigma_c = 1$.
Later we generalize the calculation by means of an affine transformation of @eq:yaglom.
From the previous Chapter, the TACK is just the 1D bias-augmented version of the arc cosine kernel (ACK) of @Cho2009:
$
  k^((d)) (bm(x), bm(x')) = 1/(2pi) ||bm(x)||^d ||bm(x')||^d J_d (theta)
  
$
where
$
  theta = arccos( (bm(x)^top bm(x'))/(||bm(x)|| ||bm(x')||) ) = "angle between" bm(x) "and" bm(x') in [0,pi]
$
and the $J_d$ expression is given by the generator expression
$
  J_d (theta) = (-1)^d (sin theta)^(2d + 1)
  ( 1 / (sin theta) dif/(dif theta) )^d
  ( (pi - theta) / (sin theta) ).
$
The first four expressions are:
$
  J_0(theta) & = pi - theta \
  J_1(theta) & = sin theta + (pi - theta) cos theta \
  J_2(theta) & = 3 sin theta cos theta + (pi - theta)(1 + 2 cos^2 theta) \
  J_3(theta) & = 15 sin theta - 11 sin^3 theta + (pi - theta)(9 cos theta + 6 cos^3 theta)
$
Here $theta in [0, pi]$ due to image of $arccos$; and these formulas are not valid for $theta < 0$.

So the TACK is just
$
  k^((d)) (t, t') = k^((d)) (bm(x)_t, bm(x)_(t'))
$
with
$
  bm(x)_t = vec(1, t), quad bm(x)_(t') = vec(1, t').
$
Generalizing to arbitrary $bm(Sigma)$:
$
  k^((d))_bm(Sigma) (t, t') = k^((d)) (bm(Sigma)^(1/2) bm(x)_t, bm(Sigma)^(1/2) bm(x)_(t')).
$

/*

We can absorb $sigma_a$ into $Sigma$, as $arccos$ is homogenous for global rescaling, which is equivalent to rescaling $Sigma -> alpha Sigma$. What is excellent: we managed to push one layer of hyperparameters into the amplitudes! Since we marginalized them away, we end up with a nonparametric polynomial DGF model. We have only three hyperparameters: $sigma$, $sigma_1$, $sigma_2$ instead of $O(H)$ amount. This is the key to fast multiple kernel learning.

But we can allow $bm(Sigma) = mat(sigma_b^2, 0; 0, sigma_t^2)$ as this is important to model behavior of kernel. Introducing a third parameter $rho$ (correlation in $Sigma$) breaks our FT derivation (the $tan "/" arctan$ trick), which assumes no correlation between bias and $t$. So individual rescaling is as far as we can go and probably more than enough. So we can proceed with the $N(0,I)$ case as in @Cho2009. // https://chatgpt.com/s/t_68dfa3181bf88191a3183a8138bf2969

*/

=== Harmonic expansion of the zonal kernel
// From FT of arccos kernl.pdf on reMarkable

Define
$
  bm(u)_t = bm(x)_t/(||bm(x)_t||) = vec(1/sqrt(1+t^2), t/sqrt(1 + t^2)) in S^1
$
This vector lies on the circle $S^1$ so we express it as
$
  bm(u)_t = vec(cos psi_t, sin psi_t)
$
with the neat one-to-one correspondence
#footnote[
  Incidentally, this affordance is unique to our problem due to our $(1, t)^top$ instance of the general ACK @eq:ack.
]
$
  psi_t = arctan t in (-pi/2, pi/2).
$


With this parametrization, $theta = arccos("angle between" bm(x)_t "and" bm(x)_(t'))$ reduces to
$
  theta & = arccos bm(u)_t^top bm(u)_(t') \
        & = |psi_t - psi_t'| := |Delta_psi| in [0, pi)
$
/* this needs a circle illustration */
This means that $J_d (theta)$ looks like an isotropic kernel disguise;
$
  J_d (theta) = J_d (|psi_t - psi_t'|) = J_d (|Delta_psi|)
$ <eq:jdtheta>
However, it isn't a proper kernel, because it is not symmetric nor PSD.
Define
$
  J_d^"ext" (theta) = J_d (|theta|)
$
as the natural extension of $J_d (theta)$ by even reflection; this extends the domain $[0,pi]$ to $[-pi,pi]$ and makes it a proper compact kernel.
$J_d^"ext" (theta)$ is an angular zonal kernel: isotropic on $S^1$ @Dutordoir2020 @Smola2000.
Observe that at $theta = plus.minus pi$ we have $J_d^"ext" (theta) = 0$, so gluing copies end-to-end yields a $2 pi$-periodic continuous function for free.
It is an ideal candidate for a harmonic expansion into Fourier series.
Being even also, its Fourier series consists only of cosine terms:
$
  J_d^"ext" (Delta_psi) &= sum_(m in bb(Z)) c_m^((d)) e^(i m Delta_psi) \
  c_m^((d)) &= 1/(2pi) integral_(-pi)^pi J_d^"ext" (Delta_psi) exp{-i m Delta_psi} dif Delta_psi, quad m in bb(Z),
$
where
$
  c^((d))_(-m) = [c^((d))_m]^* = c^((d))_m in bb(R).
$
/*
Such kernels are called _zonal kernels_ @Dutordoir2020.
The translation invariance implies that its Fourier transform is 1D (Bochner's theorem):
$
  J_d (theta) = sum_(m in bb(Z)) c_m^((d)) e^(i m |Delta_psi|)
$
where
$
  c^((d))_(-m) = [c^((d))_m]^* = c^((d))_m in bb(R).
$
*/
where $z^*$ denotes complex conjugate.
These coefficients can be computed analytically and can be precomputed and stored for various values of $d$ once and for all.
Convergence is extremely fast, can get away with $M <= 30$ terms for $d = 0$ and even less for higher $d$.

This is the harmonic expansion of a zonal kernel: a discrete Fourier series, not infinite Fourier transform, because we are on $S^1$.

==== Harmonic expansion for $d = 0$
Slowest convergence at $O(m^(-2))$



=== The $H_m^((d))(f)$ function
Back to @eq:yaglom, where we can now make the substitution $J_d (theta) -> J_d^"ext" (psi_t - psi_t')$ worry-free (the expansion makes sure that $J_d^"ext" ((psi_t - psi_t') in bb(R)) = J_d (theta in [0, pi])$ such that the domain of $J_d(dot) = [0,pi]$ is never violated) and expand into Fourier series:
$
  tilde(k)^((d)) (f,f'; C) &= integral.double_C k^((d)) (t, t') thin exp{- i 2 pi f t} exp{i 2 pi f' t'} dif t dif t' \
  &= integral.double_C (1+t^2)^(d/2) (1+t'^2)^(d/2) sum_(m in bb(Z)) c^((d))_m exp{i m (psi_t - psi_t')} exp{- i 2 pi (f t - f' t')} dif t dif t' \
  &= sum_(m in bb(Z)) c_m^((d)) [integral_(t_1)^(t_2) (1+t^2)^(d/2) exp{i m psi_t} exp{-i 2pi f t} dif t] times \
  &quad quad quad [integral_(t_1)^(t_2) (1+t'^2)^(d/2) exp{-i m psi_t'} exp{i 2pi f' t'} dif t'] \
  &= sum_(m in bb(Z)) c_m^((d)) thin H_m^((d))(f) thin overline(H_m^((d))(f'))
$
This is a _Mercer expansion_ of the $tilde(k)^((d)) (f,f'; C)$ kernel; an inverse Fourier transform of this expression yields a direct Mercer expansion of the original kernel, which allows $O(N)$ inference.

The calculation has been simplified to evaluating
$
  H_m^((d))(f) = integral_(t_1)^(t_2) (1 + t^2)^(d/2) exp{i m arctan t} exp{- i 2 pi f t} dif t
$
where $m in bb(Z)$ and $d in bb(N)_(>=0)$. Note immediately that
$
  H_(-m)^((d))(f) = overline(H_m^((d))(-f)).
$
Set $theta = arctan t$. Then
$
  H_m^((d))(f) = integral_(theta_1)^(theta_2) sec^(d+2) theta
  exp{i m theta} exp{- i 2 pi f tan theta} dif theta,
$
where $(theta_1, theta_2) = (arctan t_1, arctan t_2)$.

In general,
$
  H_m^((d))(f) = (1 / (i 2 pi f))
  [ upright("constant") + upright("LC")(H_(m+1)^((d-1))(f), H_(m-1)^((d-1))(f)) ] ,
$
where the constant is
$
  lr(- sec^d theta exp{i m theta} exp{- i 2 pi f tan theta} |, size: #150%)_(theta_1)^(theta_2) .
$

This exhibits the derivative operator $(i 2 pi f)^(-1)$ linking degree $d$ to $d-1$ (in a nontrivial way) and implies differentiability of order $d$ of the sample paths.

We have a general recursion in $d$, so it suffices to compute the base cases
$
  (d,m) = (0,0), (0,1), (0,3), dots
$
and by reflection in $-f$ we may restrict to $m >= 0$ since
$
  H_(-m,n)(f) = overline(H_m^((d))(-f)).
$

/*
We can usually compute $H_(m>=0,0)(f)$ by expanding in $theta$ to obtain a sum of incomplete Beta functions:
link("./Derivation with exp(- i 2pi f tan theta).pdf")[link to pdf] /
link("https://chatgpt.com/c/68542dc8-30bc-8011-b8bd-91e3c6a37ef4")[link to o3 chat, choose the second branch at question 3] /
link("https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.betainc.html#jax.scipy.special.betainc")[link to JAX implementation of betainc].
Alternatively, one may use direct numerical integration (next paragraph). Routes via $partial_f^r$ of Bessel $K$ functions require $a = -oo, b = oo$ and are less convenient.
*/

*Numerical integration.* Difficulty increases with larger frequency. Using a Gamma-integral expansion to integrate out the oscillatory factor reduces the problem to a generalized Gauss–Laguerre type integral. This likely needs precomputation (to be confirmed).

/* see obsidian: has picture of this */

== The Fourier bitransform of the TACK

=== FT of an affine warp $t mapsto alpha t + beta$

==== General $bm(Sigma)$
With $bm(Sigma) = "diag"(sigma_b^2, sigma_c^2)$ and $bm(x)_t = vec(1, t)$:

- Let $bm(A) = bm(Sigma)^(1/2) = "diag"(sigma_b, sigma_c)$, and define $alpha := sigma_c / sigma_b > 0$.
- Then $bm(A) bm(x)_t = (sigma_b, sigma_c t)^top = sigma_b bm(x)_(alpha t)$.
- For the degree-$d$ arc-cosine kernel (homogeneous of degree $d$ in each argument),
  $
    k^((d))_bm(Sigma)(t, t') = k^((d))(bm(A) bm(x)(t), bm(A) bm(y)(t'))
    = sigma_b^(2d) k^((d))(alpha t, alpha t').
  $

This is the $beta = 0$ case of the affine shift of $t$ formula below.

Can we do arbitrary covariance $bm(Sigma)$?

NO, only diagonal ones with possibly different covariances; mixing terms or higher dimensions do not allow rewriting as an affine transform with a constant outside.

// https://chatgpt.com/s/t_68dfa3181bf88191a3183a8138bf2969


==== Affine shift of $t$
We keep the window $C = (t_1, t_2) times (t_1, t_2)$ general, since phase shifts move its center and we cannot absorb that by shifting the data (we regress multiple kernels against one signal $bm(d)$; works only in the case of a single kernel).

Given
$
  tilde(k)^((d))(f, f'; t_1, t_2)
  = integral.double_((t_1, t_2) times (t_1, t_2))
  k^((d))(t, t')
  exp{- i 2 pi f t} exp{ i 2 pi f' t'} dif t dif t' .
$

Define the shifted–scaled kernel on the same window:
$
  tilde(k)^((d))_(alpha, beta)(f, f'; t_1, t_2)
  := integral.double_((t_1, t_2) times (t_1, t_2))
  k^((d))(alpha t - beta, alpha t' - beta)
  exp{- i 2 pi f t} exp{ i 2 pi f' t'} dif t dif t' ,
$
with $alpha > 0$ and $beta in bb(R)$.

Change variables
$
  u = alpha t - beta, quad u' = alpha t' - beta
$
so that
$
  t = (u + beta) / alpha, quad t' = (u' + beta) / alpha, quad
  dif t dif t' = alpha^(-2) dif u dif u'.
$
The window maps to
$
  (t_1, t_2) mapsto (alpha t_1 - beta, alpha t_2 - beta).
$

Hence
$
  tilde(k)^((d))_(alpha, beta)(f, f'; t_1, t_2)
  = alpha^(-2) exp{- i 2 pi beta (f - f') / alpha}
  integral.double_((alpha t_1 - beta, alpha t_2 - beta) times (alpha t_1 - beta, alpha t_2 - beta)) \
  k^((d))(u, u')
  exp{- i 2 pi (f / alpha) u} exp{ i 2 pi (f' / alpha) u'}
  dif u dif u' .
$

In terms of the unshifted transform evaluated on the transformed window,
$
  tilde(k)^((d))_(alpha, beta)(f, f'; t_1, t_2)
  = alpha^(-2) exp{- i 2 pi beta (f - f') / alpha}
  tilde(k)^((d))(f / alpha, f' / alpha; alpha t_1 - beta, alpha t_2 - beta) .
$

Special cases:
$
  beta = 0: quad tilde(k)^((d))_(alpha, 0)
  = alpha^(-2) tilde(k)^((d))(f / alpha, f' / alpha; alpha t_1, alpha t_2) .
$
$
  alpha = 1: quad tilde(k)^((d))_(1, beta)
  = exp{- i 2 pi beta (f - f')} tilde(k)^((d))(f, f'; t_1 - beta, t_2 - beta) .
$

== Learning Fourier features

== Evaluation on `OPENGLOT-I`

== Summary