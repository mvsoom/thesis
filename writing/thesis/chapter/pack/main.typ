#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

= The periodic arc cosine kernel
<chapter:pack>

The periodic arc cosine kernel (PACK)

We will characterize the $arccos$ glottal flow model (AGFM) in the spectral domain, as these models are best described there. Plus, at DC frequency we can already impose the closure constraint.

In principle we can go any depth in the calculation I think.

/*
Our sampling of freqs in the Nyquist band reproduces the Sinc kernel-property that we automatically band limit the smoothness => good prior information

Nyquist reconstruction essentially expresses that by bandlimiting frequency information we limit smoothness, thereby making perfect reconstruction possible in principle.

Multiplying with SqExp with low order Hilbert amplitude envelope doesn't add high freq information, neither does summing with slowly varying DC background, so principle is still preserved. However, white noise delta reintroduces full spectrum!

Matern kernels and others do not take into account Nyquist information, whereas RFF (random fourier features) / SM (spectral mixture) kernels do.
*/

== The compact Yaglom or compact bitransform

This Section is devoted to calculating the compact Fourier bitransform, also known as the Yaglom transform in the context of kernels, of the temporal arc cosine kernel:
$
  tilde(k)^((d)) (f,f'; C) = integral.double_C k^((d)) (t, t') thin exp{- i 2 pi f t} exp{i 2 pi f' t'} dif t dif t'
$
where $C = (t_1, t_2) times (t_1, t_2)$ and $k^((d))$
/*
$
  bm(tau)(t) = mat(sigma_b, 0; 0, sigma_c) vec(1, t) = vec(sigma_b, sigma_c t) in bb(R)^2
$
*/
is temporal arc cosine kernel (TACK).
$
  k^((d)) (t, t) = 1/pi ||1 + t^2||^(d/2) ||1 + t'^2||^(d/2) J_d (theta)
$
where
$
  bm(x)_t = vec(1, t), quad bm(x)_(t') = vec(1, t').
$
and
$
  k^((d))_bm(Sigma) (t, t') = k^((d)) (bm(Sigma)^(1/2) bm(x)_t, bm(Sigma)^(1/2) bm(x)_(t'))
$
and
$
  theta & = arccos(bm(u)^top bm(u')) \
        & = |arctan(t) - arctan(t')| in [0, pi)
$
and the zonal kernel for RePU degree $d$ is given by @Cho2009
$
  J_d (theta) = (-1)^d (sin theta)^(2d + 1)
  ( 1 / (sin theta) dif/(dif theta) )^d
  ( (pi - theta) / (sin theta) )
$
The first four expressions for the kernels are:
$
  J_0(theta) & = pi - theta \
  J_1(theta) & = sin theta + (pi - theta) cos theta \
  J_2(theta) & = 3 sin theta cos theta + (pi - theta)(1 + 2 cos^2 theta) \
  J_3(theta) & = 15 sin theta - 11 sin^3 theta + (pi - theta)(9 cos theta + 6 cos^3 theta)
$


==== Harmonic expansion of the zonal kernel
Due to the fact that $J_d(theta)$ is isotropic, we have the general zonal expansion
$
  J_d (theta) = sum_(m in bb(Z)) c_m^((d)) e^(i m Delta_psi)
$
where
$
  c^((d))_(-m) = c^((d))_m^* = c^((d))_m in bb(R)
$

== Strategy for $H_(m,n)(f)$

$
  H_(m,n)(f) = integral_a^b (1 + x^2)^(n/2) exp{i m arctan x} exp{- i 2 pi f x} dif x
$

where $m in bb(Z)$ and $n in bb(N)_>=0$. Note immediately that
$
  H_(-m,n)(f) = overline(H_(m,n)(-f)).
$
Set $theta = arctan x$. Then
$
  H_(m,n)(f) = integral_(theta_1)^(theta_2) sec^(n+2) theta
  exp{i m theta} exp{- i 2 pi f tan theta} dif theta,
$
where $(theta_1, theta_2) = (arctan a, arctan b)$.

In general,
$
  H_(m,n)(f) = (1 / (i 2 pi f))
  [ upright("constant") + upright("LC")(H_(m+1,n-1)(f), H_(m-1,n-1)(f)) ] ,
$
where the constant is
$
  - sec^n theta exp{i m theta} exp{- i 2 pi f tan theta} |_(theta_1)^(theta_2) .
$

This exhibits the derivative operator $(i 2 pi f)^(-1)$ linking order $n$ to $n-1$ (in a nontrivial way) and implies differentiability of the sample paths.

We have a general recursion in $n$, so it suffices to compute the base cases
$
  (n,m) = (0,0), (0,1), (0,3), dots
$
and by reflection in $-f$ we may restrict to $m >= 0$ since
$
  H_(-m,n)(f) = overline(H_(m,n)(-f)).
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

== FT of an affine warp $t mapsto alpha t + beta$

==== General $bm(Sigma)$
With $bm(Sigma) = "diag"(sigma_b^2, sigma_c^2)$ and $bm(x)_t = vec(1, t)$:

- Let $bm(A) = bm(Sigma)^(1/2) = "diag"(sigma_b, sigma_c)$, and define $alpha := sigma_c / sigma_b > 0$.
- Then $bm(A) bm(x)(t) = (sigma_b, sigma_c t) = sigma_b bm(x)(alpha t)$.
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
