= The $arccos$ glottal flow model

We will characterize the $arccos$ glottal flow model (AGFM) in the spectral domain, as these models are best described there. Plus, at DC frequency we can already impose the null integral constraint.

In principle we can go any depth in the calculation I think.

Can we do arbitrary covariance $Sigma$ then?

== FT of an affine warp $g(t) = f(a t + b)$

*Choose domain of integration as $(-1,1)$ and calculate once; then use affine scaling to move and stretch the open phase anywhere; this requires different sample points -- cannot be avoided if we keep the fix the domain once.*

Assume the continuous-time FT
$
  F(omega) := integral_(-oo)^(oo) f(t) exp(-i omega t) dif t.
$

Then, for a != 0 and g(t) = f(a t + b),
$
  G(omega)
  = integral_(-oo)^(oo) g(t) exp(-i omega t) dif t
  = 1 / abs(a) * exp(i omega b / a) * F(omega / a).
$

If you use the $2 pi$ convention
$
  F_2pi(nu) := integral_(-oo)^(oo) f(t) exp(-i 2 pi nu t) dif t,
$
then
$
  G_2pi(nu) = 1 / abs(a) * exp(i 2 pi nu b / a) * F_2pi(nu / a).
$

Reason (one line): substitute u = a t + b so t = (u - b)/a and dif t = dif u / a.
When a < 0 the bounds flip, producing 1/abs(a). The factor exp(i omega b / a) comes from
exp(-i omega t) with t = (u - b)/a.

Support bookkeeping (if f(t)=0 outside (-1,1)):
$
  "support"(g) = { t : a t + b in (-1, 1) } = ( (-1 - b)/a, (1 - b)/a ),
$
with endpoints swapped if a < 0.


== Strategy for $H_(m , n) (f)$
<strategy-for-h_mnf>
$ H_(m , n) (f) = integral_a^b (1 + x^2)^(n \/ 2) e^(i m arctan x) e^(- 2 i pi f x) dif x $

where $m in bb(Z)$ and $n in bb(N)_(gt.eq 0)$. Note immediately that
$H_(- m , n) (f) = overline(H_(m , n) (- f))$. Set $theta = arctan x$,
then

$ H_(m , n) (f) = integral_(theta_1)^(theta_2) sec^(n + 2) e^(i m theta) e^(- 2 i pi f tan x) d theta $

In general:

$ H_(m,n)(f) = \[ + (H\_{m,n-1}(f))\] $

where the constant is:

$
  \-^n() e^{im} e^{-i 2 f } \_{\_1}^{\_2}
$

where $(theta_1 , theta_2) = (arctan a , arctan b)$. This is on
notes p.7 and p.~8. You can see clearly the derivative operator
$frac(1, i 2 pi f)$ connecting the orders $n$ to $n - 1$, but in a
nontrivial manner. This also shows the differentiability of the sample
paths.

We have the general recusion formula in $n$, so only need to calculate
$(n , m) = (0 , 0) , (0 , plus.minus 1) , (0 , plus.minus 3) , dots.h$
and in addition can choose all $m gt.eq 0$ because of reflection in
$- f$: $H_(- m , n) (f) = overline(H_(m , n) (- f))$.

We can normally calculate $H_(m gt.eq 0 , 0) (f)$ via expansion in
$theta$ such that it becomes a sum of incomplete Beta functions:
#link("./Derivation%20with%20exp(-%20i%202pi%20f%20tan%20theta).pdf")[link to pdf];/#link("https://chatgpt.com/c/68542dc8-30bc-8011-b8bd-91e3c6a37ef4")[link to o3 chat, but choose the second branch at question 3];.
#link("https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.betainc.html#jax.scipy.special.betainc")[And link to JAX implementation of betainc];.
Or we can use direct numerical integration (next paragraph). Alternative
routes in terms of $partial_f^r$ of Bessel $K$ functions must assume
$a = - oo , b = oo$ and are less convenient.

#strong[Numerical integration];: Becomes more difficult for higher
$omega$. But we can use Gamma integral expansion to integrate out the
oscillating part and the integral reduces to the generalized Gauss
Laguerre type. Note that this probably just needs to be pre-calculated
once, tho im not 100% sure.
