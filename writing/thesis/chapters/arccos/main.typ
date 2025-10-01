= The $arccos(n)$ kernel as a model for $u'(t)$

In this chapter, we motivate the use of @Cho2009 $arccos(n)$ kernel to describe the glottal flow derivative $u'(t)$.

Look at models for the open phase of the glottal cycle.
Many models, but can be easily unified as per @Doval2006.



== DGF models

// **notebook**: /home/marnix/WRK/proj/sflinear/notebook/single-packet/titze-alku.ipynb



#figure(
  image("./fig/fl-model.png", width: 100%),
  caption: [
    A lineup of GF models, together with their derivatives (DGFs). From @Fujisaki1986.
  ]
)

0th order: @Alku2002


1th order: @Titze2000


/*
![](<./attachments/titze2000.jpg>)

originally here: @Verdolini1995 (pdf paywalled)

![](<./attachments/titze-applications-of-laboratory-formulas-to.jpeg>)

2nd order: KLGLOTT88 (from @Doval2006 A1.1)

3rd order: R++ (from @Doval2006 A1.2)

![](<./attachments/Pasted image 20250926134737.png>)

3rd order: @Fujisaki1986 (FL model, also in @Drugman2019a)

![](<./attachments/Pasted image 20250926134856.png>)

* also allow negative flow segment after closure
* Motivation: “rounded closure” is often seen; sometimes attributed to residual leakage, **but they argue there is also a component due to a period of *negative flow* caused by *lowering of the vocal cords* after closure**

non-polynomials:

Rosenberg-C: trig (sine) model

LF-model: trig + exp model

# jumps in the DGF-permissible?

Right — if we’re talking **theoretical/physical glottal aerodynamics** rather than “nice math,” the story is:

- **Glottal flow u(t)u(t)u(t)** must be continuous (can’t teleport air). So u˙(t)\dot u(t)u˙(t) integrates to something continuous.
    
- **The derivative u˙(t)\dot u(t)u˙(t)** itself is *not* guaranteed smooth. At the instants of vocal fold contact (closure) and separation (opening), the *effective flow cross-section* changes abruptly: one cycle you have a slit of finite width, then suddenly it’s clamped shut (or vice versa).
    
- In fluid terms: that’s a topological change of the flow channel. The Reynolds number is high; turbulence, inertance, and fold collision all happen. There’s no guarantee of differentiability at those instants.
    

So:

- **A jump in DGF is not “unphysical.”** The signal is an idealization, and discontinuities at open/close boundaries are a reasonable stand-in for rapid but finite transients in the real system.
    
- Models like **Rosenberg C** and **Klatt poly** explicitly have slope discontinuities; they were designed that way to match spectral tilt.
    
- The **only hard constraint** is continuity of u(t)u(t)u(t) (glottal flow). The derivative can, in principle, jump.

*/

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

#bibliography("../../library.bib")
