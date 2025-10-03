= From parametric to nonparametric glottal flow models

In this chapter, we motivate the use of $arccos(n)$ kernel @Cho2009  to describe the glottal flow during the open phase.

== Glottal flow model zoo

// **notebook**: /home/marnix/WRK/proj/sflinear/notebook/single-packet/titze-alku.ipynb

Many models for the glottal flow and its derivative have been proposed over the course of time in acoustic phonetics -- see @fig:gf-lineup for a catalogue already in 1986. These models differ mainly in the details, and can be unified in a common framework as per #cite(<Doval2006>, form: "prose").

#figure(
  image("./fig/fl-model.png", width: 100%),
  placement: bottom,
  caption: [
    A 1986 lineup of GF models, together with their derivatives (DGFs). From #cite(<Fujisaki1986>, form: "author").
  ],
) <fig:gf-lineup>

For a GF model to be useful, it should have the following properties: @Doval2006
- list of props here: differentiability etc // TODO
- DGF should integrate to $approx 0$
- Computationally: quick to evaluate

*Allow negative flow?* At first sight, glottal flow should be strictly postive, and most GF models enforce it. But there are exceptions like @Fujisaki1986. Motivation: "rounded closure" is often seen; sometimes attributed to residual leakage, but they argue there is also a component due to a period of negative flow caused by lowering of the vocal cords after closure, drawing DC current of air back in.

*The LF model.* The most dominant GF model by far is the LF model, number (e) in @fig:gf-lineup and see @appendix:lf for detailed description. Some shortcomings:
- Analytically awkward: null flow condition not tractable, requires numerically solving a bisection routine. There has been research into making that routine more numerically stable.
- Overparametrization: though conveniently parametrized in terms of physiological features, its parameters are not independent of each other. They are usually regressed in terms of one another, or in the LF model has been used for parametric fitting, even into a single parameter.
- Does not allow negative flow.
It does however allow for very sharp GCI events, which are of utmost importance in joint inverse filtering setting. We will use it as a base model due to its popularity.

== Classic polynomial models

We make a case for the old "forgotten" family of polynomial GF models such as @Alku2002 @Verdolini1995 @Doval2006:
- Computationally fast, analytical null flow condition
- Many exist in literature guised in orders $n = 0,1,2,3$
- Capable of very sharp events
- "Bright spectrum": very slow decay, so are excellently placed to excite GF (@chapter:gif).


/*
See images ./fig/:

0th order: @Alku2002
1th order: @Titze2000 @Verdolini1995 (pdf paywalled)
2nd order: KLGLOTT88 (from @Doval2006 A1.1)
3rd order: R++ (from @Doval2006 A1.2)
3rd order: @Fujisaki1986 (FL model, also in @Drugman2019a)
* also allow negative flow segment after closure
* Motivation: “rounded closure” is often seen; sometimes attributed to residual leakage, **but they argue there is also a component due to a period of *negative flow* caused by *lowering of the vocal cords* after closure**

non-polynomials:
Rosenberg-C: trig (sine) model
LF-model: trig + exp model
*/

The modern-day revival of piecewise functions (linear, quadratic, ...) puts these ancient models in a new light. Changepoint modeling ("hard ifs") in the guise of decision surfaces is what drives deep architectures today, and it is exactly the same kind we need for GFs. Plus, these models are already embedded in zero DC line (ie, a polynomial of order 0) as they model only open phase.

There are conventially several changepoints in the glottal cycle to be modeled: the primary changepoints are opening onset and closure instant, with optional landmarks like max flow (maximum of $u(t)$) and closing phase onset (minimum of $u'(t)$) used to quantify shape. The simplest and arguably most succesful polynomial model is the triangular pulse model proposed in #cite(<Alku2002>, form: "prose") which is asserts $u(t)$ $n = 1$ piecewise linear in GF and $u'(t)$ $n = 0$ piecewise constant. It is used mainly as a more robust way to estimate OQ (a time domain parameter) from the amplitude domain and not as a GF model in itself, but we can use it as a starting point for our generalization from parametric to nonparametric models.

// TODO: show a figure of the glottal cycle as fotographed, to show the changepoints, possibly next to Alku
#figure(
  image("./fig/alku2002.png", width: 60%),
  placement: auto,
  caption: [
    The triangular pulse model proposed in #cite(<Alku2002>, form: "prose").
  ],
) <fig:alku>

=== The triangular pulse model

@fig:alku shows the triangular pulse model. Its derivative is a piecewise constant function:
$
  u'(t) = cases(
    0 quad quad & t in (-oo, t_o],
    f_"ac" / T_1 quad quad & t in (t_0, t_m],
    -f_"ac"/T_2 quad quad & t in (t_m, t_c],
    0 quad quad & t in (t_c, +oo),
  )
$ <eq:dgf-piece>
This function is parametrized by the time domain constants ${T, T_1, T_2}$ (or equivalently: ${t_o = T - T_1, t_m = T - T_2, t_e = T}$) and the amplitudes ${f_"ac", d_"peak"}$. Note that the latter does not appear in @eq:dgf-piece because the null integral constraint $integral_0^T u'(t) dif t = 0$ removes one degree of freedom, so any single one of these can be expressed in terms of the others. Thus $d_"peak" = f_"ac"/T_2$ or $T_2 = f_"ac"/d_"peak"$. #cite(<Alku2002>, form: "prose") point out that this last relation expresses a difficult-to-measure time domain quantity as the ratio of two easy-to-measure quantities in the amplitude domain and exploit this fact to measure the open quotient (OQ) more robustly.



This model contains two jumps in the derivative domain so we can write it conveniently as a linear combination of two Heaviside functions during the open phase:
$
  u'(t) = a_1 H(t - t_o) + a_2 H(t - t_m) quad (t_o <= t <= t_c)
$

where $a_1 = f_"ac"/T_1$, $a_2 = -f_"ac" (1/T_1+1/T_2)$ and $H(t)$ is the Heaviside function:
$
  H(t) = integral_(-oo)^t delta(tau) dif tau = cases(
    0 quad & t < 0,
    1 quad & t > 0.
  )
$

*Generalizing the model.* For increased resolution, we can generalize this to a linear combination of $K$ arbitrarily scaled Heaviside jumps centered at change points $t_(1:K) in [t_o, t_e]$:
$
  u'(t) = sum_(k=1)^K a_k H(t - t_k) quad (t_o <= t, t_k <= t_c)
$
If we assume the $t_(1:K)$ given for now and let $a_k ~ N(0, sigma^2_a)$, then using $integral_(-oo)^t H(tau - c) dif tau = H(t - c)(t - c)$ we get the null integral condition for free in expectation:
$
  bb(E)_a [integral_0^T u'(t) dif t] = sum_(k=1)^K bb(E)[a_k] H(T-t_k) (T-t_k) = 0.
$
This motivates setting the mean of the $a$ to zero, otherwise than symmetry of ignorance around $a=$.
We can also enforce it: set $b_k = H(T-t_k) (T-t_k)$ then
$
  sum_(k=1)^K a_k b_k = 0 ==> "Gaussian of rank" K - 1
$
This is a linear constraint on the $a$, so we can update the prior to always respect the constraint.
$
  Sigma, mu
$
This shows how prior of $a$ can encode properties we care about. This is a central theme in what follows: we will find that our priors for the linear amplitude can encode a host of features such as differentiability, null integral, ... in other words, the gross features of the expected spectrum of the DGF.

/* picture of triangular pulse model with K=2 ... K = 5 with t_k chosen uniformly */

== Parametric polynomial models

Generalize $n = 0 -> n <= 3$ from previous example

The integrated flow conditions also suggest to write $H(t)$ as a RePU (rectified power unit) function:
$
  (t)_+^n = cases(
    0 quad & t < 0,
    t^n quad & t >= 0,
  )
$
so now integration and differentiation act as convenient ladder operators:
$
  integral_(-oo)^t (tau - c)_+^(n-1) dif tau & = 1/n (t - c)_+^n, \
                     dif/(dif t) (t - c)_+^n & = n (t - c)_+^(n-1) quad quad (n >= 1, c in bb(R))
$
allowing us to quickly go from DGF to GF and vice versa.
#footnote[These relations can be made precise by using distribution theory @Lighthill1958 rather than functions, but they will do for our purpose here.]

and thus includes Heaviside for $n = 0$ ($t_+^0 = H(t)$); ReLU for $n = 1$ (linear), ReQU for $n = 2$ (quadratic) and ReCU for $n = 3$ (cubic).

#figure(
  grid(
    columns: 2,
    column-gutter: { 4em },
    row-gutter: { 1em },
    align: center + bottom,
    [#include "./fig/nn-polynomial-a.typ"], [#include "./fig/nn-polynomial-b.typ"],
    [(a)], [(b)],
  ),
  placement: top,
  kind: image,
  caption: [
    (a) The triangular pulse model as a tiny neural network with a single hidden layer, linear readout and $t^0_+ = H(t)$ activation.
    (b) The general parametric polynomial model of degree $n$ with order $K$ with weights $w in bb(R)^(K times 2), a in bb(R)^K$ and RePU activation function $t_+^n$.
  ],
  gap: 1em,
) <fig:nn-polynomial>


*Connection to neural networks.* In a modern lens, these models can be seen as a neural net with a single hidden layer in the regression setting. $H(t) t$ is a ReLU, etc. This is good, because already a single hidden layer h+as the universality property. Encouraged, this also suggests the prior $t_k ~ N(0, sigma_t^2)$ truncated to $t_k in [t_0, t_e]$: these are biases in the neural net picture, and we truncate to remain in the open phase.

/* now we got a prior for t_k: we can show samples */
/* K, n picture */



== Nonparametric polynomial models

After having generalized $n$, now we generalize $K -> oo$.

$
  k_n(t, t') = bb(E)_(w ~ cal(N)(0, I))[phi(w^top x) phi(w^top y)]
$





///

We can integrate out both $a$ and $t$ with our two priors and arrive at arccos kernel.

Here the null integral constraint becomes analytically "intractable" at first sight, but can be done for SqExp model analytically, and Matern models via Matern expansion trick @Tronarp2018

Arccos is homogenous for global rescaling, which is equivalent to rescaling $Sigma -> alpha Sigma$.

But we can allow $Sigma = mat(sigma_b^2, 0; 0, sigma_t^2)$ as this is important to model behavior of kernel. Introducing a third parameter $rho$ (correlation in $Sigma$) breaks our FT derivation (the $tan "/" arctan$ trick), which assumes no correlation between bias and $t$. So individual rescaling is as far as we can go and probably more than enough. So we can proceed with the $N(0,I)$ case as in @Cho2009. // https://chatgpt.com/s/t_68dfa3181bf88191a3183a8138bf2969


// @Pandey2014: covariance arccos kernel: to go wide or deep in NNs?
// https://upcommons.upc.edu/server/api/core/bitstreams/bf52946e-0904-4e3d-afb7-d85d8a33c46a/content page 13


/*
Kernel support for various GF models without null integral constraint

Calculate p(D|k) for k in (arccos, matern, spline) <= nonparametric
Perhaps also a Bayesian network <= parametric

And D is a single open phase of poly (orders <= 3) and nonpoly (order = infty) models

See which kernel or model has most support

Also show # params and compute time for each

*/

*Neural networks again.* The marginalization above was first done by #cite(<Cho2009>, form: "prose") in the context of infinite-width neural networks in the style of #cite(<Neal1996>, form: "prose") #cite(<Williams1998>, form: "prose"). This viewpoint also allows for going beyond a depth 1 network by iteration of the $arccos$ kernel.#footnote[Kernel composition (reiteration) is indeed a valid kernel operation in general, as recently emphasized by @Dutordoir2020.]

*Why not go infinite depth?* Different character from increasing width; effective depth of deep GPs. If stable limit, it becomes independent of inputs @Diaconis1999. Seen often in DGPs as input independency, "forgetting inputs". Though one might counteract that going to infinite width also has similar "unfortunate" consequences (MacKay's baby with the bath water): "features are not learned", basis functions are fixed. This shows that kernel hyperparameters must encode (most of) features. We do this via sparse multiple kernel learning; ie static kernels on the Yoshii-grid mechanism; our hyperparams are $(T, tau, "OQ")$, ie 3 dim grid.

*Why not spline models?* Piecewise spline models are well-known and effective in low-dimensional nonparametric regression. Why not use them? Because they depend on the resolution. As Gaussian process priors, spline kernels produce posterior means that are splines with knots (hingepoints in some derivative) fixed at the observed inputs and nowhere else @MacKay1998[p. 6]. In contrast, the $arccos(n)$ kernel will learn the effective number of hingepoints from the data, which may happily remain $O(1)$ while the amount of datapoints grows indefinitely. Translated to our problem, we want the number of effective change points to be resolution-independent (independent of the sampling frequency) and not confined to the observation locations.

