= More comments on the nonparametric glottal flow model
<chapter:nonparametric-gfm>
/*
#figure(
  image("/figures/svg/20251008144755528.svg"),
  caption: [Hard changepoints are difficult, best we can do are steep slopes.],
) <fig:steep>
*/

Some aspects of the nonparametric glottal flow model (GFM) derived in @chapter:gfm are discussed in more detail in what follows.

/*
==== What happened?
It is worth pausing to ask what really happened here.
At first sight, taking $H -> oo$ might seem like an act of reckless generalization: we blow up the number of parameters without bound, yet somehow end up with something *simpler*—a single Gaussian process with a fixed kernel. Conditional on the random features, the model was already Gaussian, so the only effect of the limit is that the *random design itself* stops being random. The empirical kernel freezes to its mean, and with it the whole architecture of the network becomes a static, deterministic map from inputs to covariances.

This is the peculiar balance of the Gaussian process limit. The randomness of finite networks (the accident of which features you drew) disappears, while the *expressive field* of possible functions becomes infinite. What looks like a loss of freedom in one space is a gain of freedom in another. You trade a random, high-dimensional parameterization for a deterministic law over an infinite-dimensional function space. The model collapses in its parameter dimension but expands in its functional reach. That is the sense in which the GP limit achieves “infinite resolution”: it no longer needs to enumerate features to approximate every smooth behavior the kernel supports. The prior already spans that continuum.
*/

=== Infinite precision?
In this view, increasing $H$ increases the _Monte Carlo resolution_ with which the regression model samples its feature space.
The nonparametric limit replaces explicit random changepoints with their continuous Gaussian measure, yielding a Gaussian process prior over $u'(t)$ whose covariance is the arc-cosine kernel restricted to the open phase of the glottal cycle.

There is a caveat however; we got "infinite" precision (meaning full rank), but smoother functions.
This is the analogue of the amplitude prior of footnote 17 which constrains the ellipsoid.


/* main point here: YES, we compromised; we got fast inference, but also diminshed support for true O(1) amount of jumps like a Levy process would. This is a practical question -- need to find out by running nested sampling and see how much support for these jumps is really there -- just calculate! */

/* This means we should model u(t), not du(t)!

u(t) ~ arccos: arccos is always continuous -- we cant do real jumps

but if we can get du(t) implicitly -- from spectral domain OR via AR pole -- we can maybe cheat this thing

Maybe we can LEARN the radiation characteristic by priming our AR prior to look for that spectral decay of +6db/oct

So we model u(t) * (h(t) * r(t)) and the latter factor of combined VT + radiation impulse response is learned directly

another option is to integrate data (move diff to data) but this possibly nasty

pre-emphasis just flattens spectral slope as to partition poles better -- has nothing to do with radiation factor.
Radition factor can only be undone by integrating, not further diffing.
Pre-emphasis tilts the spectrum "up" by emphasizing higher freqs and is a fitting trick

On the other hand
LPC just proclaims s(t) = e(t) * h(t) and shoves radiation into the source such that h(t) cna remain all-pole
since differentiation is ~ i2pi x => zero, not a pole
so we are generalizing LPC, so also choose e(t) => u(t) * r(t)
*/

=== Why not go infinite depth?
Different character from increasing width; effective depth of deep GPs. If stable limit, it becomes independent of inputs @Diaconis1999. Seen often in DGPs as input independency, "forgetting inputs". Though one might counteract that going to infinite width also has similar "unfortunate" consequences (MacKay's baby with the bath water): "features are not learned", basis functions are fixed. This shows that kernel hyperparameters must encode (most of) features. We do this via sparse multiple kernel learning; ie static kernels on the Yoshii-grid mechanism; our hyperparams are $(T, tau, "OQ")$, ie 3 dim grid.

=== Why not spline models?
Piecewise spline models are well-known and effective in low-dimensional nonparametric regression. Why not use them? Because they depend on the resolution. As Gaussian process priors, spline kernels produce posterior means that are splines with knots (hingepoints in some derivative) fixed at the observed inputs and nowhere else @MacKay1998[p. 6]. In contrast, the $arccos(n)$ kernel will learn the effective number of hingepoints from the data, which may happily remain $O(1)$ while the amount of datapoints grows indefinitely. Translated to our problem, we want the number of effective change points to be resolution-independent (independent of the sampling frequency) and not confined to the observation locations.

=== Why not Lévy processes?
These encode Poisson-style jumps $O(1)$ in number in time. But inference in these is always $O("# of jumps")$. So can't really marginalize out these jump points, and we want to avoid MCMC. We want to stack everything in the amplitude marginalization. But, actual discontinuities require Lévy processes; the arc cosine GP alone can only fake it with steep ramps /*(see @fig:steep)*/.