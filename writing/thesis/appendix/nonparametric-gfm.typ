= More comments on the nonparametric glottal flow model
<chapter:nonparametric-gfm>

/*
#figure(
  image("/figures/svg/20251008144755528.svg"),
  caption: [Hard changepoints are difficult, best we can do are steep slopes.],
) <fig:steep>
*/

Some aspects of the nonparametric GFM derived in @chapter:gfm are discussed in more detail here.

=== Infinite precision?
In the GP view, increasing $H$ increases the _Monte Carlo resolution_ with which the finite neural network model samples the arc cosine kernel — the deviation from the limiting kernel vanishes as $cal(O)(1\/sqrt(H))$ by @eq:dev86.
The nonparametric limit replaces explicit random changepoints with their continuous Gaussian measure, and the prior over $u'(t)$ becomes a GP whose covariance is the arc cosine kernel restricted to the open phase.

There is a caveat though.
We gained full rank — the prior now has support over an infinite-dimensional function space — but we lost the ability to represent true discontinuities.
Arc cosine GP sample paths are continuous; they cannot jump.#footnote[
  Specifically, sample paths of a GP with a kernel of degree $d$ are $d$-times continuously differentiable with probability one @Rasmussen2006.
  For $d = 0$ this gives continuous but nowhere-differentiable paths; for $d >= 1$, paths are smoother still.
  In either case there are no jumps.
]
This is the analogue of the amplitude prior constraining the ellipsoid in the finite-$H$ case (footnote~17 of @chapter:gfm): there we traded a large discrete family of possible changepoint configurations for a continuous Gaussian prior that concentrates mass inside a smooth ellipsoid; here we trade a distribution over Poisson-style jump processes for a Gaussian prior that concentrates mass on smooth function classes.
The best the arc cosine GP can do is approximate a sharp GCI with a steep ramp. /*(see @fig:steep)*/

Whether this is a problem in practice is an empirical question.
The closed-phase GCI is physiologically sharp but not infinitely so, and the DGF data from VocalTractLab are sampled at finite rate, so the distinction between "true jump" and "ramp steep enough that it spans one sample" may not be detectable.

/* main point here: YES, we compromised; we got fast inference, but also diminished support for true O(1) amount of jumps like a Levy process would. This is a practical question -- need to find out by running nested sampling and see how much support for these jumps is really there -- just calculate! */

=== Why this matters for the choice of modeled signal
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
The continuity constraint has a direct implication for which signal we should model with the arc cosine GP.
We model $u'(t)$ — the DGF — because the GCI appears as a sharp negative peak there, which is easier to detect and align than the integral $u(t)$.
But if the arc cosine GP is continuous, then $u'(t)$ is smooth, which means $u(t)$ is differentiable — and real glottal flow does not have a differentiable closure.

One way around this is to model $u(t)$ directly with the arc cosine GP instead of $u'(t)$.
Then the GP sample paths are continuous (which $u(t)$ should be) and discontinuities can be pushed into the _derivative_ implicitly via the radiation factor: the combined vocal tract and radiation impulse response is modeled as the AR filter, and the combined source signal becomes $u(t) * r(t)$ rather than $u'(t)$ alone.
This is, at bottom, what classical LPC does — it posits $s(t) = e(t) * h(t)$ and allows $h(t)$ to absorb the radiation effect $r(t)$, since differentiation corresponds to a zero in the transfer function, not a pole, so $h(t)$ can remain all-pole.
From this viewpoint the present approach generalizes LPC in exactly that way.

The alternative of obtaining $u'(t)$ implicitly — from the spectral domain or from the AR pole structure — is also possible and may allow the sharp-closure structure to be recovered without committing the GP to model it directly.

=== Why not go infinite depth?
Increasing depth has a different character from increasing width.
For a deep GP with Gaussian kernels at each layer, letting the depth diverge drives the process toward one that is independent of its inputs @Diaconis1999 — a known pathology in practice, where deep GPs tend to "forget" their inputs and produce nearly constant sample paths.
Finite but large depth mitigates this, but one might note that infinite width carries a vaguely analogous cost: in the GP limit, the basis functions are frozen at their prior expectation and no longer adapt to the data as drawn features.
What is learned is entirely encoded in the kernel hyperparameters.
This makes kernel learning — and in particular the multiple-kernel structure of IKLP — the mechanism through which the model does acquire data-driven spectral features, rather than learning them through random feature adaptation.

=== Why not spline models?
Piecewise spline models are well-studied and effective in low-dimensional nonparametric regression.
The issue here is resolution dependence.
As GP priors, spline kernels produce posterior means that are splines with knots fixed at the observed input locations @MacKay1998[p.~6]: add more data, add more knots.
The arc cosine kernel, by contrast, learns the effective number of hingepoints from the data through its hyperparameters, and this number can remain $cal(O)(1)$ even as the number of observations grows.
For glottal flow this matters: a DGF waveform sampled at $16$~kHz and one sampled at $44$~kHz should both require $cal(O)(1)$ effective changepoints to describe the open phase, not a number proportional to the sampling rate.

=== Why not Lévy processes?
These encode Poisson-style jumps, $cal(O)(1)$ in number, which is exactly the right prior for a signal with a small number of hard discontinuities.
The problem is inference: the posterior over a Lévy process is indexed by the jump locations, and marginalizing those out requires MCMC or particle methods that scale with the number of jumps.
The GP approach achieves fast closed-form posterior inference precisely because everything — changepoints, amplitudes, timing — has been absorbed into the kernel.
Trading exact jumps for steep smooth ramps is the price of that tractability.