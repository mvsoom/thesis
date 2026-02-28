#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/equate:0.3.2": equate, share-align
#show: equate.with()

= Priors for autoregressive filters
<chapter:arprior>

The vocal tract filter is modeled throughout this thesis as an AR($P$) process: a rational all-pole transfer function whose $P$ coefficients $bm(a) = (a_1, dots, a_P)^top$ encode the resonant structure of the vocal tract.
Estimating $bm(a)$ from speech data is done traditionally by LPC methods, or in our case, by IKLP.
Like any Bayesian model, IKLP requires a prior $p(bm(a))$, and throughout their paper #pcite(<Yoshii2013>) use a simple isotropic Gaussian $bm(a) ~ mono("Normal")(bm(0), lambda bm(I)_P)$ with $lambda = 0.1$.
This choice is openly acknowledged to be a regularizer rather than an informative prior: its job is to prevent the normal matrix from becoming singular, not to encode anything about the vocal tract.

Two things about this are unsatisfying.
First, an isotropic Gaussian with $lambda = 0.1$ means $sigma_a^2 = 10$, a standard deviation of about $3.16$ per coefficient — a very loose prior under which the AR filter will almost surely be unstable.
Second, we actually know a great deal about vocal tract filters: their resonances are spaced roughly $500$~Hz apart, their spectral rolloff is controlled by a handful of dominant poles, and the filter must be causal and passive.
None of this is encoded in $mono("Normal")(bm(0), 0.1 bm(I)_P)$.

This chapter develops two priors that attempt to do better.
The first, the _Monahan prior_, is the closest Gaussian to the uniform distribution over stable AR coefficient vectors; it is derived by moment matching and gives a principled explanation of the empirical $sigma_a^2 prop 1/P$ lore.
The second, the _soft spectral shaping prior_, is a new construction that encodes arbitrary spectral features — resonances, rolloff, formant locations — directly into the prior mean and covariance through a closed-form projection.
Both priors remain Gaussian and are therefore conjugate with IKLP's variational inference without modification.

== The isotropic prior

=== A problem: power blows up linearly in $P$

To understand why $lambda = 0.1$ is potentially dangerous, consider what an isotropic Gaussian prior does to the expected output power of the AR system.
For an AR($P$) filter driven by unit-power white noise, the output power is determined by the frequency response $H(e^(i omega)) = 1 \/ A(e^(i omega))$ where $A(z) = 1 - sum_(p=1)^P a_p z^(-p)$.
For small $|a_p|$, a Taylor expansion of $|H(e^(j omega))|^2$ around the zero-coefficient filter gives
$
  |H(e^(j omega))|^2 approx 1 + 2 sum_(p=1)^P a_p cos(p omega) + sum_(p=1)^P sum_(ell=1)^P a_p a_ell cos(p omega) cos(ell omega) + dots
$
Taking the expectation over the prior $bm(a) ~ mono("Normal")(bm(0), sigma_a^2 bm(I))$ and using $bb(E)[a_p] = 0$, the cross-terms vanish and the expected spectral power becomes
$
  bb(E)_bm(a) [|H(e^(j omega))|^2] approx 1 + sigma_a^2 sum_(p=1)^P cos^2(p omega).
$
Integrating over frequency and applying Parseval's identity gives the expected output variance
$
  bb(E)["var"(x)] approx 1 + P/2 sigma_a^2.
$ <eq:power-blowup>
With $P = 30$ and $sigma_a^2 = 10$ as in #pcite(<Yoshii2013>), this gives an expected variance of $1 + 150 = 151$: the prior is blowing up the expected power by two orders of magnitude.
Even a seemingly modest loosening to $sigma_a^2 = 0.05$ gives $bb(E)["var"(x)] = 1.75$, a 75% inflation.
The lesson from @eq:power-blowup is direct: to keep power approximately controlled under an isotropic prior, one needs $sigma_a^2 prop 1/P$.
This is the principled origin of the empirical "$lambda prop P$" or "$sigma_a^2 prop 1/P$" lore that one can see from time to time in the signal processing literature.

=== Stability

Beyond power, there is a more fundamental problem: a generic isotropic Gaussian places most of its mass outside the stability region $cal(A)_P subset bb(R)^P$, which is the set of coefficient vectors for which all roots of $A(z) = 1 - sum_p a_p z^(-p)$ lie strictly inside the unit disk.
For $P = 1$ the stability region is the interval $(-1, 1)$, and a unit Gaussian already places about 32% of its mass outside.
For larger $P$ the stability region is a complicated polytope and the fraction of unstable draws grows rapidly.
The posterior from IKLP is a regularized normal equation, so it can still produce stable estimates from stable data, but a prior that actively prefers unstable filters is working against the inference from the start and makes optimization fragile.

== The Monahan prior
<sec:monahan-prior>

=== The Monahan map

A classical result from time series analysis @Monahan1984 is that the partial autocorrelation function (PACF) parameters $bm(phi) = (phi_1, dots, phi_P)^top$ provide a smooth bijection between the hypercube $(-1, 1)^P$ and the stability region $cal(A)_P$:
$
  T : (-1, 1)^P -> cal(A)_P, quad bm(a) = T(bm(phi)).
$
Drawing $phi_p ~^("iid") mono("Uniform")[-1, 1]$ therefore produces only stable AR coefficient vectors.
The push-forward of this uniform PACF draw defines a distribution $q$ on $cal(A)_P$ that is the natural "uniform over stable filters" measure.

=== Moment matching as KL minimization

We want the best Gaussian approximation to $q$.
Minimizing $D_"KL" (q || p_(bm(mu), bm(Sigma)))$ over all $mono("Normal")(bm(mu), bm(Sigma))$ is a convex problem in the precision $bm(Sigma)^(-1)$, and the unique global minimizer is the moment-matched Gaussian:
$
  bm(mu)^* = bb(E)_q [bm(a)] = bm(0), quad bm(Sigma)^* = bb(E)_q [bm(a) bm(a)^top] =: bm(S)_q.
$ <eq:monahan-prior>
The mean is zero by the sign-symmetry of the uniform PACF law and the fact that the Monahan map is odd in each coordinate.
The covariance $bm(S)_q$ is the second moment of the push-forward distribution and captures the full geometry of the stability region within Gaussian capacity.

=== Computing $bm(S)_q$ and its structure

$bm(S)_q$ has no closed form but is easy to estimate by Monte Carlo: draw $N$ samples $bm(phi)^((n)) ~^("iid") mono("Uniform")[-1,1]^P$, compute $bm(a)^((n)) = T(bm(phi)^((n)))$, and average the outer products,
$
  hat(bm(S))_q = 1/N sum_(n=1)^N bm(a)^((n)) bm(a)^((n) top).
$
This converges at $cal(O)(N^(-1\/2))$ and can be precomputed for any desired $P$ and tabulated.#footnote[Variance reduction via quasi-Monte Carlo (Sobol or Halton sequences) or Latin hypercube sampling is straightforward and reduces the required $N$ by an order of magnitude in practice.]

Plotting the diagonal of $bm(S)_q$ reveals a clean decay: higher-lag coefficients $a_p$ are increasingly shrunk toward zero.
This is physically sensible — the stability region genuinely constrains high-order coefficients more tightly than low-order ones — and it is the feature that purely isotropic priors miss entirely.
The diagonal of $bm(S)_q$ also decays roughly as $1/p$, recovering the $1/P$ variance scaling as an average over lags rather than a single global shrinkage.

=== A note on the Padé tension
<sec:pade-tension>

The Monahan prior rests on the assumption that $A(z)$ is a genuine causal stable filter.
In the source-filter model of speech this is physically motivated: the vocal tract is a passive acoustic resonator and its transfer function must have poles inside the unit disk.
From this viewpoint, the Monahan prior is exactly right.

There is a worthy competing viewpoint, however.
The AR polynomial $A(z)$ of order $P$ can also be read as a classical Padé approximation to whatever the true underlying transfer function happens to be — a rational expansion that is only approximately physical and whose poles are not expected a priori to coincide with the true formants one-to-one @Stevens2000.
Under this reading, enforcing strict stability through the prior imposes a constraint that is an artifact of the approximation order rather than a physical necessity, and one might prefer to keep the coefficient space as free as possible.

We take no strong position here, since both views have merit and the right choice likely depends on the pitch and phonation type of the frame being analyzed.
What we can say is that for the soft spectral shaping prior developed in the next section, the two views are not in conflict: the spectral shaping construction is purely about the prior mean and leaves the covariance free to be either $bm(S)_q$ (from Monahan) or $sigma_a^2 bm(I)_P$ (isotropic), depending on which philosophical stance one takes.

== Soft spectral shaping priors
<sec:soft-spectral-shaping>

Suppose we want the prior to encode the expectation that the AR filter has a resonance near $500$~Hz, or that its spectrum rolls off at a particular rate, or that it has a formant structure typical of a particular vowel.
None of these beliefs fit naturally into a Gaussian prior $mono("Normal")(bm(0), bm(Sigma))$ with any standard choice of $bm(Sigma)$.
The soft spectral shaping prior is a construction that embeds exactly these beliefs — encoded as desired root locations of $A(z)$ — into the prior mean through a closed-form linear projection.

=== Spectral features as divisibility conditions

The spectral envelope of the AR filter is $S(omega) prop 1 \/ |A(e^(-i omega))|^2$.
The poles of the envelope are the zeros of $A(z)$.
Requiring $A(z)$ to have a root at $z = rho e^(i omega_0)$ (a resonance near frequency $omega_0$ with damping $1 - rho$) is therefore equivalent to requiring that the quadratic factor
$
  Q(z) = 1 - 2 rho cos(omega_0) z + rho^2 z^2
$
divides $A(z)$.
More generally, any desired spectral feature — a resonance, a rolloff pole, a seasonal unit root — can be expressed as a monic polynomial $Q(z)$ whose roots encode the feature, with the requirement that $Q(z) divides A(z)$.

/*

/* TODO: table of spectral features */

Several spectral features and their corresponding $Q$ polynomials are listed in @table:spectral-features.
Multiple features $Q_1, dots, Q_K$ combine by taking their least common multiple $M(z) = "lcm"{Q_1(z), dots, Q_K(z)}$, since $Q_k divides A$ for all $k$ if and only if $M divides A$.
Over $bb(C)$, this just means collecting the union of desired roots with their maxibm(mu)m multiplicities, so the effective degree of the combined constraint is $L_"eff" = deg M$.

*/

=== Divisibility in the frequency domain

Before developing the prior, it is worth pausing on what divisibility means spectrally.
If $A(z) = M(z) R(z)$ exactly, then on the unit circle
$
  A(e^(i omega)) = M(e^(i omega)) R(e^(i omega)),
  quad
  |A(e^(i omega))|^2 = |M(e^(i omega))|^2 |R(e^(i omega))|^2,
$
so the spectral envelope factors as
$
  S(omega) prop 1/(|M(e^(i omega))|^2 |R(e^(i omega))|^2).
$
The spectral shape of $M$ is therefore a *guaranteed factor* of the envelope: wherever $M$ has a pole near the unit circle, $S(omega)$ must have a peak, and wherever $M$ has a zero, $S(omega)$ must have a trough.
The remaining polynomial $R(z)$ is free to fill in whatever the data requires.
In the soft version developed below, divisibility becomes an *expected* rather than guaranteed condition, and the prior is nudged toward envelopes whose spectral shape contains the features encoded in $M$, while leaving the data free to adjust $R(z)$ in any direction.
This is why literally any spectral feature that can be expressed as a root structure can be embedded by this mechanism: formants, antiformants, rolloff poles, seasonal components — all reduce to a choice of $M(z)$.

=== Divisibility as linear constraints on $bm(a)$

Whether $M(z) divides A(z)$ is a purely algebraic condition, and it is linear in the coefficients of $A$.
To see this, note that $M divides A$ if and only if the remainder of the polynomial division $A mod M$ is zero.
This remainder can be computed by the Sylvester or convolution matrix of $M$: there exists a matrix $bm(F) in bb(R)^(L_"eff" times P)$ and a vector $bm(c) in bb(R)^(L_"eff")$ (determined by the monic leading term $a_0 = 1$) such that
$
  bm(f)(bm(a); M) := bm(F) bm(a) + bm(c) = bm(0) quad arrow.l.r.double quad M divides A.
$ <eq:divisibility-constraint>
Exact divisibility is $bm(f)(bm(a); M) = bm(0)$; soft divisibility in the prior sense means asking that $bb(E)_p [bm(f)(bm(a); M)] = bm(0)$, i.e., $bm(F) bm(mu) + bm(c) = bm(0)$.

=== Projection onto the soft constraint set

Given a base Gaussian prior $bm(a) ~ mono("Normal")(bm(mu), bm(Sigma))$ — which could be the Monahan prior $mono("Normal")(bm(0), bm(S)_q)$ or the isotropic prior $mono("Normal")(bm(0), sigma_a^2 bm(I))$ — we seek the closest Gaussian $mono("Normal")(bm(m)^*, bm(S)^*)$ that satisfies the soft constraint $bm(F) bm(m)^* + bm(c) = bm(0)$.
Minimizing $D_"KL" (mono("Normal")(bm(m), bm(S)) || mono("Normal")(bm(mu), bm(Sigma)))$ subject to $bm(F) bm(m) + bm(c) = bm(0)$ is an I-projection onto a linear affine constraint on the mean.
Since the constraint touches only the mean and not the covariance, the solution has $bm(S)^* = bm(Sigma)$ unchanged, while the mean is shifted by the $bm(Sigma)^(-1)$-orthogonal projection:
$
  bm(m)^* = bm(mu) - bm(Sigma) bm(F)^top (bm(F) bm(Sigma) bm(F)^top)^(-1) (bm(F) bm(mu) + bm(c)).
$ <eq:soft-spectral-shaping>
If the base prior already satisfies the constraint ($bm(F) bm(mu) + bm(c) = bm(0)$) then $bm(m)^* = bm(mu)$ and the prior is unchanged.
Otherwise, the mean is shifted by the minibm(mu)m amount (in the $bm(Sigma)^(-1)$ metric) necessary to bring the expected $bm(f)(bm(a); M)$ to zero.
The cost of this shift is quantifiable as the KL penalty
$
  1/2 log(1 + bm(m)^(*top) bm(S)_q^(-1) bm(m)^*),
$
which grows with how incompatible the base prior is with the desired spectral features.

Equation @eq:soft-spectral-shaping is the soft spectral shaping prior.
It is closed form, requires no sampling or numerical optimization beyond a matrix solve of size $L_"eff"$, and is directly usable as the Gaussian prior in IKLP without any modification to the inference algorithm.

If one prefers to enforce divisibility exactly rather than in expectation — for instance if a particular formant structure is known with certainty — the conditioning forbm(mu)la gives the hard version: same mean shift, but the covariance is additionally deflated along the constrained directions,
$
  bm(S)_"hard" = bm(Sigma) - bm(Sigma) bm(F)^top (bm(F) bm(Sigma) bm(F)^top)^(-1) bm(F) bm(Sigma).
$
This is the Gaussian posterior after observing $bm(f)(bm(a); M) = bm(0)$ exactly, and it removes all prior uncertainty along those directions.

=== What spectral features can be encoded

The construction places no restriction on the choice of $M(z)$ beyond its degree satisfying $L_"eff" <= P$.
Any spectral feature expressible as a root structure of $A(z)$ can therefore be embedded.
A few examples that are directly relevant to vocal tract modeling:

A _damped resonance_ near frequency $omega_0$ with bandwidth controlled by $1 - rho$ corresponds to the quadratic $Q(z) = 1 - 2 rho cos(omega_0) z + rho^2 z^2$, with $rho < 1$ for stability.
Repeating this factor raises the local spectral slope around $omega_0$ by $12$~dB/octave per repetition.
For a full vowel formant structure, one would take the product of two or three such quadratics at the expected first, second, and third formant frequencies.

A _spectral rolloff_ at rate $6m$~dB/octave is encoded by the $m$-fold repeated real pole $(1 - rho z)^m$ for some $rho$ close to but inside the unit disk.
This is directly related to the "pre-emphasis" step common in speech processing, but now encoded as a prior belief rather than a hard preprocessing step.

A _DC notch_ or _Nyquist notch_, corresponding to $Q(z) = 1 - z$ or $Q(z) = 1 + z$, removes prior probability mass from filters with runaway low or high-frequency response.

Multiple features combine multiplicatively: a prior that sibm(mu)ltaneously expects a first formant near $700$~Hz, a second near $1200$~Hz, and a $6$~dB/octave overall rolloff uses $M(z) = Q_"F1"(z) dot Q_"F2"(z) dot Q_"rolloff"(z)$, with $L_"eff" = deg M = 5$.

== Summary

The standard isotropic prior used in IKLP is a regularizer, not a model.
It is too loose by a factor of $P/2$ in its effect on output power, and it is indifferent to the stability constraint that any physical vocal tract filter must satisfy.

The Monahan prior corrects the stability problem by moment-matching to the push-forward of a uniform PACF draw, yielding the optimal Gaussian approximation to the uniform distribution over stable AR filters.
Its covariance $bm(S)_q$ captures the geometry of the stability region and recovers the $sigma_a^2 prop 1/P$ scaling law as a consequence rather than a heuristic.
Whether to use it in practice depends on how one thinks about the AR model: as a physical transfer function (where stability is a hard constraint and Monahan is the right prior) or as a Padé approximation (where the coefficient space should remain free, and Monahan's constraint may be too strong).

The soft spectral shaping prior addresses a different and arguably more practically useful question: not stability, but rather spectral content.
By encoding desired spectral features — formants, rolloff, notches — as divisibility conditions on $A(z)$, and then I-projecting the base Gaussian prior onto the resulting linear affine constraint on the mean, one obtains a closed-form Gaussian prior that actively expects the vocal tract filter to look like voiced speech.
The Fourier interpretation is clean: divisibility in the coefficient domain corresponds to guaranteed spectral factors in the envelope, so the soft version tilts the prior toward envelopes that carry the desired spectral shape while leaving all remaining degrees of freedom to the data.
Both priors are Gaussian and slot directly into IKLP without modifying the inference algorithm.