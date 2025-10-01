= Maximum-entropy (I-projection) updates for AR coefficient priors with soft divisibility features
<maximum-entropy-i-projection-updates-for-ar-coefficient-priors-with-soft-divisibility-features>
== 1) AR context and notation
<ar-context-and-notation>
We model an AR(P) via
$ A (z) = 1 + a_1 z + dots.h.c + a_P z^P , #h(2em) a = (a_1 , dots.h , a_P)^tack.b . $
The (unnormalized) spectrum is
$ S (omega) prop frac(1, \| A (e^(- i omega)) \|^2) . $ Thus zeros of
$A (z)$ near the unit circle become spectral poles; multiplicity
controls local slope/peakedness. Stability/causality requirements
translate to constraints on the root locations of $A$ (sign convention
dependent).

== 2) Collapse multiple features to one effective divisor
<collapse-multiple-features-to-one-effective-divisor>
Given monic feature polynomials
$Q_k (z) = 1 + q_(k , 1) z + dots.h.c + q_(k , L_k) z^(L_k)$ (possibly
of different degrees), define
$ M (z) = upright(l c m) { Q_1 (z) , dots.h , Q_K (z) } , #h(2em) L_(upright(e f f)) = deg M . $
Then $Q_k divides A med forall k$ iff $M divides A$. All independent
constraints come from $M$.

#strong[Interpretation.] Over $bb(C)$, if
$Q_k (z) = product_r (1 - rho_r z)^(m_r^((k)))$, then
$ M (z) = product_r (1 - rho_r z)^(m_r) , quad m_r = max_k m_r^((k)) , quad L_(upright(e f f)) = sum_r m_r . $
So $M$ collects the union of desired roots with their maximum
multiplicities across features.

== 3) Linear remainder constraints
<linear-remainder-constraints>
Let $[1 ; a] in bb(R)^(P + 1)$ be the full coefficient vector. Build any
left–null basis $N (M) in bb(R)^(L_(upright(e f f)) times (P + 1))$ for
the convolution/Sylvester matrix of $M$ so that
$ N (M) thin [1 ; a] = 0 med arrow.l.r.double med M divides A . $ Write
$N (M) = [thin c med med F thin]$ with $c in bb(R)^(L_(upright(e f f)))$
(acts on the fixed $a_0 = 1$) and
$F in bb(R)^(L_(upright(e f f)) times P)$. The $L_(upright(e f f))$
linear statistics are
$ f (a ; M) = F a + c in bb(R)^(L_(upright(e f f))) . $ Exact
divisibility is $f (a ; M) = 0$.

== 4) Gaussian base prior
<gaussian-base-prior>
$ a tilde.op cal(N) (mu , Sigma) , #h(2em) Sigma succ 0 . $

== 5) Soft divisibility via I-projection (moment constraints)
<soft-divisibility-via-i-projection-moment-constraints>
Impose only expectations $ bb(E) [f (a ; M)] = F mu + c = 0 . $ Find
$cal(N) (m , S)$ minimizing
$D_(upright(K L)) (cal(N) (m , S) thin parallel thin cal(N) (mu , Sigma))$
subject to $F m + c = 0$. Because constraints touch the mean only,
covariance is unchanged and the mean is the $Sigma^(- 1)$-orthogonal
projection onto the affine set:
$ S^star.op = Sigma , #h(2em) m^star.op = mu - Sigma F^tack.b (F Sigma F^tack.b)^(- 1) thin (F mu + c) . $
If rows of $F$ are redundant, replace $(F Sigma F^tack.b)^(- 1)$ by the
Moore–Penrose pseudoinverse.

== 6) Hard divisibility (conditioning)
<hard-divisibility-conditioning>
If you want $f (a ; M) = 0$ almost surely,
$ m_(upright(c o n d)) & = mu - Sigma F^tack.b (F Sigma F^tack.b)^(- 1) (F mu + c) ,\
S_(upright(c o n d)) & = Sigma - Sigma F^tack.b (F Sigma F^tack.b)^(- 1) F Sigma . $
Same mean shift; variance along constrained directions is removed.

== 7) What kinds of AR features can $Q$ encode?
<what-kinds-of-ar-features-can-q-encode>
Each factor in $M$ prescribes root structure for $A$, hence spectral
behavior:

- Unit roots (trend/seasonality)
  - $Q (z) = 1 - z$: root at $z = 1$ → DC pole in $S (omega)$
    (nonstationary drift).
  - $Q (z) = 1 + z$: root at $z = - 1$ → Nyquist pole (alternating
    component).
  - $Q (z) = 1 - z^s$: $s$-seasonal unit roots (econometric
    seasonality).
- Damped resonances (complex conjugate pairs)
  - $Q (z) = 1 - 2 r cos omega thin z + r^2 z^2$ gives roots
    $(1 \/ r) e^(plus.minus i omega)$ → spectral poles near $omega$ with
    bandwidth controlled by $1 - r$.
  - Repetition $(1 - 2 r cos omega thin z + r^2 z^2)^m$ sharpens the
    peak (higher effective slope).
- Real poles (spectral tilt/rolloff)
  - $Q (z) = 1 - rho z$ with $rho in (- 1 , 1)$ biases
    low/high-frequency tilt (sign-convention dependent).
  - Repeated $(1 - rho z)^m$ steepens the rolloff by roughly $6 m$
    dB/octave locally.
- Compositions
  - Multiply quadratics for multiple formants (speech).
  - Mix seasonal/unit-root factors with resonant/tilt factors.
  - The net constraint set always reduces to the $L_(upright(e f f))$
    statistics from $M$.

#strong[Stationarity note.] Exact unit roots make the AR nonstationary;
use soft constraints (I-projection) or choose $rho$ just inside the unit
circle to encode "almost-unit-root" behavior while staying stable.

== 8) Degrees of freedom and feasibility
<degrees-of-freedom-and-feasibility>
- Soft constraints are always feasible (you can always shift the mean).
- For exact divisibility by all $Q_k$, you need
  $P gt.eq L_(upright(e f f))$, in which case $A (z) = M (z) R (z)$ with
  monic $R$ of degree $P - L_(upright(e f f))$. Solutions form an affine
  space of dimension $P - L_(upright(e f f))$ (unique only when
  $P = L_(upright(e f f))$).

== 9) Practical recipe
<practical-recipe>
+ Build $M = upright(l c m) (Q_1 , dots.h , Q_K)$, set
  $L_(upright(e f f)) = deg M$.
+ Form a left–null basis $N (M) = [c med F]$ for the convolution matrix
  of $M$.
+ Soft update:
  $ m^star.op = mu - Sigma F^tack.b (F Sigma F^tack.b)^(- 1) (F mu + c) , quad S^star.op = Sigma . $
+ If needed, hard-enforce divisibility via
  $(m_(upright(c o n d)) , S_(upright(c o n d)))$.

#strong[Takeaway.] Divisibility-by-${ Q_k }$ is a linear,
polynomially-structured feature map on AR coefficients. I-projection
preserves Gaussianity, leaves uncertainty intact, and injects exactly
the desired spectral biases (trends, seasonality, resonances, tilt)
through a single closed-form mean shift.
