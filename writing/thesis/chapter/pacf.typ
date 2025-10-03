= Stable Gaussian Priors for AR($P$) Coefficients
<stable-gaussian-priors-for-arp-coefficients>
via Moment Matching of a Uniform PACF Law

== Abstract
<abstract>
Sampling the partial–autocorrelation function (PACF) uniformly on
$(- 1 , 1)^P$ generates only covariance–stationary autoregressive (AR)
coefficient vectors. We derive the #emph[unique] Gaussian distribution
$p (a divides mu , Sigma) = cal(N) (a divides mu , Sigma)$ that
minimises the Kullback–Leibler divergence
$D_(upright(K L)) #scale(x: 120%, y: 120%)[\(] q thin parallel thin p #scale(x: 120%, y: 120%)[\)]$,
where $q$ is the PACF push–forward law. The optimum has $mu^star.op = 0$
by symmetry and $Sigma^star.op = bb(E)_q [a thin a^tack.b]$, the second
moment of $q$. Because the optimisation is convex, $Sigma^star.op$ is
the global solution. We also discuss Monte–Carlo estimation of
$Sigma^star.op$ and how the resulting covariance clarifies empirical
shrinkage rules such as $lambda prop 1 \/ P$ in isotropic priors.

== Background: stability and the Monahan map
<background-stability-and-the-monahan-map>
For an AR($P$) process

$ x_t = sum_(p = 1)^P a_p thin x_(t - p) + epsilon_t , #h(2em) epsilon_t tilde.op cal(N) (0 , sigma^2) , $

covariance stationarity holds iff the roots of
$1 - sum_(p = 1)^P a_p z^p$ lie inside the unit disk. The recursion of
Monahan (1984) provides a smooth bijection

$ T : (- 1 , 1)^P #h(0em) arrow.r #h(0em) cal(A)_P subset bb(R)^P , #h(2em) a = T (phi.alt) , $

where $phi.alt = (phi.alt_1 , dots.h , phi.alt_P)$ are PACF parameters
and $cal(A)_P$ is the stability region. Drawing
$phi.alt_i tilde.op^(upright("iid")) upright(U n i f) [- 1 , 1]$
therefore yields only admissible $a$.

== KL objective and global optimum
<kl-objective-and-global-optimum>
Let $q$ be the density of $a = T (phi.alt)$ under the uniform PACF draw.
We minimise

$ D_"KL"(q thin parallel thin p_(mu , Sigma) ) = bb(E)_q #scale(x: 120%, y: 120%)[\[] log q (a) - log p_(mu , Sigma) (a) #scale(x: 120%, y: 120%)[\]] , #h(2em) p_(mu , Sigma) (a) = cal(N) (a divides mu , Sigma) . $

Terms involving $log q$ do not depend on $(mu , Sigma)$. Writing the
second moment $S_q := bb(E)_q [a thin a^tack.b]$ and expanding
$log p_(mu , Sigma)$ gives the objective

$
  F(mu,Sigma)= 1/2
    (
      "logdet" Sigma
      + tr Sigma^(-1) S_q
      +(mu-bb(E)_q[a])^top Sigma^(-1)(mu- bb(E)_q[a]))
    + "const."
$

=== Optimal mean
<optimal-mean>
Uniformity of $phi.alt$ is sign–symmetric and the Monahan map is odd in
each coordinate, hence $bb(E)_q [a] = 0$; the quadratic term is
minimised by

$ mu^star.op = 0 . $

=== Optimal covariance
<optimal-covariance>
Setting $mu = 0$, differentiate $F$ with respect to $Sigma$:

$ nabla_Sigma F = 1 / 2 Sigma^(- 1) S_q Sigma^(- 1) - 1 / 2 Sigma^(- 1) = 0 #h(0em) #h(0em) arrow.r.double #h(0em) #h(0em) Sigma^star.op = S_q . $

=== Global optimality
<global-optimality>
Parameterise by the precision $Lambda = Sigma^(- 1)$. Then

$
  bb(E)\q [log p_(0,Lambda^(-1))(a)]
  =-1/2 "tr" (Lambda S_q)+ 1/2 "logdet" Lambda + "const."
$

The trace term is linear in $Lambda$ and $log det Lambda$ is concave, so
the objective is concave in $Lambda$. Maximising a concave function over
$Lambda succ 0$ is a convex problem, hence the stationary point
$Sigma^star.op$ is the #emph[global] minimiser of the KL divergence.

== Monte–Carlo estimation of $Sigma^star.op$
<montecarlo-estimation-of-sigmastar>
Generate $N$ draws
$phi.alt^((n)) tilde.op^(upright("iid")) upright(U n i f) [- 1 , 1]^P$
and let $a^((n)) = T (phi.alt^((n)))$. The unbiased estimator

$ hat(Sigma) = 1 / N sum_(n = 1)^N a^((n)) a^((n) tack.b) $

converges at the usual $cal(O) (N^(- 1 \/ 2))$ rate. Variance may be
reduced further by quasi–Monte Carlo (Sobol/Halton), Latin–hypercube
sampling, or simple control variates.

== Implications for popular shrinkage rules
<implications-for-popular-shrinkage-rules>
A common heuristic prior is isotropic

$ p (a) = cal(N) ( a divides 0 , lambda I_P ) $

with $lambda prop 1 \/ P$ or, in some empirical Bayes treatments,
$lambda prop 1 \/ P^2$. Moment matching provides the (seemingly) first
principled derivation of #emph[why] such scalings arise:

- For large $P$, the diagonal of $Sigma^star.op$ shrinks roughly like
  $1 \/ P$ because each coordinate $a_p$ is a bounded random transform
  of a uniform PACF parameter.
- Off–diagonal (lag–to–lag) correlations in $Sigma^star.op$ concentrate
  additional mass inside the stability region, explaining why the naive
  isotropic prior must shrink even harder (occasionally down to
  $1 \/ P^2$) to avoid unstable draws.

Hence $Sigma^star.op$ not only recovers the familiar $1 \/ P$ law but
also quantifies the #emph[full] covariance structure that the heuristic
ignores.

== Decay behavior: dampening
<decay-behavior-dampening>
Plotting $"diag" (Sigma^(\*))$ reveals non-trivial structure, but with a
clear decay as $p$ grows. Thus for any $P$ higher order AR coefficients
are dampened.

== Discussion on PACF prior
<discussion-on-pacf-prior>
A prior is proposed that captures stationarity optimally within Gaussian
capacities. Moment matching yields a closed form that captures all
second–order geometry of the stability region and the solution is a
simple Monte Carlo integral which can be tabulated for all $P$.
Pleasantly, the prior exhibits much-needed damping of higher order $a_p$
coefficients. The connection to shrinkage rules is yet to be
investigated.

== Bring in the spectral features: Constrained KL optimisation with linear moment conditions
<bring-in-the-spectral-features-constrained-kl-optimisation-with-linear-moment-conditions>
#strong[#emph[Joint optimization with soft divisibility constraints is
also possible:];]

So far we minimised \
$ D_(upright(K L)) #h(-1em) #scale(x: 120%, y: 120%)[\(] q thin parallel thin cal(N) (mu , Sigma) #scale(x: 120%, y: 120%)[\)] $
\
without additional restrictions. Suppose instead we require the
#emph[approximating Gaussian] itself to satisfy a linear moment
condition \
$ bb(E)_p [f (a)] = 0 , #h(2em) f (a) = C a + d , $ \
that is, \
$ C mu + d = 0 . $

=== Solution form
<solution-form>
Let $S := bb(E)_q [a thin a^tack.b]$ with $bb(E)_q [a] = 0$. The
constrained optimum has:

- #strong[Mean:] \
  $ mu^star.op = arg min_(C mu = - d) #h(0em) mu^tack.b S^(- 1) mu = - thin S thin C^tack.b #scale(x: 120%, y: 120%)[\(] C S C^tack.b #scale(x: 120%, y: 120%)[\)]^(- 1) d , $
  assuming $C S C^tack.b$ is invertible. Otherwise use a pseudoinverse
  and take the minimum–$S^(- 1)$–norm feasible $mu$.

- #strong[Covariance:] \
  $ Sigma^star.op = S + mu^star.op mu^star.op^tack.b . $

=== Derivation sketch
<derivation-sketch>
For fixed $mu$, the KL objective in $Sigma$ is minimised by
$Sigma = S + mu mu^tack.b$. Using the determinant lemma and
Sherman–Morrison identity, \
$ "tr" (Sigma^(- 1) S) + mu^tack.b Sigma^(- 1) mu = P , $ \
so the objective reduces to \
$ 1 / 2 log det (S) + 1 / 2 log #h(-1em) #scale(x: 120%, y: 120%)[\(] 1 + mu^tack.b S^(- 1) mu #scale(x: 120%, y: 120%)[\)] + upright("const") . $
\
Hence one only needs to minimise $mu^tack.b S^(- 1) mu$ under
$C mu = - d$, giving the expression above.

=== Remarks
<remarks>
- If $d = 0$ (homogeneous constraint) then $mu^star.op = 0$,
  $Sigma^star.op = S$, i.e.~the original unconstrained solution. \
- The covariance is always a rank–1 inflation of $S$, reflecting the
  fact that enforcing a nonzero mean requires additional spread in the
  Gaussian. \
- The KL penalty for enforcing the constraint is \
  $ 1 / 2 log #h(-1em) #scale(x: 120%, y: 120%)[\(] 1 + mu^(star.op tack.b) S^(- 1) mu^star.op #scale(x: 120%, y: 120%)[\)] . $

The constraint is enforced exactly: the approximating Gaussian $p$ must
satisfy $bb(E)_p [f (a)] = 0$. The optimisation then chooses
$(mu , Sigma)$ that lie in the feasible set and minimise divergence from
$q$. Hence the "compromise" is only on matching $q$: the mean is shifted
and the covariance inflated so that the constraint holds, while support
for $q$ is preserved as well as possible. The KL objective quantifies
the price of that compromise.

== References
<references>
- Monahan, G.E. (1984). #emph[A note on enforcing stationarity in
  autoregressive models];. Biometrika 71(2), 403–404.