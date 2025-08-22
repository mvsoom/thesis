# Stable Gaussian Priors for AR($P$) Coefficients  
via Moment Matching of a Uniform PACF Law  

**Date:** \today  

---

## Abstract  

Sampling the partial–autocorrelation function (PACF) uniformly on $(-1,1)^P$ generates only covariance–stationary autoregressive (AR) coefficient vectors. We derive the *unique* Gaussian distribution $p(a\mid\mu,\Sigma)=\mathcal N(a\mid\mu,\Sigma)$ that minimises the Kullback–Leibler divergence $D_{\mathrm{KL}}\bigl(q\,\|\,p\bigr)$, where $q$ is the PACF push–forward law. The optimum has $\mu^\star=0$ by symmetry and $\Sigma^\star=\mathbb E_q[a\,a^{\top}]$, the second moment of $q$. Because the optimisation is convex, $\Sigma^\star$ is the global solution. We also discuss Monte–Carlo estimation of $\Sigma^\star$ and how the resulting covariance clarifies empirical shrinkage rules such as $\lambda\propto 1/P$ in isotropic priors.  

---

## Background: stability and the Monahan map  

For an AR($P$) process  

$$
  x_t=\sum_{p=1}^{P} a_p\,x_{t-p}+\varepsilon_t,
  \qquad \varepsilon_t\sim\mathcal N(0,\sigma^2),
$$

covariance stationarity holds iff the roots of $1-\sum_{p=1}^{P}a_p z^p$ lie inside the unit disk. The recursion of Monahan (1984) provides a smooth bijection  

$$
  T:(-1,1)^P\;\longrightarrow\;\mathcal A_P\subset\mathbb R^P,
  \qquad a = T(\phi),
$$

where $\phi=(\phi_1,\dots,\phi_P)$ are PACF parameters and $\mathcal A_P$ is the stability region. Drawing $\phi_i \stackrel{\text{iid}}{\sim}\mathrm{Unif}[-1,1]$ therefore yields only admissible $a$.  

---

## KL objective and global optimum  

Let $q$ be the density of $a=T(\phi)$ under the uniform PACF draw. We minimise  

$$
  D_{\mathrm{KL}}\!\bigl(q\,\|\,p_{\mu,\Sigma}\bigr)
  = \mathbb E_q\bigl[\log q(a)-\log p_{\mu,\Sigma}(a)\bigr],
  \qquad
  p_{\mu,\Sigma}(a)=\mathcal N(a\mid\mu,\Sigma).
$$

Terms involving $\log q$ do not depend on $(\mu,\Sigma)$. Writing the second moment $S_q:=\mathbb E_q[a\,a^{\top}]$ and expanding $\log p_{\mu,\Sigma}$ gives the objective  

$$
  F(\mu,\Sigma)=
    \tfrac12\!\Bigl(
      \log\det\Sigma
      +\operatorname{tr}\Sigma^{-1}S_q
      +(\mu-\mathbb E_q[a])^{\!\top}\Sigma^{-1}(\mu-\mathbb E_q[a])
    \Bigr)
    +\text{const.}
$$

### Optimal mean  

Uniformity of $\phi$ is sign–symmetric and the Monahan map is odd in each coordinate, hence $\mathbb E_q[a]=0$; the quadratic term is minimised by  

$$
  \mu^\star=0.
$$

### Optimal covariance  

Setting $\mu=0$, differentiate $F$ with respect to $\Sigma$:  

$$
  \nabla_\Sigma F
  =\tfrac12\Sigma^{-1}S_q\Sigma^{-1}-\tfrac12\Sigma^{-1}=0
  \;\;\Longrightarrow\;\;
  \Sigma^\star=S_q.
$$

### Global optimality  

Parameterise by the precision $\Lambda=\Sigma^{-1}$. Then  

$$
  \mathbb E_q[\log p_{0,\Lambda^{-1}}(a)]
  =-\tfrac12\,\mathrm{tr}(\Lambda S_q)+\tfrac12\log\det\Lambda+{\rm const.}
$$

The trace term is linear in $\Lambda$ and $\log\det\Lambda$ is concave, so the objective is concave in $\Lambda$. Maximising a concave function over $\Lambda\succ0$ is a convex problem, hence the stationary point $\Sigma^\star$ is the *global* minimiser of the KL divergence.  

---

## Monte–Carlo estimation of $\Sigma^\star$  

Generate $N$ draws $\phi^{(n)}\stackrel{\text{iid}}{\sim}\mathrm{Unif}[-1,1]^P$ and let $a^{(n)}=T(\phi^{(n)})$. The unbiased estimator  

$$
  \widehat\Sigma
  =\frac1N\sum_{n=1}^{N}a^{(n)}a^{(n)\top}
$$

converges at the usual $\mathcal O(N^{-1/2})$ rate. Variance may be reduced further by quasi–Monte Carlo (Sobol/Halton), Latin–hypercube sampling, or simple control variates.  

---

## Implications for popular shrinkage rules  

A common heuristic prior is isotropic  

$$
  p(a)=\mathcal N\!\bigl(a\mid0,\lambda I_P\bigr)
$$  

<!-- This is likely blabber -->

with $\lambda\propto 1/P$ or, in some empirical Bayes treatments, $\lambda\propto 1/P^2$. Moment matching provides the (seemingly) first principled derivation of *why* such scalings arise:  

- For large $P$, the diagonal of $\Sigma^\star$ shrinks roughly like $1/P$ because each coordinate $a_p$ is a bounded random transform of a uniform PACF parameter.
- Off–diagonal (lag–to–lag) correlations in $\Sigma^\star$ concentrate additional mass inside the stability region, explaining why the naive isotropic prior must shrink even harder (occasionally down to $1/P^2$) to avoid unstable draws.  

Hence $\Sigma^\star$ not only recovers the familiar $1/P$ law but also quantifies the *full* covariance structure that the heuristic ignores.

---

## Decay behavior: dampening

Plotting $\operatorname{diag}(\Sigma^*)$ reveals non-trivial structure, but with a clear decay as $p$ grows. Thus for any $P$ higher order AR coefficients are dampened.

## Discussion on PACF prior

A prior is proposed that captures stationarity optimally within Gaussian capacities. Moment matching yields a closed form that captures all second–order geometry of the stability region and the solution is a simple Monte Carlo integral which can be tabulated for all $P$. Pleasantly, the prior exhibits much-needed damping of higher order $a_p$ coefficients. The connection to shrinkage rules is yet to be investigated. 


## Bring in the spectral features: Constrained KL optimisation with linear moment conditions

***Joint optimization with soft divisibility constraints is also possible:***

So far we minimised  
$$
D_{\mathrm{KL}}\!\bigl(q\,\|\,\mathcal N(\mu,\Sigma)\bigr)
$$  
without additional restrictions. Suppose instead we require the *approximating Gaussian* itself to satisfy a linear moment condition  
$$
\mathbb E_p[f(a)] = 0, 
\qquad f(a)=Ca+d,
$$  
that is,  
$$
C\mu + d = 0.
$$  

### Solution form  

Let $S:=\mathbb E_q[a\,a^\top]$ with $\mathbb E_q[a]=0$. The constrained optimum has:  

- **Mean:**  
  $$
  \mu^\star = \arg\min_{C\mu=-d}\;\mu^\top S^{-1}\mu 
  = -\,S\,C^\top\bigl(C S C^\top\bigr)^{-1} d,
  $$
  assuming $C S C^\top$ is invertible. Otherwise use a pseudoinverse and take the minimum–$S^{-1}$–norm feasible $\mu$.  

- **Covariance:**  
  $$
  \Sigma^\star = S + \mu^\star{\mu^\star}^\top.
  $$  

### Derivation sketch  

For fixed $\mu$, the KL objective in $\Sigma$ is minimised by $\Sigma=S+\mu\mu^\top$. Using the determinant lemma and Sherman–Morrison identity,  
$$
\operatorname{tr}(\Sigma^{-1}S)+\mu^\top\Sigma^{-1}\mu = P,
$$  
so the objective reduces to  
$$
\tfrac12\log\det(S)+\tfrac12\log\!\bigl(1+\mu^\top S^{-1}\mu\bigr)+\text{const}.
$$  
Hence one only needs to minimise $\mu^\top S^{-1}\mu$ under $C\mu=-d$, giving the expression above.  

### Remarks  

- If $d=0$ (homogeneous constraint) then $\mu^\star=0$, $\Sigma^\star=S$, i.e. the original unconstrained solution.  
- The covariance is always a rank–1 inflation of $S$, reflecting the fact that enforcing a nonzero mean requires additional spread in the Gaussian.  
- The KL penalty for enforcing the constraint is  
  $$
  \tfrac12\log\!\bigl(1+\mu^{\star\top}S^{-1}\mu^\star\bigr).
  $$  

The constraint is enforced exactly: the approximating Gaussian $p$ must satisfy $\mathbb E_p[f(a)]=0$. The optimisation then chooses $(\mu,\Sigma)$ that lie in the feasible set and minimise divergence from $q$. Hence the “compromise” is only on matching $q$: the mean is shifted and the covariance inflated so that the constraint holds, while support for $q$ is preserved as well as possible. The KL objective quantifies the price of that compromise.

---

## References  

- Monahan, G.E. (1984). *A note on enforcing stationarity in autoregressive models*. Biometrika 71(2), 403–404.  
