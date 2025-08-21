# Maximum-entropy (I-projection) updates for AR coefficient priors with soft divisibility features

## 1) AR context and notation
We model an AR(P) via
$$
A(z)=1+a_1 z+\cdots+a_P z^P,\qquad a=(a_1,\dots,a_P)^\top.
$$
The (unnormalized) spectrum is
$$
S(\omega)\propto \frac{1}{\lvert A(e^{-i\omega})\rvert^{2}}.
$$
Thus zeros of $A(z)$ near the unit circle become spectral poles; multiplicity controls local slope/peakedness. Stability/causality requirements translate to constraints on the root locations of $A$ (sign convention dependent).

## 2) Collapse multiple features to one effective divisor
Given monic feature polynomials $Q_k(z)=1+q_{k,1}z+\cdots+q_{k,L_k}z^{L_k}$ (possibly of different degrees), define
$$
M(z)=\mathrm{lcm}\{Q_1(z),\dots,Q_K(z)\},\qquad L_{\mathrm{eff}}=\deg M.
$$
Then $Q_k\mid A\ \forall k$ iff $M\mid A$. All independent constraints come from $M$.

**Interpretation.** Over $\mathbb{C}$, if $Q_k(z)=\prod_r(1-\rho_r z)^{m_r^{(k)}}$, then
$$
M(z)=\prod_r (1-\rho_r z)^{m_r},\quad m_r=\max_k m_r^{(k)},\quad L_{\mathrm{eff}}=\sum_r m_r.
$$
So $M$ collects the union of desired roots with their maximum multiplicities across features.

## 3) Linear remainder constraints
Let $[1;a]\in\mathbb{R}^{P+1}$ be the full coefficient vector. Build any left–null basis $N(M)\in\mathbb{R}^{L_{\mathrm{eff}}\times(P+1)}$ for the convolution/Sylvester matrix of $M$ so that
$$
N(M)\,[1;a]=0\ \Longleftrightarrow\ M\mid A.
$$
Write $N(M)=[\,c\ \ F\,]$ with $c\in\mathbb{R}^{L_{\mathrm{eff}}}$ (acts on the fixed $a_0=1$) and $F\in\mathbb{R}^{L_{\mathrm{eff}}\times P}$. The $L_{\mathrm{eff}}$ linear statistics are
$$
f(a;M)=F a + c\in\mathbb{R}^{L_{\mathrm{eff}}}.
$$
Exact divisibility is $f(a;M)=0$.

## 4) Gaussian base prior
$$
a\sim \mathcal{N}(\mu,\Sigma),\qquad \Sigma\succ 0.
$$

## 5) Soft divisibility via I-projection (moment constraints)
Impose only expectations
$$
\mathbb{E}[f(a;M)]=F\mu+c=0.
$$
Find $\mathcal{N}(m,S)$ minimizing $D_{\mathrm{KL}}(\mathcal{N}(m,S)\,\|\,\mathcal{N}(\mu,\Sigma))$ subject to $F m + c=0$. Because constraints touch the mean only, covariance is unchanged and the mean is the $\Sigma^{-1}$-orthogonal projection onto the affine set:
$$
S^\star=\Sigma,\qquad
m^\star=\mu-\Sigma F^\top (F\Sigma F^\top)^{-1}\,(F\mu+c).
$$
If rows of $F$ are redundant, replace $(F\Sigma F^\top)^{-1}$ by the Moore–Penrose pseudoinverse.

## 6) Hard divisibility (conditioning)
If you want $f(a;M)=0$ almost surely,
$$
\begin{aligned}
m_{\mathrm{cond}}&=\mu-\Sigma F^\top (F\Sigma F^\top)^{-1}(F\mu+c),\\
S_{\mathrm{cond}}&=\Sigma-\Sigma F^\top (F\Sigma F^\top)^{-1} F\Sigma.
\end{aligned}
$$
Same mean shift; variance along constrained directions is removed.

## 7) What kinds of AR features can $Q$ encode?
Each factor in $M$ prescribes root structure for $A$, hence spectral behavior:

- Unit roots (trend/seasonality)
    - $Q(z)=1-z$: root at $z=1$ → DC pole in $S(\omega)$ (nonstationary drift).
    - $Q(z)=1+z$: root at $z=-1$ → Nyquist pole (alternating component).
    - $Q(z)=1-z^s$: $s$-seasonal unit roots (econometric seasonality).

- Damped resonances (complex conjugate pairs)
    - $Q(z)=1-2r\cos\omega\, z + r^2 z^2$ gives roots $(1/r)e^{\pm i\omega}$ → spectral poles near $\omega$ with bandwidth controlled by $1-r$.
    - Repetition $(1-2r\cos\omega\, z + r^2 z^2)^m$ sharpens the peak (higher effective slope).

- Real poles (spectral tilt/rolloff)
    - $Q(z)=1-\rho z$ with $\rho\in(-1,1)$ biases low/high-frequency tilt (sign-convention dependent).
    - Repeated $(1-\rho z)^m$ steepens the rolloff by roughly $6m$ dB/octave locally.

- Compositions
    - Multiply quadratics for multiple formants (speech).
    - Mix seasonal/unit-root factors with resonant/tilt factors.
    - The net constraint set always reduces to the $L_{\mathrm{eff}}$ statistics from $M$.

**Stationarity note.** Exact unit roots make the AR nonstationary; use soft constraints (I-projection) or choose $\rho$ just inside the unit circle to encode “almost-unit-root” behavior while staying stable.

## 8) Degrees of freedom and feasibility
- Soft constraints are always feasible (you can always shift the mean).
- For exact divisibility by all $Q_k$, you need $P\ge L_{\mathrm{eff}}$, in which case $A(z)=M(z)R(z)$ with monic $R$ of degree $P-L_{\mathrm{eff}}$. Solutions form an affine space of dimension $P-L_{\mathrm{eff}}$ (unique only when $P=L_{\mathrm{eff}}$).

## 9) Practical recipe
1. Build $M=\mathrm{lcm}(Q_1,\dots,Q_K)$, set $L_{\mathrm{eff}}=\deg M$.
2. Form a left–null basis $N(M)=[c\ F]$ for the convolution matrix of $M$.
3. Soft update:
     $$
     m^\star=\mu-\Sigma F^\top (F\Sigma F^\top)^{-1}(F\mu+c),\quad S^\star=\Sigma.
     $$
4. If needed, hard-enforce divisibility via $(m_{\mathrm{cond}},S_{\mathrm{cond}})$.

**Takeaway.** Divisibility-by-$\{Q_k\}$ is a linear, polynomially-structured feature map on AR coefficients. I-projection preserves Gaussianity, leaves uncertainty intact, and injects exactly the desired spectral biases (trends, seasonality, resonances, tilt) through a single closed-form mean shift.
