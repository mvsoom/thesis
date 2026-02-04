# PRISM and t-PRISM: Process-Induced Surrogate Modeling with Collapsed Sparse Gaussian Processes

This note defines PRISM (process-induced surrogate modeling) as a variational learning procedure that maps a dataset of **independent variable length time series** into a **shared fixed dimensional latent space with uncertainty**, by learning a **global kernel-induced basis** and projecting each example into a **Gaussian posterior over basis amplitudes**. It then introduces **t-PRISM**, a robust variant that replaces Gaussian observation noise by a Student-t likelihood via local latent scale variables, yielding a local/global variational structure aligned with stochastic variational inference in the sense of Hoffman et al. (2013).

The derivations are written to serve as an implementation check against the reference code path in `svi.py` :contentReference[oaicite:0]{index=0} and the earlier PRISM/QGP concept note :contentReference[oaicite:1]{index=1}. Background motivation around collapsing inducing-variable variational parameters follows the Titsias collapsed sparse GP construction :contentReference[oaicite:2]{index=2}.

---

## 1. Problem setting and goal

### 1.1 Data

We have a dataset of $I$ independent examples

$$
\mathcal D = \{(t_i, y_i)\}_{i=1}^I,
$$

where example $i$ consists of a time grid

$$
t_i = (t_{i1},\dots,t_{iN_i}), \qquad N_i \text{ varies with } i,
$$

and observations

$$
y_i = (y_{i1},\dots,y_{iN_i}).
$$

In practice these may be irregularly sampled, time warped, and stored as NaN padded arrays (the masked waveform model used in `collapsed_elbo_masked` in `svi.py`) :contentReference[oaicite:3]{index=3}.

### 1.2 Modeling principle

Each example is modeled by an _independent_ latent function:

$$
f_i(\cdot) \sim \mathcal{GP}(0, k_\theta(\cdot,\cdot)),
$$

sharing the same kernel hyperparameters $\theta$ across $i$, but with independent draws of $f_i$.

### 1.3 PRISM goal

PRISM is not only regression. The goal is a **projection operator**

$$
(t_i,y_i)\ \mapsto\ q(\varepsilon_i) = \mathcal N(\mu_i,\Sigma_i)\ \in \mathbb R^M
$$

where:

- $M$ is fixed (the number of inducing features / basis functions),
- $\varepsilon_i$ are example-specific basis amplitudes,
- $(\mu_i,\Sigma_i)$ represent **uncertainty in latent space** for each example,
- downstream tasks can be performed in this latent space: density learning, clustering (mixture PPCA), compression, mixtures of cheap BLRs, and so on :contentReference[oaicite:4]{index=4}.

To make this feasible at scale, PRISM must avoid maintaining large per-example variational parameters.

---

## 2. A shared basis from inducing points: the Nyström prism view

Choose inducing inputs (basis locations)

$$
Z = (z_1,\dots,z_M), \qquad z_m \in \mathbb R \text{ (time coordinate)}.
$$

Define:

$$
K_{ZZ} \in \mathbb R^{M\times M}, \quad (K_{ZZ})_{mn} = k_\theta(z_m,z_n),
$$

and for example $i$ with times $t_i$:

$$
K_{Z i} \in \mathbb R^{M\times N_i}, \quad (K_{Z i})_{m n} = k_\theta(z_m, t_{in}),
$$

$$
K_{i i} \in \mathbb R^{N_i\times N_i}, \quad (K_{ii})_{n n'} = k_\theta(t_{in}, t_{in'}).
$$

A standard sparse/inducing approximation introduces inducing function values

$$
u_i = f_i(Z) \in \mathbb R^M.
$$

Conditionally,

$$
p(f_i(t_i)\mid u_i) = \mathcal N\!\left(K_{iZ}K_{ZZ}^{-1}u_i,\ K_{ii} - Q_{ii}\right),
$$

where

$$
Q_{ii} = K_{iZ}K_{ZZ}^{-1}K_{Zi}.
$$

Equivalently, in a whitened weight space view, define $L_{ZZ}$ such that

$$
K_{ZZ} = L_{ZZ}L_{ZZ}^\top,
$$

and define whitened amplitudes

$$
\varepsilon_i = L_{ZZ}^{-1} u_i.
$$

Then the (Nyström) feature map for a time $t$ is

$$
\psi_\theta(t;Z) = L_{ZZ}^{-1} k_\theta(Z,t) \in \mathbb R^M,
$$

and the low rank component of the GP can be written as

$$
f_i(t) \approx \psi_\theta(t;Z)^\top \varepsilon_i.
$$

This is exactly the “prism”: it maps each irregular time stamp $t_{in}$ to a fixed dimensional feature vector $\psi(t_{in})$, so each variable length series becomes a design matrix

$$
\Psi_i =
\begin{bmatrix}
\psi(t_{i1})^\top\\
\vdots\\
\psi(t_{iN_i})^\top
\end{bmatrix}
\in \mathbb R^{N_i\times M}.
$$

---

## 3. Gaussian PRISM: full model, ELBOs, and collapse

We begin with Gaussian noise PRISM, because it is the base case implemented in `collapsed_elbo_masked` :contentReference[oaicite:5]{index=5} and the collapse mathematics is clearest in this setting :contentReference[oaicite:6]{index=6}.

### 3.1 Likelihood and per-example joint model

Assume homoscedastic Gaussian noise with variance $\sigma^2$:

$$
p(y_i \mid f_i(t_i),\sigma^2) = \mathcal N(y_i \mid f_i(t_i),\ \sigma^2 I_{N_i}).
$$

Introduce inducing variables $u_i=f_i(Z)$ and write the joint:

$$
p(y_i,f_i,u_i)
= p(y_i\mid f_i)\ p(f_i\mid u_i)\ p(u_i),
$$

with

$$
p(u_i)=\mathcal N(0,K_{ZZ}).
$$

### 3.2 Non-collapsed variational family (what we avoid)

A standard sparse variational GP would choose, for each example $i$,

$$
q_i(u_i) = \mathcal N(m_i,S_i),
$$

and define

$$
q_i(f_i) = \int p(f_i\mid u_i)\ q_i(u_i)\ du_i.
$$

A per-example ELBO is then:

$$
\mathcal L_i^{\text{SVGP}}(m_i,S_i;Z,\theta)
=
\mathbb E_{q_i(f_i)}[\log p(y_i\mid f_i)]
-
\text{KL}\big(q_i(u_i)\,\|\,p(u_i)\big).
$$

In PRISM, examples are independent GPs but share $(Z,\theta)$. If we do not collapse, the natural construction becomes:

- global: $(Z,\theta,\sigma^2)$
- local: $(m_i,S_i)$ for each example.

This requires storing and optimizing $I$ Gaussian parameters:

- storage cost $O(IM^2)$ for the $S_i$ alone,
- update cost scales with $I$ and makes minibatching awkward because the local state is large.

This is the core computational motivation for collapse in PRISM.

### 3.3 Collapsed variational family (Titsias style)

Titsias (2009) constructs an augmented variational family that uses the exact conditional $p(f_i\mid u_i)$ and only approximates $p(u_i\mid y_i)$, then shows that for Gaussian likelihood the optimal $q_i(u_i)$ can be eliminated analytically, yielding a bound that depends only on $(Z,\theta,\sigma^2)$ :contentReference[oaicite:7]{index=7}.

We write the per-example collapsed bound directly in matrix form because it matches the code.

Define:

$$
A_i = L_{ZZ}^{-1} K_{Zi} / \sigma \in \mathbb R^{M\times N_i},
$$

and

$$
B_i = I_M + A_i A_i^\top \in \mathbb R^{M\times M}.
$$

Then the Titsias collapsed ELBO for example $i$ is:

$$
\mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2)
=
\log\mathcal N(y_i \mid 0,\ Q_{ii} + \sigma^2 I)
-\frac{1}{2\sigma^2}\text{Tr}(K_{ii}-Q_{ii}).
$$

We now express this entirely with $A_i,B_i$.

#### Step 1: log Gaussian term via Woodbury and determinant lemma

We have

$$
Q_{ii}=K_{iZ}K_{ZZ}^{-1}K_{Zi}
=
K_{iZ}L_{ZZ}^{-\top} L_{ZZ}^{-1} K_{Zi}.
$$

Define $\Psi_i = L_{ZZ}^{-1}K_{Zi}$ so that $Q_{ii} = \Psi_i^\top\Psi_i$.

Then:

$$
Q_{ii} + \sigma^2 I
=
\sigma^2\left(I + \frac{1}{\sigma^2}\Psi_i^\top\Psi_i\right).
$$

But

$$
I + \frac{1}{\sigma^2}\Psi_i^\top\Psi_i
=
I + A_i^\top A_i.
$$

By the matrix determinant lemma:

$$
\det(I + A_i^\top A_i) = \det(I + A_i A_i^\top) = \det(B_i).
$$

Hence:

$$
\log\det(Q_{ii}+\sigma^2 I)
=
N_i\log\sigma^2 + \log\det(B_i).
$$

For the quadratic form, Woodbury gives:

$$
(Q_{ii}+\sigma^2 I)^{-1}
=
\frac{1}{\sigma^2}\left(I - A_i^\top B_i^{-1} A_i\right).
$$

Therefore:

$$
y_i^\top (Q_{ii}+\sigma^2 I)^{-1} y_i
=
\frac{1}{\sigma^2}\left(
y_i^\top y_i - y_i^\top A_i^\top B_i^{-1}A_i y_i
\right).
$$

Let

$$
v_i = A_i y_i \in \mathbb R^M.
$$

Then

$$
y_i^\top A_i^\top B_i^{-1}A_i y_i = v_i^\top B_i^{-1} v_i.
$$

So:

$$
\log\mathcal N(y_i\mid 0,Q_{ii}+\sigma^2 I)
=
-\frac{1}{2}\left(
N_i\log(2\pi) + N_i\log\sigma^2 + \log\det(B_i)
+ \frac{1}{\sigma^2}\left(y_i^\top y_i - v_i^\top B_i^{-1}v_i\right)
\right).
$$

#### Step 2: trace correction term

We have:

$$
\text{Tr}(K_{ii}-Q_{ii}) = \text{Tr}(K_{ii}) - \text{Tr}(Q_{ii}).
$$

But

$$
\text{Tr}(Q_{ii})
=
\text{Tr}(\Psi_i^\top\Psi_i)
=
\text{Tr}(\Psi_i\Psi_i^\top)
=
\sum_{n=1}^{N_i}\|\psi(t_{in})\|^2
=
\| \Psi_i \|_F^2.
$$

In code, $\Psi_i$ is `Psi` and this becomes the `Qxx_diag`/`trace(AAT)` style term in `collapsed_elbo_masked` :contentReference[oaicite:8]{index=8}.

Thus:

$$
-\frac{1}{2\sigma^2}\text{Tr}(K_{ii}-Q_{ii})
=
-\frac{1}{2\sigma^2}\left(\text{Tr}(K_{ii}) - \text{Tr}(Q_{ii})\right).
$$

#### Final per-example collapsed ELBO (Gaussian PRISM)

Combining, the per-example objective is:

$$
\boxed{
\mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2)
=
-\frac{1}{2}\left(
N_i\log(2\pi) + N_i\log\sigma^2 + \log\det(B_i)
+ \frac{1}{\sigma^2}\left(y_i^\top y_i - v_i^\top B_i^{-1}v_i\right)
\right)
-\frac{1}{2\sigma^2}\left(\text{Tr}(K_{ii}) - \text{Tr}(Q_{ii})\right).
}
$$

The full dataset ELBO is the sum:

$$
\mathcal L^{\text{PRISM}}(Z,\theta,\sigma^2)
=
\sum_{i=1}^I \mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2).
$$

This is exactly what `batch_collapsed_elbo_masked` implements (with masking and stochastic scaling) :contentReference[oaicite:9]{index=9}.

### 3.4 Stochastic optimization over waveforms (collapsed SVI)

Minibatch a subset $\mathcal B$ of waveforms. An unbiased estimate is:

$$
\widehat{\mathcal L}^{\text{PRISM}}
=
\frac{I}{|\mathcal B|}
\sum_{i\in\mathcal B} \mathcal L_i^{\text{coll}}.
$$

Then use gradient ascent in $(Z,\theta,\sigma^2)$.

This is “collapsed SVI” in the pragmatic PRISM sense:

- SVI because we minibatch over independent examples.
- collapsed because $q_i(u_i)$ is not parameterized or stored.

---

## 4. Projection to latent Gaussians (the “prism output”)

Once $(Z,\theta,\sigma^2)$ are trained, each example is projected to a Gaussian over amplitudes.

### 4.1 Posterior over whitened amplitudes in Gaussian PRISM

In the whitened BLR form:

$$
y_i = \Psi_i \varepsilon_i + \epsilon_i, \qquad \epsilon_i\sim \mathcal N(0,\sigma^2 I),
$$

with prior $\varepsilon_i \sim \mathcal N(0, I)$.

Then the posterior is:

$$
q(\varepsilon_i\mid y_i) = \mathcal N(\mu_i,\Sigma_i),
$$

where

$$
\Sigma_i = (I + \sigma^{-2}\Psi_i^\top\Psi_i)^{-1},
\qquad
\mu_i = \sigma^{-2}\Sigma_i \Psi_i^\top y_i.
$$

This is exactly the computation in `infer_eps_posterior_single` (masked) :contentReference[oaicite:10]{index=10}.

### 4.2 PRISM latent dataset

PRISM outputs:

$$
\{(\mu_i,\Sigma_i)\}_{i=1}^I.
$$

This is a **fixed dimension probabilistic representation** of variable length examples. It supports:

- mixture PPCA / mixture of factor analyzers on $\mu_i$ and $\Sigma_i$,
- clustering in latent space with uncertainty,
- downstream surrogate construction: local linear surrogates in weight space :contentReference[oaicite:11]{index=11}.

---

## 5. Why collapse is essential for shared global basis learning

This section states the practical reasons explicitly, using the motivating points you gave.

### 5.1 What non-collapsed VI would require in PRISM

If we used ordinary sparse VI with non-Gaussian likelihoods, we would typically need:

- per-example variational posterior $q_i(u_i)=\mathcal N(m_i,S_i)$, or
- per-example variational posterior $q_i(\varepsilon_i)=\mathcal N(\mu_i,\Sigma_i)$, equivalently.

Either way, the variational parameters are per-example. Maintaining them across SVI minibatches requires storing and updating a list of Gaussians:

$$
\{(m_i,S_i)\}_{i=1}^I \quad \text{or}\quad \{(\mu_i,\Sigma_i)\}_{i=1}^I.
$$

This is exactly what PRISM tries to avoid during training: it wants to reconstruct local posteriors on the fly from a shared global basis.

### 5.2 Why Gauss-Hermite or generic stochastic VI does not solve it

If we keep a non-Gaussian likelihood and approximate $\mathbb E_{q(f_i)}[\log p(y_i\mid f_i)]$ by Gauss-Hermite quadrature (or other generic stochastic VI tricks), we are still in the non-collapsed setting:

- we need $q_i(u_i)$ (or $q_i(\varepsilon_i)$) to define $q(f_i)$,
- therefore we still need per-example Gaussian variational parameters.

So the fundamental issue is not the quadrature method. The issue is the need to represent per-example posterior uncertainty during training unless collapse removes those degrees of freedom analytically.

### 5.3 What collapse changes

Collapse replaces “store and optimize $(m_i,S_i)$” by:

- evaluate $\mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2)$ for each $i$,
- by solving linear algebra problems that depend only on $(Z,\theta,\sigma^2)$ and the observed $(t_i,y_i)$.

Thus global basis learning becomes feasible:

- $Z$ and $\theta$ are learned from all waveforms without carrying a large local state.
- after learning, each waveform is projected to $(\mu_i,\Sigma_i)$.

This is the PRISM logic.

---

## 6. t-PRISM: robust PRISM via Student-t likelihood and local CAVI

We now introduce t-PRISM: a robust extension of PRISM that downweights outliers within a waveform while preserving the collapsed global-basis learning structure.

### 6.1 Student-t likelihood as a scale mixture

We replace the Gaussian noise by a Student-t likelihood with degrees of freedom $\nu$ and scale $\sigma^2$:

$$
p(y_{in}\mid f_i(t_{in}),\nu,\sigma^2) = \text{StudentT}\big(y_{in}\mid f_{in},\nu,\sigma^2\big).
$$

Use the standard Normal-Gamma augmentation:

$$
\lambda_{in} \sim \text{Gamma}\left(\frac{\nu}{2},\frac{\nu}{2}\right),
\qquad
y_{in}\mid f_{in},\lambda_{in} \sim \mathcal N\left(f_{in}, \frac{\sigma^2}{\lambda_{in}}\right).
$$

Given $\lambda_i = (\lambda_{i1},\dots,\lambda_{iN_i})$, the likelihood becomes Gaussian with diagonal noise:

$$
p(y_i\mid f_i,\lambda_i,\sigma^2)
=
\mathcal N\left(y_i\mid f_i,\ \sigma^2 \Lambda_i^{-1}\right),
\qquad
\Lambda_i = \text{diag}(\lambda_{i1},\dots,\lambda_{iN_i}).
$$

### 6.2 Full per-example joint model (augmented)

For each example $i$:

$$
p(y_i,f_i,u_i,\lambda_i)
=
p(y_i\mid f_i,\lambda_i)\ p(f_i\mid u_i)\ p(u_i)\ p(\lambda_i).
$$

### 6.3 Mean-field variational family: local and global structure

t-PRISM uses the mean-field factorization:

$$
q_i(u_i,\lambda_i) = q_i(u_i)\ q_i(\lambda_i),
\qquad
q_i(\lambda_i) = \prod_{n=1}^{N_i} q_{in}(\lambda_{in}).
$$

Crucially:

- $q_i(\lambda_i)$ are **local variables** per example (and per time point within example).
- global parameters are $(Z,\theta,\sigma^2,\nu)$.
- we do not store $q_i(u_i)$; we will collapse it conditionally on $q_i(\lambda_i)$.

This matches the Hoffman et al. (2013) local/global pattern: local latent variables optimized per minibatch, global parameters optimized by stochastic gradients of the ELBO.

### 6.4 The correct ELBO for t-PRISM (per example)

The per-example ELBO is:

$$
\mathcal L_i^{t} =
\mathbb E_{q_i(u_i)q_i(\lambda_i)}[\log p(y_i\mid f_i,\lambda_i)]
+
\mathbb E_{q_i(u_i)}[\log p(u_i) - \log q_i(u_i)]
+
\mathbb E_{q_i(\lambda_i)}[\log p(\lambda_i) - \log q_i(\lambda_i)].
$$

We now simplify each term carefully.

#### Term A: expected log likelihood given $q_i(u_i)$ and $q_i(\lambda_i)$

Because $p(y_i\mid f_i,\lambda_i)$ is Gaussian with diagonal precision $\Lambda_i/\sigma^2$:

$$
\log p(y_i\mid f_i,\lambda_i)
=
-\frac{1}{2}\left(
N_i\log(2\pi) + N_i\log\sigma^2
-\log\det(\Lambda_i)
+ \frac{1}{\sigma^2}(y_i-f_i)^\top \Lambda_i (y_i-f_i)
\right).
$$

Take expectation under $q_i(u_i)q_i(\lambda_i)$:

- define expected precisions and log precisions:
  $$
  w_{in} = \mathbb E_{q_{in}}[\lambda_{in}],
  \qquad
  \ell_{in} = \mathbb E_{q_{in}}[\log\lambda_{in}].
  $$
  Then $\mathbb E[\log\det\Lambda_i] = \sum_n \ell_{in}$.
- for the quadratic term we need $\mathbb E[(y_i-f_i)^\top \Lambda_i (y_i-f_i)]$.
  Since $\Lambda_i$ is diagonal and independent of $f_i$ under mean-field:
  $$
  \mathbb E[(y_i-f_i)^\top \Lambda_i (y_i-f_i)]
  =
  \sum_{n=1}^{N_i} w_{in}\ \mathbb E[(y_{in}-f_{in})^2].
  $$
  And
  $$
  \mathbb E[(y_{in}-f_{in})^2]
  =
  (y_{in}-m_{in})^2 + v_{in},
  $$
  where
  $$
  m_{in}=\mathbb E[f_{in}],\qquad v_{in}=\text{Var}(f_{in})
  $$
  under the marginal induced by $q_i(u_i)$ and the sparse conditional.

So Term A becomes:

$$
\mathbb E[\log p(y_i\mid f_i,\lambda_i)]
=
-\frac{1}{2}\left(
N_i\log(2\pi) + N_i\log\sigma^2
-\sum_{n}\ell_{in}
+ \frac{1}{\sigma^2}\sum_n w_{in}\big((y_{in}-m_{in})^2+v_{in}\big)
\right).
$$

#### Term B: KL for inducing variables (collapsed later)

The inducing KL contribution is:

$$
\mathbb E_{q_i(u_i)}[\log p(u_i)-\log q_i(u_i)]
=
-\text{KL}(q_i(u_i)\|p(u_i)).
$$

#### Term C: KL for Gamma latent scales (local term)

Because $p(\lambda_{in}) = \text{Gamma}(\nu/2,\nu/2)$ and we will choose Gamma $q_{in}$, this KL is closed form:

$$
\mathbb E_{q_i(\lambda_i)}[\log p(\lambda_i)-\log q_i(\lambda_i)]
=
-\sum_{n=1}^{N_i} \text{KL}(q_{in}(\lambda_{in})\|p(\lambda_{in})).
$$

So the complete per-example ELBO is:

$$
\boxed{
\mathcal L_i^t
=
-\frac{1}{2}\left(
N_i\log(2\pi) + N_i\log\sigma^2
-\sum_{n}\ell_{in}
+ \frac{1}{\sigma^2}\sum_n w_{in}\big((y_{in}-m_{in})^2+v_{in}\big)
\right)
-\text{KL}(q_i(u_i)\|p(u_i))
-\sum_n \text{KL}(q_{in}(\lambda_{in})\|p(\lambda_{in})).
}
$$

This is the starting point for both the local CAVI updates and the global optimization.

### 6.5 Collapsing $q_i(u_i)$ conditional on $q_i(\lambda_i)$

Conditional on fixed weights $w_{in}$ (equivalently a diagonal noise covariance $\sigma^2\Lambda_i^{-1}$), the likelihood is Gaussian and the Titsias collapse applies exactly as in Section 3, but with per-point weights.

Define the weighted design:

$$
W_i = \text{diag}(w_{i1},\dots,w_{iN_i}),
\qquad
\widetilde y_i = W_i^{1/2} y_i,
\qquad
\widetilde \Psi_i = W_i^{1/2}\Psi_i.
$$

Then the Gaussian collapsed term for that example is:

$$
\mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2; W_i)
=
\log\mathcal N(\widetilde y_i \mid 0,\ \widetilde Q_{ii} + \sigma^2 I)
-\frac{1}{2\sigma^2}\text{Tr}(W_i(K_{ii}-Q_{ii})),
$$

where $\widetilde Q_{ii}=\widetilde\Psi_i \widetilde\Psi_i^\top$.

This matches the weighted linear algebra pattern already present in your robust sketch (weights multiply the columns of $K_{Zi}$, or equivalently rows of $\Psi_i$) and matches how one must implement the Student-t augmentation in the collapsed setting.

After collapsing $q_i(u_i)$, the per-example ELBO becomes:

$$
\boxed{
\mathcal L_i^t(Z,\theta,\sigma^2,\nu; q_i(\lambda_i))
=
\mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2; W_i)
+\frac{1}{2}\sum_{n=1}^{N_i}\ell_{in}
-\sum_{n=1}^{N_i}\text{KL}(q_{in}(\lambda_{in})\|p(\lambda_{in})).
}
$$

Explanation of the additional $\frac12\sum \ell_{in}$ term:

- in $\mathcal L_i^{\text{coll}}(\cdot;W_i)$ we used the weighted Gaussian log likelihood with covariance $\sigma^2 W_i^{-1}$, which contributes $+\frac12\log\det(W_i)$ inside the Gaussian normalization.
- the expected Gaussian normalization requires $\mathbb E[\log\det\Lambda_i] = \sum \ell_{in}$.
  So we must explicitly keep $\sum \ell_{in}$ in the ELBO, and then the remaining Gamma KL term ensures correctness.

This is exactly where robust implementations often accidentally drop terms if they only reweight residuals without including the latent-scale KL.

### 6.6 Local CAVI updates for $q_{in}(\lambda_{in})$ (closed form)

Choose

$$
q_{in}(\lambda_{in}) = \text{Gamma}(\alpha_{in},\beta_{in})
$$

(shape-rate parameterization).

To derive the coordinate update, write the terms in the ELBO that depend on $\lambda_{in}$.
From Term A and Term C, the relevant part is:

$$
\mathbb E_{q_{in}}[\log p(\lambda_{in})] - \mathbb E_{q_{in}}[\log q_{in}(\lambda_{in})]
+\frac{1}{2}\mathbb E_{q_{in}}[\log\lambda_{in}]
-\frac{1}{2\sigma^2}\mathbb E_{q_{in}}[\lambda_{in}]\,\mathbb E[(y_{in}-f_{in})^2],
$$

where $\mathbb E[(y_{in}-f_{in})^2]=(y_{in}-m_{in})^2+v_{in}$.

Using the Gamma prior $p(\lambda_{in})=\text{Gamma}(\nu/2,\nu/2)$ and conjugacy, the optimal $q_{in}$ is Gamma with:

$$
\alpha_{in} = \frac{\nu+1}{2},
\qquad
\beta_{in} = \frac{1}{2}\left(\nu + \frac{(y_{in}-m_{in})^2+v_{in}}{\sigma^2}\right).
$$

Then the moments needed in the ELBO are:

$$
w_{in} = \mathbb E[\lambda_{in}] = \frac{\alpha_{in}}{\beta_{in}},
\qquad
\ell_{in} = \mathbb E[\log\lambda_{in}] = \psi(\alpha_{in}) - \log\beta_{in}.
$$

These are exactly the closed-form local updates you were aiming for: no gradient steps, no Gauss-Hermite, just conjugate CAVI.

### 6.7 Global stochastic optimization for t-PRISM

For a minibatch $\mathcal B$ of waveforms, define:

$$
\widehat{\mathcal L}^{t\text{-PRISM}}
=
\frac{I}{|\mathcal B|}
\sum_{i\in\mathcal B}
\mathcal L_i^t(Z,\theta,\sigma^2,\nu; q_i(\lambda_i)).
$$

Algorithmically, for each waveform $i$ in the minibatch:

1. initialize local Gamma parameters (or weights) for that waveform (e.g. $w_{in}=1$),
2. iterate a fixed number of CAVI steps:
   - compute the collapsed Gaussian quantities given $W_i$ (hence compute $m_{in},v_{in}$),
   - update $(\alpha_{in},\beta_{in})$ and moments $(w_{in},\ell_{in})$,
3. evaluate $\mathcal L_i^t$.

Then take a gradient step in the global parameters $(Z,\theta,\sigma^2,\nu)$.

This is the local/global variational structure emphasized by Hoffman et al. (2013): local variational updates inside minibatches, global stochastic gradients on the ELBO.

---

## 7. t-PRISM projection: robust latent Gaussians per example

After training, we want the PRISM output for each example:

$$
q(\varepsilon_i\mid y_i) = \mathcal N(\mu_i,\Sigma_i),
$$

but now under Student-t observation noise.

With the augmentation and mean-field $q_i(\lambda_i)$, we condition on $y_i$ robustly by performing the local CAVI updates for that example (same derivation as training, but with globals fixed). Once we have weights $W_i$, the posterior over amplitudes is just the weighted BLR posterior:

$$
\Sigma_i = (I + \sigma^{-2}\Psi_i^\top W_i \Psi_i)^{-1},
\qquad
\mu_i = \sigma^{-2}\Sigma_i \Psi_i^\top W_i y_i.
$$

So t-PRISM still outputs a Gaussian per example, but the observation influence has been robustly reweighted by the inferred local scales.

This is exactly the robust conditioning you stated is essential in PRISM.

---

## 8. Implementation mapping to `svi.py` (what to change, conceptually)

The current Gaussian PRISM implementation in `svi.py` provides:

- `collapsed_elbo_masked(q,t,y)` implementing $\mathcal L_i^{\text{coll}}$ with masking :contentReference[oaicite:12]{index=12},
- `infer_eps_posterior_single(q,t,y)` implementing $(\mu_i,\Sigma_i)$ for Gaussian BLR :contentReference[oaicite:13]{index=13},
- `svi_basis(q,t)` implementing $\psi(t)$.

To implement t-PRISM, each place where the likelihood uses $\sigma^2 I$ must generalize to $\sigma^2 W_i^{-1}$ (equivalently, multiply rows by $\sqrt{w_{in}}$), and the ELBO must include the local Gamma KL terms.

The key conceptual refactor is:

### 8.1 t-PRISM ELBO per waveform

Implement a function that, for one waveform:

- runs a fixed number of local CAVI updates for $(w,\ell)$,
- evaluates
  $$
  \mathcal L_i^t
  =
  \mathcal L_i^{\text{coll}}(\cdot;W_i)
  +\frac{1}{2}\sum_n \ell_{in}
  -\sum_n \text{KL}(q_{in}\|p_{in}).
  $$

### 8.2 t-PRISM posterior over amplitudes per waveform

Implement `infer_eps_posterior_single_t` that:

- runs the same local CAVI loop to get $W_i$,
- returns weighted BLR posterior:
  $$
  \Sigma_i = (I + \sigma^{-2}\Psi_i^\top W_i \Psi_i)^{-1},
  \quad
  \mu_i = \sigma^{-2}\Sigma_i \Psi_i^\top W_i y_i.
  $$

### 8.3 Masking compatibility

Masking is already treated in `svi.py` by zeroing padded points and dropping them from kernel cross-covariances :contentReference[oaicite:14]{index=14}. For t-PRISM:

- treat masked points as absent (set $w_{in}=0$, $\ell_{in}=0$),
- ensure all sums are over effective points.

---

## 9. Summary of what PRISM and t-PRISM provide

### 9.1 PRISM (Gaussian)

- Learns a shared basis via global parameters $(Z,\theta,\sigma^2)$ using a collapsed ELBO:
  $$
  \max_{Z,\theta,\sigma^2}\ \sum_i \mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2).
  $$
- Projects each variable length example into:
  $$
  q(\varepsilon_i) = \mathcal N(\mu_i,\Sigma_i)
  $$
  giving a fixed dimensional representation with uncertainty.

### 9.2 Why collapse is essential in PRISM

PRISM wants to learn the shared basis without carrying a large per-example variational state. Non-collapsed VI would require storing and optimizing $q_i(u_i)=\mathcal N(m_i,S_i)$ (or equivalently $q(\varepsilon_i)$) per example, which is computationally and memory expensive and conflicts with the “reconstruct on the fly” design.

### 9.3 t-PRISM (robust)

t-PRISM replaces Gaussian noise by Student-t noise using Gamma latent scales, yielding:

- local variables per observed point within each waveform ($\lambda_{in}$),
- global parameters $(Z,\theta,\sigma^2,\nu)$,
- mean-field ELBO with conjugate CAVI updates for the locals and collapsed algebra for $q(u_i)$.

The central per-example objective is:

$$
\mathcal L_i^t
=
\mathcal L_i^{\text{coll}}(Z,\theta,\sigma^2;W_i)
+\frac{1}{2}\sum_n \ell_{in}
-\sum_n \text{KL}(q_{in}(\lambda_{in})\|p(\lambda_{in})).
$$

This is the robust “hack” that preserves PRISM’s core benefit:

- global basis learning remains collapsed and scalable,
- robustness is local and waveform-specific,
- projection still returns Gaussian latent representations.

---

## Appendix A: closed-form KL between Gamma distributions (shape-rate)

If

$$
q(\lambda)=\text{Gamma}(\alpha_q,\beta_q),\quad p(\lambda)=\text{Gamma}(\alpha_p,\beta_p),
$$

(shape-rate), then:

$$
\text{KL}(q\|p)
=
(\alpha_q-\alpha_p)\psi(\alpha_q)
-\log\Gamma(\alpha_q)+\log\Gamma(\alpha_p)
+\alpha_p(\log\beta_q-\log\beta_p)
+\alpha_q\left(\frac{\beta_p}{\beta_q}-1\right).
$$

This is the term needed in $\sum_n \text{KL}(q_{in}\|p_{in})$ for the t-PRISM ELBO.

---

## Appendix B: mapping to the older PRISM/QGP note

The earlier note frames the prism map as a reduced rank GP / Nyström basis and then applies mixture PPCA on the posterior weight distributions :contentReference[oaicite:15]{index=15}. The present document supplies the missing training-time variational structure that makes global basis learning feasible at scale, and introduces a robust t-PRISM variant that preserves the same downstream latent Gaussian outputs.
