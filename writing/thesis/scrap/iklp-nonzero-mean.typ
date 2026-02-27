==== Nonzero mean for $p(bm(w))$
Equation (6) in @Yoshii2013:
$
  epsilon.alt(t) & = sum_(j=1)^J w_j phi.alt_j (t) + eta(t) \
                 & = bm(phi.alt)(t)^top bm(w) + eta(t)
$
When sampled at times $bold(t) = {t_n}_(n=1)^N$:
$
  bm(epsilon.alt) = bm(Phi) bm(w) + bm(eta).
$
Assume
$
    bm(w) & ~ mono("Normal")(bm(mu)_w, nu_w bm(Sigma)_w) \
  bm(eta) & ~ mono("Normal")(bm(0), nu_e bm(I))
$
Then the excitation $epsilon.alt(t)$ is a GP with marginal#footnote[
  We marginalized both over $bm(w)$ and over all other values of $t$ here.
]
$
  bm(epsilon.alt) ~ mono("Normal")(bm(Phi) bm(mu)_w, nu_w bm(Phi) bm(Sigma)_w bm(Phi)^top + nu_e bm(I)).
$
The data likelihood becomes via Eq. (5) in @Yoshii2013:
$
  bm(x) ~ mono("Normal")(bm(Psi)^(-1) bm(Phi) bm(mu)_w, bm(Psi)^(-1) (nu_w bm(Phi) bm(Sigma)_w bm(Phi)^top + nu_e bm(I)) bm(Psi)^(- top))
$
For _multiple kernel learning_, we have for each $i$th GP the linear regression model
$
  {bm(Phi)_i; bm(mu)_w^((i)), nu_w bm(Sigma)_w^((i))}.
$
We assume that the unit power of each $i$th GP is comparable, so $nu_w$ plays the role of overall scale to match the excitation to the data.
#footnote[
  The AR($P$) process also has a nontrivial influence on gain, because gain, phase and frequency are all entangled, so $nu_w$ also has a beneficial untangling effect on these three -- we want the AR($P$) process to focus on phase and frequency, not gain.
]
The marginal of the total GP (summed over all indexes $i$) becomes:
$
  bm(x) ~ mono("Normal")(bm(Psi)^(-1) sum_(i=1)^I theta_i bm(m)_i, bm(Psi)^(-1) (nu_w sum_(i=1)^I theta_i bm(K)_i + nu_e bm(I)) bm(Psi)^(- top))
$ <eq:likelihood-x>
where $bm(m)_i = bm(Phi)_i bm(mu)_w^((i))$ and $bm(K)_i = bm(Phi)_i bm(Sigma)_w^((i)) bm(Phi)_i^top$.

We now have two variables controlling scale of the signal part:
1. $nu_w$: controls _overall_ covariance (power).
2. $theta_i$: controls scale of means and covariances.

So in terms of normalization:
- $nu_w$ can learn the overall scale needed for the signal, so don't need to worry here about the effects of AR($P$) models on output scale.
- Just need to make sure the individual excitation GPs $epsilon.alt_i (t)$ are comparable in terms of power, because then $theta_i$ can play the role of relative importance faithfully.
  (If say $epsilon.alt_3(t)$ and $epsilon.alt_7(t)$ have comparable importance but the latter has much lower power, than $theta_7$ will be much larger than $theta_3$ to make up for this power difference.)
  Therefore when we normalize (gauge) all $u'(t)$ examplars to unit power, the learned GPs will have means and covariances with similar power in turn, and we are good.

==== Derivation
A nonzero mean for $bm(w)_i$ changes the VI algorithm.
The mean of the likelihood for $bm(x)$ is, from @eq:likelihood-x:
$
  bm(mu) = bm(Psi)^(-1) sum_(i=1)^I theta_i bm(m)_i
$
In Eq. (23) in @Yoshii2013 we need to do
$
  bm(x) --> (bm(x) - bm(mu)) = bm(Psi) bm(x) - sum_(i=1)^I theta_i bm(m)_i
$
Here $bm(m)_i$ are constants in the optimization, but ${bm(Psi), theta_i}$ are not.
We thus get another $bb(E)[theta_i]$ in the VI steps and need to make some changes in the downstream derivations.
// we can do that derivation here

[TODO]
