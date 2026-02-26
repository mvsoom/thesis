// ============================================================
// Independent section: test-set scoring
// (To be incorporated into experimental chapter)
// ============================================================

= Test-set scoring via importance-weighted marginal likelihood
<sec:test-scoring>

We evaluate held-out waveforms using an approximation to the marginal likelihood
$
  p(bm(y)_i) = integral p(bm(y)_i | f_i, bm(Lambda)_i) thin p(f_i) thin p(bm(Lambda)_i) thin dif f_i thin dif bm(Lambda)_i.
$
The latent function $f_i$ can be analytically marginalized by the collapsed GP machinery of @algo:prism, so only the integral over the precision variables remains:
$
  p(bm(y)_i) = integral p(bm(y)_i | bm(Lambda)_i) thin p(bm(Lambda)_i) thin dif bm(Lambda)_i,
$ <eq:marginal-lambda>
where $p(bm(y)_i | bm(Lambda)_i) = mono("Normal")(bm(y)_i | bm(0), bm(Q)_i^W + sigma^2 bm(W)_i^(-1))$ is the collapsed Gaussian marginal from @eq:weighted-collapsed, evaluated at $bm(W)_i = "diag"(lambda_(i 1), dots, lambda_(i N_i))$.

The integral @eq:marginal-lambda is intractable, so we estimate it by importance sampling, using the variational posterior $q_i(bm(Lambda)_i) = product_n mono("Gamma")(lambda_(i n); alpha_(i n)^*, beta_(i n)^*)$ fitted at training time as the proposal.
Drawing $S$ samples $bm(Lambda)_i^((s)) ~ q_i(bm(Lambda)_i)$ and computing importance weights
$
  w_i^((s)) = p(bm(y)_i | bm(Lambda)_i^((s))) thin p(bm(Lambda)_i^((s))) \/ q_i(bm(Lambda)_i^((s))),
$
the log marginal likelihood is estimated as
$
  log hat(p)(bm(y)_i) = log (1/S sum_(s=1)^S w_i^((s))).
$ <eq:iwelbo>
This is an importance-weighted ELBO and provides a tighter approximation to the true evidence than the variational lower bound alone.

=== Length normalization

The log evidence is extensive in the number of valid samples $n_("eff", i)$, growing roughly as $cal(O)(n_("eff", i))$.
To remove this dependence and compare models on equal footing, we measure improvement over a null model consisting of independent Student-t noise fitted to the marginal distribution of observations,
$
  Delta_i = log hat(p)_("model")(bm(y)_i) - log p_("null")(bm(y)_i),
$ <eq:delta>
and normalize per sample,
$
  s_i = Delta_i \/ n_("eff", i).
$ <eq:score>
The quantity $s_i$ is an average log-evidence improvement per datapoint: positive means the model explains that waveform better than iid noise, and the magnitude reflects how much structure has been captured.
The reported test score is the mean of $s_i$ across the test set, with its standard error across waveforms.
