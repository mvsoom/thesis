# pack_refine

- Add a null model to calibrate the cosine similarity score (see below)
- Try beta option => saw a lot of Nans in `pack_fs`; see if this helps

- Try 'refine' option??
  - WARNING; very sloppy implementation; JUST A TEST RUN
  - We set enforce_zero_mean=True because IKLP doesnt support prior means yet

## null model explained

We use the model itself to construct a principled internal null for calibration, corresponding to `kernel = "whitenoise"`.

The inference model is IKLP (Yoshii et al. 2013), i.e. a sum of GP kernels plus additive white noise trained using variational inference. When the kernel list is empty, inference is performed under the white-noise prior alone, while keeping the same variational family, parameterization, latent dimensionality, and numerical pipeline.

This yields a null model representing "no structure", but processed through exactly the same inference machinery as the structured models.

From the null run we estimate the distribution of cosine similarities between inferred latent vectors. In the simplest calibration, we extract

- $E_0 = \mathbb{E}[\cos(x_i, x_j)]$
- $V_0 = \mathbb{V}[\cos(x_i, x_j)]$.

For a structured kernel, cosine similarities are then standardized as  
$z_{\text{cos}}(i,j) = (\cos(x_i, x_j) - E_0) / \sqrt{V_0}$.

This removes the dominant dependence on latent dimensionality and VI-induced anisotropy, yielding a dimensionless score interpretable as surprisal relative to the model’s own null.

An associated summary statistic is the effective latent dimensionality,  
$d_{\mathrm{eff}} = 1 / V_0$,  
which reflects how many independent directions effectively contribute variance under the null and may vary across kernel configurations.

As a basic diagnostic, the null cosine distribution should be narrow and stable across conditions; strong deviations indicate residual structure or inference artefacts.

---

**Appendix: distribution-level (CDF-based) calibration**

For more precise control, calibration is performed using the full null distribution. Let $s$ denote a cosine similarity score and $x$ a vector of nuisance covariates (e.g. waveform length, pitch, analysis settings). From the white-noise run, estimate the conditional null CDF  
$F_0(s \mid x) = \Pr_0(S \le s \mid x)$.

For any kernel and observation, define the calibrated score  
$u = F_0(s \mid x)$,  
which is Uniform$(0,1)$ under the null, independent of dimensionality or other nuisance parameters. Optionally, map to a standard normal scale via $z = \Phi^{-1}(u)$.

Kernel comparisons are then carried out on the distribution of $u$ (or $z$), rather than raw cosine similarity, allowing dimension- and nuisance-free comparison using the full score distribution.
