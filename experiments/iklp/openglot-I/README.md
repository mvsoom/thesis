# IKLP/OPENGLOT-I experiment

Config:
```
{
    "jax_enable_x64": jax_enable_x64,   # much accuracy loss/nans with x32?
    "r": r,                             # how does accuracy decrease with SVD rank?
    "beta": beta,                       # does cholesky jitter influence results much?
    "alpha_scale": alpha_scale,         # for I = 36, does alpha do something?
    "kappa": kappa,                     # lower kappa: more freedom for nu_w and nu_e
    "prior_pi": prior_pi,               # how much do results depend on prior voicedness?
    "ell": ell,                         # how does kernel lengthscale influence results?
}
```

We use periodic squared exponential kernel (ExpSqPeriodic) to model the glottal flow (GF).

~~We set `P = 9` to the known order of the AR filter used.~~
This was incorrect: `P = 8`: see below.

We use the F0 actually used to generate the data, with 10 Hz interpolation:
```
f0 = np.arange(100, 360 + 1, 10)  # I = 36 frequencies
```

This is because we want to be in the Woodbury regime which scales O[(I*r)^3]

Some preliminary observations before looking at the run results:

- The results depend dramatically on `ell`. If `ell == 2` (our original default), all we can infer is GF = noise ~= 0 and noise = GF-like shaped signal ~= O(1), and completely miss F0 inference. Once `ell <= 1`, F0 inference works perfectly and GF/formant inference further depends on hyperparamaters.

- In general, however, ExpSqPeriodic is way too smooth for the GF, even for small `ell`. Because its spectrum decays very fast, it models most of the low frequency content; alongside which a very specific AR filter is inferred to shape the higher frequencies. The inferred GF is not convincing; specifically the polarity is not picked up (ie positive or negative) and the DC is zero (we want pos or neg).

- Sometimes we get very good (perfect) AR filter inference; this happens when the noise is modeling the very sharp transients at GCI. So getting those right is the key to good AR inference.

- Preliminary tests with our ARPrior do not show any real influence of it; the AR prior is overwhelmed by the kernel behavior. Only when we have our GF model more realistic can this be expected to influence the fine details.

- x32 mode can produce nans: 99% of these come from safe_cholesky(), which can be eliminated by setting `beta = 1`. The other 1% probably comes from the GIG distribution calculations. The `elbo` always flags if a nan anywhere in the calculation occured as it aggregates all the information, and the `vi_run_criterion` will stop.

- The main determinant of VRAM requirement seems to be `r` (SVD rank), even in Cholesky regime. We should always choose SVD cutoff based on hard rank, not on energy or noise floor conditions, because then the rank depends very strongly on `ell`, and we can get cryptic OOM behavior rapidly. Better to choose rank and then quantify total energy coverage and prediction accuracy. OOMs start happening in this setup at around `r >= 40`.

## DGF

Code at `data/OPENGLOT/RepositoryI/Generation_codes` shows that DGF not GF is the source. The "9th pole" mentioned in the paper is just a differentiation pole, so true `P = 8`.

Setting `P = 8` and comparing inferred signal to DGF gave a big performance boost!

Working with DGF also gives rise to scale issue: the `d/dt` makes DGF amplitude = O(500) while amplitude(x) = O(1). This while the model is setup for O(1) power of both signal ($\nu_w, \theta_i$) and noise ($\nu_e$), in contradiction with true DGF used to generate data.

*This large DGF scale might only be a problem with the LF model (very sharp) anyway.*

Nevertheless, the model is able to find good estimates for `u'(t)` with O(1) amplitude -- in accordance with the prior, but not with ground truth. Thus it uses the AR filter to "lift power" -- thereby not completely wrecking the formant estimates however, but nevertheless sacrificing accuracy there.

I tried normalizing DGF to unit power and thereby rescaling data but this was disastrous for the model. So we leave data at unit power (and mean zero) and get the model to learn the scale. This can be done in 3 ways:

1. Sharp resonances in AR (bad)
2. Set $\nu_w$ to correct scale (preferred)
3. Make $\theta_i$ larger (acceptable)

Option 2 is controlled by `s` and `kappa` hyperparams. We set `s = 1` (uninformative), so that leaves `kappa` as a knob to control variance of `E(nu_w + nu_e)`. `kappa` also interferes with `p` so not ideal.

Option 3 is controlled by `alpha`: smaller `alpha` means smaller `I_eff` **and** larger variance of the sum of $\theta_i$. So this can be a nice knob.

This is essentially an empirical tradeoff, so we try out some combinations.