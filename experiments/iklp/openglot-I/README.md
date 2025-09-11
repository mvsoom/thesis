# IKLP/OPENGLOT-I experiment

Config:
```
{
    "jax_enable_x64": jax_enable_x64,   # much accuracy loss/nans with x32?
    "r": r,                             # how does accuracy decrease with SVD rank?
    "beta": beta,                       # does cholesky jitter influence results much?
    "alpha_scale": alpha_scale,         # for I = 36, does alpha do something?
    "prior_pi": prior_pi,               # how much do results depend on prior voicedness?
    "ell": ell,                         # how does kernel lengthscale influence results?
}
```

We use periodic squared exponential kernel (ExpSqPeriodic) to model the glottal flow (GF).

We set `P = 9` to the known order of the AR filter used.

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

- The main determinant of VRAM requirement seems to be `r` (SVD rank), even in Cholesky regime. We should always choose SVD cutoff based on hard rank, not on energy or noise floor conditions, because then the rank depends very strongly on `ell`, and we can get cryptic OOM behavior rapidly. Better to choose rank and then quantify total energy coverage and prediction accuracy.