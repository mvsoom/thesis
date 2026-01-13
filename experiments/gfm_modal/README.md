# gfm_modal

We redo the relaxation_centered experiment, but

- With LF modalities used in OPENGLOT-I (4 in total)
- With the full LF modal so the closed phase is "baked in"
- Use period of 7 msec (= geometric mean for pitch ranges of adults)

And:

- With noise floor added (-60 dB)
- Correctly multiplying the kernels with the variance, not amplitude, of sigma_a. This was a bug before

Also halved nlive: 500 -> 256

## Prelim results

Looks good, tack kernels win for all modalities

- Normalized kernels are typically better than unnormalized, but much higher information and very large sigma_a => same as in relaxation_centered experiment