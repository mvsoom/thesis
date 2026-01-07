# gfm_modal_amp_noise

We redo gfm_modal experiment, but:

- With noise floor added (-60 dB)
- Correctly multiplying the kernels with the variance, not amplitude, of sigma_a. This was a bug before

Also halved nlive: 500 -> 256
