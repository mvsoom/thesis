# relaxation_centered

We redo the relaxation experiment, which

- showed that tack kernels were inferior to matern kernels
- and that `centered` kernels had an advantage
  but now with centering to be inferred as well

This showed that centering made a HUGE difference: all `centered` tack kernels come out superior

And the centering is on the GCI

This was all only during _open phase_.

Now we do another final variation where we fit Rd "relaxation" waveforms with `open_phase_only` or the full LF waveform, which for Rd < 1 has a long closed phase

- and we log the posterior estimates of the parameters for all kernels (sigma_noise, ell, sigma_a, sigma_b, sigma_c, center)
