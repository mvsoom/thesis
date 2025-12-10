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

## Run done

### Observations

- tack:2 with [centered=True, normalized=True] ALWAYS win for ANY Rd
  * thus precise centering near GCI (|center-te| ~ 0 always) is absolutely crucial for the kernels to perform
  * matern:nu and periodickernel are obliterated
  * matern:32 is best of stationary kernels, but is only at the level of noncentered tack:d kernels

- Top 5 is always stack:d
  * Surprisingly, d=2 is favored strongly everywhere

- For normalized=True kernels, <sigma_a> is always VERY large
  * and this is not true for normalized=False kernels => reason to choose these, as they are typically in top 3

### More info

- Other columns we did not check yet:
  ```
  logzerr
  sigma_noise               # noise scale ~ O(0.01), as expected
  ell                       # inferred 
  ncall, niter, walltime    # NS information. Walltime ~O(1 min)
  ```

### Errors

These are all normalized tack:{2,3} kernels, and NOT(all iterations consistently fail)

```
errored:
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=True_kernel=tack:2_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=True_kernel=tack:2_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=True_kernel=tack:2_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=True_kernel=tack:2_centered=False_normalized=True_iter=4.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=True_kernel=tack:2_centered=False_normalized=True_iter=5.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=0.3_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=5.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=4.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=5.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=4.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=5.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=4.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=5.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=1.5_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=4.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=2.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.0_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=3.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=4.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.0_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=5.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.7_open_phase_only=True_kernel=tack:3_centered=False_normalized=True_iter=1.ipynb
- /home/marnix/thesis/experiments/relaxation_centered/runs/Rd=2.7_open_phase_only=False_kernel=tack:3_centered=False_normalized=True_iter=4.ipynb
```