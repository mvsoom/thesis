# svi/aplawd

Follow up from lvm/aplawd

NOT using dimensionality reduction (better)

We tried M=128 but it gives very little advantage over M=64, plus average (n_effective) points is ~50, so we dont need to have 128 inducing points arguably

To get std_loglike down, we do 16 iterations

Dump SVI models

FIXME: TODO: set correct model in get_meta_grouped() !!!

## Observations

Currently (runs not finished yet), the scoreboard is:
```
# best models
    kernelname     M    score  score95 lengthscale   obs_std
 1:        rbf    32 2.061(7)  2.06(1)    15.28(2)  0.148(2)
 2:  matern:52    32 1.995(7)  1.99(1)    17.23(4) 0.1538(7)
 3:        rbf    16 1.975(9)  1.98(2)    14.95(4)  0.158(1)
 4:  matern:32    32 1.953(8)  1.95(2)    18.76(5) 0.1624(6)
 ```
Here `score` is just log likelihood over test set averaged both over all test points and runs with different iterations, and `score95` is the same but at 2 sigma level.
The scores answer:
> Which model will score best on average for a random draw from the test distribution?

Lengthscale and obs_std are only averaged over iterations.

Results are very well behaved

### Averaging rationale

We estimate model performance as the expected test log likelihood for a random draw from the test distribution. Each training seed produces an estimate of this quantity based on a finite test set, so uncertainty arises from both optimization randomness (between seeds) and finite test sampling (within seed variability). The total uncertainty of the reported mean performance is therefore obtained by combining the variance of seed level means with the average within test set variance scaled by the test set size. This yields a standard error that provides a natural scale for judging whether differences between models are likely to reflect genuine performance differences rather than stochastic variation.
