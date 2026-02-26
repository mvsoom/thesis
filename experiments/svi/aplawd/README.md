# svi/aplawd

Follow up from lvm/aplawd

NOT using dimensionality reduction (better)

We tried M=128 but it gives very little advantage over M=64, plus average (n_effective) points is ~50, so we dont need to have 128 inducing points arguably

To get std_loglike down, we do 16 iterations: OK

Dump SVI models: OK

## Observations

RBF wins at M=64 (so quantized RBF wins). The scoreboard is:
```
# best models
    kernelname     M    score  score95 lengthscale   obs_std
 1:        rbf    64 2.117(7)  2.12(1)    15.25(2)  0.137(2)
 2:  matern:52    64 2.070(7)  2.07(1)    17.35(4) 0.1442(7)
 3:        rbf    32 2.051(9)  2.05(2)    15.28(4)  0.147(1)
 4:  matern:32    64 2.029(8)  2.03(2)    18.65(5) 0.1482(6)
 5:  matern:52    32 2.008(8)  2.01(2)   17.232(8)  0.153(2)
```
Here `score` is just log likelihood over test set averaged both over all test points and runs with different iterations, and `score95` is the same but at 2 sigma level.
The scores answer:
> Which model will score best on average for a random draw from the test distribution?

Lengthscale and obs_std are only averaged over iterations.

Results are very well behaved

Note: Rational Quadratic kernel at `experiments/svi/aplawd_rational_quadratic` does not perform better than rbf

Note: Tronarp Matern-nu model at `experiments/svi/aplawd_nu` also does not perform better than rbf

### Averaging rationale

We estimate model performance as the expected test log likelihood for a random draw from the test distribution. Each training seed produces an estimate of this quantity based on a finite test set, so uncertainty arises from both optimization randomness (between seeds) and finite test sampling (within seed variability). The total uncertainty of the reported mean performance is therefore obtained by combining the variance of seed level means with the average within test set variance scaled by the test set size. This yields a standard error that provides a natural scale for judging whether differences between models are likely to reflect genuine performance differences rather than stochastic variation.
