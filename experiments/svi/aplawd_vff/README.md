# svi/aplawd_rational_quadratic

Follow up from svi/aplawd but with Rational Quadratic kernel, with free alpha

## Performance

Defeated by RBF at M=64

```
# best models
          kernelname     M    score  score95 lengthscale     alpha   obs_std
              <char> <int> <errors> <errors>    <errors>  <errors>  <errors>
1: rationalquadratic    64 2.064(9)  2.06(2)    16.25(2) 0.6056(3)  0.147(1)
2: rationalquadratic    32 1.968(9)  1.97(2)   15.533(8) 0.0412(5) 0.1583(8)
3: rationalquadratic    16  1.81(1)  1.81(2)    17.07(6) 0.0322(1)  0.189(2)
4: rationalquadratic     8 1.546(9)  1.55(2)    18.16(2)   0.02(3) 0.2459(9)
5: rationalquadratic     4 1.180(9)  1.18(2)    18.38(3) 0.0146(2)  0.360(2)
```

Compare to results of `experiments/svi/aplawd`:

```
# best models
    kernelname     M    score  score95 lengthscale   obs_std
 1:        rbf    64 2.117(7)  2.12(1)    15.25(2)  0.137(2)
 2:  matern:52    64 2.070(7)  2.07(1)    17.35(4) 0.1442(7)
 3:        rbf    32 2.051(9)  2.05(2)    15.28(4)  0.147(1)
 4:  matern:32    64 2.029(8)  2.03(2)    18.65(5) 0.1482(6)
 5:  matern:52    32 2.008(8)  2.01(2)   17.232(8)  0.153(2)
```

Lengthscales are comparable however