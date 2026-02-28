# svi/aplawd_nu

Quick test of fitting Tronarp+ (2018) expansion with M=128 (very large)

Looks good:

Observation sigma_noise: NonNegativeReal( # 1 (8 B)
value=Array(0.14168944, dtype=float64),
tag='non_negative'
)
Learned lengthscales: PositiveReal( # 1 (8 B)
value=Array(11.69549566, dtype=float64),
tag='positive'
)
Learned variance: PositiveReal( # 1 (8 B)
value=Array(1., dtype=float64, weak_type=True),
tag='positive'
)
Learned nu: PositiveReal( # 1 (8 B)
value=Array(9.54519094, dtype=float64),
tag='positive'
)

Smaller lengthscale, nu=9.5

## Actually lets try over 16 iterations

       M    score  score95       nu lengthscale  obs_std
1:    64  2.11(1)  2.11(2) 11.27(2)    16.18(6) 0.139(1)
2:    32  2.06(1)  2.06(3) 11.29(1)    16.56(2) 0.147(2)
3:    16  1.97(3)  1.97(6)  12.5(1)     16.3(3) 0.159(9)
4:     8  1.93(1)  1.93(2) 16.98(1)    19.07(3) 0.165(1)
5:     4  1.46(1)  1.46(2) 22.01(8)     25.5(1) 0.276(2)


best model M=64 scores on par with RBF [score = 2.117(7)] and has slightly longer lengthscale