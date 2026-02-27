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

Ongoing