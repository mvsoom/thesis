# Scaling for Mercer ops

## Approx cutoffs

|Var|Small  |Large  |
|--:|:-----:|:-----:|
|`M`|`≲2000`|`≳5000`|
|`r`|`≲50`  |`≳200` |
|`I`|`≲5`   |`≳50`  |

Other factors: precision, GPU vs CPU.

## Reduced-rank GP (I = 1)
|Regime              |Method    |Compute         |Memory              |
|--------------------|----------|---------------:|--------------------|
|`M` small, `r` small|Cholesky  |O(M^3)          |O(M^2)              |
|`M` small, `r` large|Cholesky  |O(M^3)          |O(M^2)              |
|`M` large, `r` small|Woodbury  |O(M r^2 + r^3)  |O(M r + r^2)        |
|`M` large, `r` large|Krylov (≈)|O(M r (p m + k))|O(M r + M p + M m p)|

## VI Mercer mixture (L = I·r)
|Regime              |Method    |Compute                    |Memory                    |
|--------------------|----------|--------------------------:|--------------------------:|
|`M` small           |Cholesky  |O(M^3 + I M^2 r)           |O(M^2 + I M r)            |
|`M` large, `L/M ≲ 2`|Woodbury  |O(M L^2 + L^3)             |O(I M r + L^2)            |
|`M` large, `L/M > 2`|Krylov (≈)|O(I M r (p m + p + k))     |O(I M r + M p + M m p)    |

## Variables

- `M`: number of data points.  
- `r`: number of basis functions in reduced-rank GP (per kernel).  
- `I`: number of kernels in a Mercer mixture.  
- `L = I r`: total reduced rank across all kernels.  
- `p`: number of probe vectors for stochastic trace/logdet estimation.  
- `m`: Krylov/Lanczos subspace depth.  
- `k`: number of CG iterations.
