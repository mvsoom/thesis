# GFM experiment

- Projected time for 1 batch (210 notebooks): 34u

Modifications:

- Run on CPU only
- Add STACK to list of kernels
- More iterations (can sum with independent sd bars)

# Observations preliminary stats during run

See `./stats.R`

## Examplar: `hard_gci`

- periodickernel consistently performs worst => encouraging news, because IKLP uses this, so we can expect improvements

  - This is because of smoothness + it can't model the hard GCI jump (needs to be smoothly joined at period end and beginning)

- There is the expected correlation between `d` and `tack:{d}`, ie, `d`-approximation of the examplar favours `tack:d` consistently

  - But, for `d = {0,1}` tack:0 and tack:1 on par with matern12 and matern32, respectively; equal within error
  - Then, for `d = {2,3}`

  - The major question is: does this remain so for the more challenging `soft_gci` case?

- For `d=100`, ie raw LF waveform examplar, matern:inf wins by large margin since it is very smooth
  => will this still hold for `soft_gci` however?

  - However, tack:2 and tack:1 follow suit; they seem to be able to model smooth curves very well but also permit (hopefully) changepoint behavior

- Centering and normalizing for the tack kernels:

  - Statistical tests don't really pick up a consistent difference for logz.

  - H = information. Normalization tends to INCREASE H (p=0.02), centering tends to DECREASE H (p=0.12).

  - Conclusion: normalization unncessary, centering slightly beneficial

## Examplar: `soft_gci`

(has a challenging return phase)

[no sufficient data yet]
