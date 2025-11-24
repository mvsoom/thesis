# GFM experiment

- Projected time for 1 batch (210 notebooks): 34u

Modifications:

- Try/except for dyplots!!
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

[some runs failed here due to plotting errors]

- For `d=0;1` tack:0 and tack:1 align, and centered and normalized is preferred

- This expected lineup does not happen for `d=2;3`; matern:inf is preferred
  * This result is due to the fact that the fits of the NN(d=2;3) models are quite wiggly, and matern:inf picks up on this

- For `d=100`, ie the raw LF waveforms the winners are tack:1 (centered) and matern:32, which accords with differentiability and our previous thesis
  * Surprisingly, tack:0 scores very badly, and periodickernel scores better


# General observations

- Surprisingly, ACK(d) is not always paired with the model generating the examplar approximation of degree (d).
  - This is due to the details of each fit and thus not necessarily super representive; matern-inf wins for d=2 and d=3 because these are very wiggly fits for example

- Centered is definitely important, normalized less so (inc

CONCLUSION: results look good but we are more interested in just fitting variations of LF(R_d) with all models represented here; rather than examplar approximations, as the results depend too much on the details.
=> we do the `relaxation` experiment