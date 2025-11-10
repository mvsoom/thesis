# OPENGLOTII

Fully synthesized with varying pitch in middle of each fragment.

## Pre-results

Notes before results:

- Set f0 dependent on target_gender because I = len(f0) does not seem to want to go higher than 30; runtime increases by a factor 10. Mysterious.
- f0 bounds take into account +/- 15% deviation of true_pitch in middle of fragment
- Pitch changes during 1sec clip; not taken into account in evaluation results
- The window length has been decreased to track quick changes in pitch; and hop length has been doubled to halve compute

Errors:

- Preliminary with only a batch of 3 data points had: "All-NaN slice" in RMSE of formant estimates => should be resolved by doing full batch

Possible issues:

- The window length still too large, together with I = only 30, such that pitch changes can't be tracked well
- Good value of P? there are 4 resonances below 4000 EXCEPT for female 'i' and 'u', so I choose P = 8

## During results

- Posterior mean need not be periodic, even though SqExpPeriodic kernel is used
  - Because it is a mixture at different pitches
  - Example: ![this image](/home/marnix/thesis/figures/svg/20251108080634990.svg) from [this notebook](experiments/iklp/openglot-II/runs/True_5_0.0_1.0_1.0_0.95_0.5_male.ipynb)
  - **Consequence**: need a quasiperiodic kernel _as otherwise it will use the mixture components to get at variability_
