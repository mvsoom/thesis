# iklp/egifa_stationary

Test run for stationary SqExpKernel on EGIFA

Rank r=16 fixed: >99% energy captures of pack:0, so even better for pack:{1,2,3} and periodickernel

pack:d kernels got loose init values for {sigma_b, sigma_c} from the lf/pack experiment

FIXED: forgot to output 'results.voiced_group' so can't identify group level

FIXED: set P = 20 like all other algorithms

FIXED: just sample means, no variability by setting num_metrics_samples=-1

FIXED: allow only positive lags. This should have nearly zero effect as previous runs had 99%+ lags < zero.

FIXME: window length (see below)

TODO: separate vowel and speech: bimodality likely due to lower scores on speech due to (long windows AND strict periodicity)

Total time: 1550 min = 25 hr

## Important notes

Chien+ (2017) use pair tests on (file level) and aggregate from cycles to file via median. Seems wasteful.
  * We can refine the experimental unit to voiced group
  * When we do this anyway on (file level), we get same results as per frame

I_eff is a perfect predictor of performance: if < 15, good performance; else bad
  * Simply increasing size of f0 grid will help here, but will help all roughly equally anyway

Our window size is quite too long for stable LPC coeffs, so need to check that tradeoff. Stable LPC needs 32 msec or so.

We can bootstrap lag_est: very stable distribution around -0.60 with 3sigma = 1 msec span; trimodal

## Results

- ECDF view shows absolute performance against a null
  * and neutralizes effects from the equivalence class: bimodality disappears

- Wilcoxon shows relative performance
  * also neutralizes effects etc via '<' operation (Skilling style)
  * we do exactly the same as in Chien+ (2017) on (file level) but also on (frame level); results are identical in both cases
  * It says:
    ```
      pack:1
         >
      pack:2
         >
      periodickernel
         >
      pack:3
         >
      pack:0
         >
      whitenoise
   ```

- Dominance plots show which models dominate at which threshold, so you can separate "best overall" vs "best in top performance"
  * It shows periodickernel is dominated
  * It shows pack:1 best overall, pack:2 best in top performance


- Pair tests on `speech` (excluding `vowel`) give more nuanced results:
   * Here you can see that only pack:1 reliably beats everyone
   ```
                  pack:0 pack:1 pack:2 pack:3 periodickernel whitenoise
   pack:0              1      1   1.00  1.000          1.000          0
   pack:1              0      1   0.00  0.000          0.000          0
   pack:2              0      1   1.00  0.000          0.541          0
   pack:3              0      1   1.00  1.000          0.967          0
   periodickernel      0      1   0.46  0.034          1.000          0
   whitenoise          1      1   1.00  1.000          1.000          1
   ```