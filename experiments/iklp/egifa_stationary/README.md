# iklp/egifa_stationary

Test run for stationary SqExpKernel on EGIFA

Each iteration takes ~3 hrs

Rank r=16 fixed: >99% energy captures of pack:0, so even better for pack:{1,2,3} and periodickernel

pack:d kernels got loose init values for {sigma_b, sigma_c} from the lf/pack experiment

## Preliminary results [on `vowel` collection only]

- ECDF view shows absolute performance against a null
  * and neutralizes effects from the equivalence class: bimodality disappears

- Wilcoxon shows relative performance
  * also neutralizes effects etc via '<' operation (Skilling style)
  * we do exactly the same as in Chien+ (2017)
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