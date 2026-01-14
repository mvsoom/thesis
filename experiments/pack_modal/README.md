# pack_modal

Fit PACK: periodic, stationary, isotropic

Optionally allow separate sigma_c's per embedded harmonic.

Note: just 1 iteration for now

# Results

Short walltimes ~ 90 sec with some outliers

- Most score comparably in logz
- pack:1 is preferred
- a single sigma_c is sometimes better, sometimes worse => drop it
- same for normalization => drop it

## vs pmatern_modal

Not enough data; pack:1 and pmatern:32 score comparably with slight preference for latter