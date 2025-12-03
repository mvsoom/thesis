# PACK

Initial quick test

Walltime per notebook: 1 min

Config:
```
{
    pitch,                          # select OPENGLOT files and kernels with that known true pitch
    kernel,                         # periodickernel or spack:d
    prior_pi,                       # prior prob of voicedness
    P,                              # do extra poles help absorbing GCI errors for spack?
    gauge,                          # does centering at GCI and applying polarity help?
    scale_dgf_to_unit_power,        # if `gauge`, does rescaling data such that DGF ~ O(1) help?
                                    # Note: polarities are NOT applied as RESKEW has 25% error on OPENGLOTI
    *,
#   refine                          # inactive: use refined spack?
}
```

## Observations

- Likely that GCI (`te`) timing is crucial as there are no "extra" poles to model errors in offset => check if P = 9 improves spack fit wrt. P = 8


## Questions

- Is `polarity_skew` always correctly inferred?
  * Note: polarities are not applied, but this has no influence anyway on spack or periodickernel (only would if spack:refined)
- Is spack:d > periodickernel?
- Does `prior_pi` matter?