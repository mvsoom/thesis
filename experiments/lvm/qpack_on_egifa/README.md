# lvm/qpack_dgf_test

Test learning from EGIFA with high quality GCI estimates

DGF are smoothed with 0.1 msec Gaussian derivative

- and therefore also resampled to 20 kHz = 2 samples per 0.1 msec
- turns out to be native resolution of EGIFA experiments

TODO: try rational quadratic kernel by setting `am = "rationalquadratic"`

TODO: after having learned hyperparams and gotten best kernel, _refine Z_ on 32 msec windows with fixed hyperparams!!

TODO: multiple iterations

TODO: increase WIDTH (see below)

## Bugs

OQ was wrongly calculated almost whole run through (up until halfway M=128 runs) as 1-OQ in fact
So don't use it

## Conclusions

IMPORTANT: setting WIDTH = 8192 results in svi_am_lengthscale ~= 8, while doubling WIDTH results in ~= 13 lengthscale
So this is underset

Learned surrogate model looks good, sharp features

Not enough test data to resolve model performance dependent on K

- Therefore need more synthetic data

Top performers:
```
    df[]
    compute    score svi_am_lengthscale     M     Q results.K
      <int> <errors>              <num> <int> <int>     <int>
 1:    2048   0.3(2)           8.700906   128     1        16
 2:    1024   0.3(2)           8.700906   128     1         8
 3:     512   0.3(2)           8.700906   128     1         4
 4:     256   0.3(2)           8.700906   128     1         2
 5:     128   0.3(2)           8.700906   128     1         1
 6:    1024   0.3(2)           8.683429   128     6         8
 7:    2048   0.3(2)           8.683429   128     6        16
 8:     256   0.3(2)           8.683429   128     6         2
 9:     512   0.3(2)           8.683429   128     6         4
10:     128   0.3(2)           8.683429   128     6         1
11:     128   0.3(2)           8.683303   128     3         1
12:     512   0.3(2)           8.683303   128     3         4
13:    1024   0.3(2)           8.683303   128     3         8
14:    2048   0.3(2)           8.683303   128     3        16
15:    1024   0.3(2)           8.624889   128     9         8
16:    2048   0.3(3)           8.624889   128     9        16
17:     256   0.3(2)           8.624889   128     9         2
18:     512   0.3(2)           8.624889   128     9         4
19:     128   0.2(2)           8.624889   128     9         1
20:     256   0.2(2)           8.683303   128     3         2
21:    1024   0.2(2)           9.029132    64     1        16
22:     512   0.2(2)           9.029132    64     1         8
23:     256   0.2(2)           9.029132    64     1         4
24:     128   0.2(2)           9.029132    64     1         2
25:      64   0.2(2)           9.029132    64     1         1
26:     256   0.1(2)           9.072810    64     3         4
27:     512   0.1(2)           9.072810    64     3         8
28:    1024   0.1(2)           9.072810    64     3        16
29:     128   0.1(2)           9.072810    64     3         2
30:      64   0.1(2)           9.072810    64     3         1
```

Increasing M increases score, beyond that experiment is not sensitive enough to discriminate (Q, K) influence
Although interestingly Q=1 seems to perform best