# lvm/qpack_dgf_test

Test learning from EGIFA with high quality GCI estimates

DGF are smoothed with 0.1 msec Gaussian derivative

- and therefore also resampled to 20 kHz = 2 samples per 0.1 msec

TODO: try rational quadratic kernel by setting `am = "rationalquadratic"`

TODO: after having learned hyperparams and gotten best kernel, _refine Z_ on 32 msec windows with fixed hyperparams!!

TODO: multiple iterations

## Prelim exploration

Learned surrogate model looks good, sharp features

Not enough test data to resolve model performance dependent on K

- Therefore need more synthetic data
