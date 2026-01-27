# pmatern_lf

Measure D_KL on 100 D_KL samples

Expected runtime per notebook: ~2hrs
Total runtime ~ 2 days

nu=100 means SqExpPeriodic, which is much slower: ~6 min per data point vs 1 min for the others
And in that case we take number of harmonics to be J = M // 2 which results in 2J + 1 harmonics

# Results

nu=1.5 beats the rest, but needs high M

D_KL_neff ~ 3.35 nats/sample

where 'neff' means that logz and log_prob_u are scaled by length of the waveform
-- this is simply done to acquire a sense of scale
