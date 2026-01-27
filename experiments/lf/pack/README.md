# pack_lf

Measure D_KL on 100 D_KL samples

Expected runtime per notebook: ~2hrs
Total runtime ~ 2 days

# Results

pack:1 with J=2 wins, but we select J=1 for simplicity

D_KL_neff ~ 3.37 nats/sample

where 'neff' means that logz and log_prob_u are scaled by length of the waveform
-- this is simply done to acquire a sense of scale

Close to pmatern:1.5, which is the overall winner but needs M=256 or M=512, which is a LOT compared to our J=1 and has

D_KL_neff ~ 3.35 nats/sample

So they perform virtually identical, and information H is comparable

**However**, pmatern kernels are way slower to evaluate; pack:d kernels are faster and have unlimited spectrum generated from a single call

Runtimes for lf/pack (min/median/max, in sec):
[ 17 / 65 / 895 ]
Runtimes for lf/pmatern:
[ 26 / 157 / 3971 ]

So we have a real simplicity gain and for quasiperiodic real signals we expect this harmonic capacity to give further gains
