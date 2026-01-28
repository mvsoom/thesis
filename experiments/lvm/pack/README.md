# lvm/pack

Learn pack:1[J=1] as a surrogate model of LF samples with qGPLVM

J=1 is chosen because best performing on pack_lf experiment

## Prelim observations

Q has weak influence on D_KL; M most

Only the product M\*K matters for downstream IKLP; Q only influences quality of the latent GMM

## Error modes

For M=64, M=128 I got nans in first ELBO optimizatin (SVI)
This was caused by Kuu = k(Z,Z) being near singular because:
1. Init Z ~ Uniform(0,1) for M = (# of inducing points) >= 64 will have at least one pair very close (distance closest pair scales like 1/M²)
2. Jitter was applied to it absolutely, not relative to diagonal

I changed 1. by grid initialization + jitter noise + random phase init and 2. by relative jitter of 1e-4 and all is well now.