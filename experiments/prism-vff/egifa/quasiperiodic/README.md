# egifa/quasiperiodic

We init inducing freqs via score and freeze afterwards

- Tempering of alpha=1/4 to disperse introducing point and combat degeneracy in x32

We include a baseline: approximation to Matern-nu

We set num harmonics of PACK to 32 and expansion of Matern-nu to 16, fixed

All in all quasiperiodic kernel has MUCH more freedom than periodic kernel (whose inducing frequencies were also fixed, like here)
