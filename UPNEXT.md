# UPNEXT

PRocess-Induced Surrogate Modeling

PRISM method:
https://chatgpt.com/g/g-p-696643a2ef3c8191906b9083e9248891-prism/c/69661563-f694-832d-846d-9c723f864682

Now comparing periodic Matern with PACK in pmatern-modal vs pack-modal

Interesting:

- Periodic Matern: Mercer expansion analytically available, via expansion spectrum drops
- PACK: spectrum not analytically available (via FFT), spectrum always "complete"
  - PACK has different semantics

PACK:

- STATIONARY and isotropic!! Has a spectrum, though analytically not available
- Depends only on sum of cosines
- Still has neural flavor
- Could be used directly in IKLP experiment after we see which is best; same for Matern kernels
  - See .spectrum_from_fft() method to get Mercer expansion

Next up:

- Check comparison on modal
- Run IKLP experiments with SqExp, Matern (make SqExp once nu > 100 say), PACK tuned on modal

  - Implement baseline whitening: experiments/pack_refine/README.md
  - And more proper window: experiments/pack_fs/README.md

- Get inducing point method running
  - source.get_lf_samples(10_000) is running => can use more samples
- Do the mixture PPCA and check D_KLs

## currently running

tmux:0
source.get_lf_samples(10_000)

tmux:1
$ python -m experiments.run execute experiments/pack_lf && python -m experiments.run collect experiments/pack_lf && python -m experiments.run execute experiments/pmatern_lf && python -m experiments.run collect experiments/pmatern_lf
