# pack_fs

Quick thoughts:

- Previous `pack` run had scaling wrong and kernel misaligned => expected to be useless
- Windows seem way too long compared to other GIF methods, so we try out 3 window_type
  - GCI alignment really tough with long windows [see below]
  - This changes all the semantics: different hops, different number of datapoints etc!
- Still add P = {8, 10} to see if it can absorb errors in offsets

In general there really seems to be a scale problem

- We solve that normally with refining on examplars from the data set, but it remains fishy

Also:

- Time alignment is really crucial, we'll need many samples I = O(100)
- But we'll smooth over all of them and have good uncertainty, which is neat

## Too long window straining results?

For “today’s” GIF methods on sustained vowels:

IAIF / QCP variants:
typically 30–50 ms, sometimes “10 cycles (≤60 ms)”.

More aggressive LP/GIF (SWLP/LPC) on vowels:
windows up to 250 ms for robust AR estimation.

Epoch/GCI estimation blocks:
local windows roughly 1–2 pitch periods for the actual closure cues, but that is a different layer than the GIF LP window.

So your 2048 samples at 16 kHz ≈ 128 ms window is:

definitely longer than the classic “32 ms IAIF” style,

but shorter than the 250 ms SWLP window used in at least one well-cited GIF paper, and

conceptually closer to the “many cycles of a steady vowel” regime that a lot of sustained-vowel GIF work uses when they say “10 cycles or up to 60 ms” (for F0 around 150–200 Hz that’s already 50–70 ms).


## Quick results

Many nans => need to look at this

- spack:d about O(1) percent better than periodickernel, spack:1 worse
- iklp window is horrendous, "adaptive" and "iaif" both much better, with "adaptive" best overall
- we need to use cosine similarity metric from aligned RMSE, it is a simple transformation but shows MUCH more whats going on
- gauge=False slightly better, especially at low pitch
- scale_dgf_to_unit_power and P very little influence

important:

- modality of LF play big role
  * creaky < normal < whispery < breathy
    - OpenGLOTI uses 4 handpicked LF values, see gfm/lf.py for plots!
  * only for last two do we get a clear win for spack:d
  * this is because creaky and normal have long closed phase
  * could be solved with refinement (`pack_refine`)