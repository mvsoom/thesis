== Equivalence Classes in Source and Filter Evaluation

=== Motivation

Evaluation of speech representations is never performed on raw signals without qualification.
Instead, metrics operate on equivalence classes induced by non-identifiable or intentionally discarded degrees of freedom.
This is well understood in the spectral domain, where formant- and envelope-based metrics explicitly ignore absolute phase, gain, and fine spectral structure.
In contrast, source-domain evaluation has historically treated waveform alignment and amplitude more rigidly, despite comparable identifiability limits.
We argue that source evaluation should adopt equivalence classes analogous to those already standard in filter and formant evaluation.

=== Non-identifiability of the Glottal Source

Let the observed speech signal be modeled as
$
  y(t) = (u'(t) * h(t)) + epsilon(t),
$
where $u'(t)$ is the glottal flow derivative, $h(t)$ is the vocal tract impulse response including radiation, and $epsilon(t)$ is noise.
This factorization is not unique.
For any scalar $a != 0$,
$
  u'(t) -> a u'(t), quad
  h(t) -> a^(-1) h(t)
$
produces the same $y(t)$.
Absolute source amplitude is therefore unidentifiable.

Similarly, for any constant delay $tau$,
$
  u'(t) -> u'(t - tau), quad
  h(t) -> h(t + tau)
$
also leaves $y(t)$ unchanged.
Absolute source timing is not observable from acoustic pressure alone.
These ambiguities are fundamental to source–filter decomposition and do not depend on algorithmic details.

As a consequence, any metric that penalizes constant amplitude scaling or constant delay between estimated and reference source waveforms is penalizing non-identifiable degrees of freedom.

=== LPC and Local Entanglement of Gain, Phase, and Delay

In practice, vocal tract filters are estimated per frame using LPC or related AR models.
An all-pole filter
$
  H(z) = 1 / (1 - sum_(k=1)^p a_k z^(-k))
$
does not represent delay, phase, and gain independently.
Small changes in pole locations modify both magnitude and phase responses, inducing effective group delay variations.
A shift in formant frequency or bandwidth produces a frequency-dependent phase slope that is indistinguishable from a local time shift of the excitation within the analysis window.

Because LPC is estimated independently per frame, this entanglement is inherently local in time.
There is no physical or statistical justification for enforcing a single global delay or scale across frames.
Any meaningful source evaluation must therefore quotient out gain and delay *at the same temporal resolution at which the filter is estimated*, namely per frame.

=== Equivalence Classes in Formant-Based Metrics

Formant metrics make these invariances explicit.
By operating on spectral envelopes rather than waveforms, they implicitly define the equivalence class
$
  x(t) ~ a x(t - tau) + text("fine structure"),
$
discarding absolute amplitude, absolute timing, phase, and excitation-dependent detail.
Formant extraction compares signals only through the low-dimensional structure of resonances, typically using peak locations in smoothed log-magnitude spectra.
This is accepted practice and rarely questioned, precisely because the discarded degrees of freedom are known to be non-identifiable or irrelevant to the representation’s purpose.

Importantly, formant analysis is performed per frame.
No attempt is made to enforce global phase or delay consistency across an utterance, because such quantities are neither stable nor meaningful under short-time LPC analysis.

=== Source-Domain Analogy

The same reasoning applies to source evaluation.
Comparing glottal flow derivatives pointwise in time without accounting for affine and delay ambiguities is no more principled than comparing raw complex spectra including phase.
A source-domain metric that optimally removes per-frame affine scaling and small time shifts is the direct analogue of envelope-based spectral metrics.

Formally, the proposed source metric compares equivalence classes of the form
$
  [u'(t)] = { a u'(t - tau) + b | a in bb(R)^+, tau in bb(R), b in bb(R) },
$
evaluated per frame.
Cosine similarity after affine normalization is a natural choice on this quotient space, just as cosine or log-spectral distances are natural in envelope-based spectral spaces.

=== Why Per-Frame, Not Per-Cycle

Cycle-level normalization implicitly assumes a pitch-synchronous reference frame and discards inter-cycle variability.
This is appropriate only if the model itself is defined at the cycle level.
In contrast, LPC-based source–filter models estimate both source and filter per frame.
Delay, gain, and phase entanglement therefore occur at the frame level, not the cycle level.

Applying affine normalization per frame respects the resolution at which the inverse problem is posed and avoids conflating modeling assumptions with evaluation artifacts.
Global or per-cycle normalization imposes constraints that neither the physics nor the estimation procedure supports.

=== Conclusion

Formant-based filter metrics and proposed source-domain metrics operate on closely analogous equivalence classes.
Both explicitly discard absolute amplitude and timing to isolate identifiable structure.
The difference is historical rather than principled.
Once the non-identifiability of the glottal source and the local entanglement induced by LPC are acknowledged, per-frame affine and delay-invariant source evaluation is not only defensible but necessary.
Treating source equivalence classes with the same seriousness long afforded to spectral representations leads to metrics that are both physically grounded and statistically meaningful.

== Databases

=== OPENGLOT database

=== Chien database

/*
The MAE score used here is very nondiscriminate and VERY sensitive to delays in the time domain
Its SD (stddev) is also insane, and makes ranking basically obsolete (even though they use ranking tests)
Worse: when using the --oracle method (ie use ground truth as the test solution), the oracle scores EQUALLY to most methods, even slightly worse than a few
This is because the orace is off by 14 samples; when using --oracle-delayed the oracle has zero error, as it should
They also use per-cycle metrics but this is bullocks: there is an affine shift equivalence class for scoring metrics THAT MUST OPERATE ON THE FRAME LEVEL (not cycle level) BECAUSE LPC IS APPLIED AT THE FRAME LEVEL
The authors (Chien+) also did not do null model (--whitenoise) calibration and go over this fixed delay of 14 samples (0.65 msec) very lightly
The other metrics (H1H2, NAQ) also don't really fix any of these problems; many of them are sensitive to this shift too

Therefore: we test our method with their eval suite, and expect to score basically ~in the middle => really basic test. Plus we also reproduce their numbers so all implementations work (already confirmed)
- We only do this with our final implementation; it can choose its own window and has access to GCI information, both estimated from waveform or ground truth
- We can also do this with our own whitekernel/periodickernel implementations for baselines

The REAL test is bringing these algos to our testing env, which is far superior and actually implements the correct equivalence class (affine shifts)
AND evaluates the inferred spectrum via formants, which is a metric that ALSO IMPLEMENTS EQUIVALENCE CLASSES
- We only do this with our final implementation
- Each GIF implemented here has its own params etc (and windows)

*/