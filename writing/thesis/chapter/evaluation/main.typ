== Equivalence Classes in Source and Filter Evaluation

// Group delay chat: https://chatgpt.com/c/697e21dc-6634-8384-bc7d-a47738105a35

=== Motivation

Evaluation of speech representations is never performed on raw signals without qualification.
Instead, metrics operate on equivalence classes induced by non-identifiable or intentionally discarded degrees of freedom.
This is well understood in the spectral domain, where formant- and envelope-based metrics explicitly ignore absolute phase, gain, and fine spectral structure.
In contrast, source-domain evaluation has historically treated waveform alignment and amplitude more rigidly, despite comparable identifiability limits.

We argue that source evaluation should adopt equivalence classes analogous to those already standard in filter and formant evaluation, and that doing so is not merely a modeling preference but a physical necessity once dispersive propagation through the vocal tract is acknowledged.

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
  h(t) -> a^(-1) h(t),
$

produces the same $y(t)$.
Absolute source amplitude is therefore unidentifiable.

Similarly, for any constant delay $tau$,

$
  u'(t) -> u'(t - tau), quad
  h(t) -> h(t + tau),
$

also leaves $y(t)$ unchanged.
Absolute source timing is not observable from acoustic pressure alone.

These ambiguities are fundamental to source–filter decomposition and do not depend on algorithmic details.
Any metric that penalizes constant amplitude scaling or constant delay between estimated and reference source waveforms is therefore penalizing non-identifiable degrees of freedom.

=== Dispersive Propagation and Group Delay

Beyond constant delay, the vocal tract introduces frequency-dependent phase shifts.
An all-pole resonator stores acoustic energy and releases it over time, producing a phase response whose slope varies with frequency.
The derivative of this phase defines the group delay, which increases near sharp resonances.

Crucially, this delay is not an artifact of LPC or inverse filtering.
It is a physical consequence of resonance and damping.

Rounded and close vowels are well known to exhibit lower and often narrower formants, and are consistently reported as more difficult for glottal inverse filtering @Chien2017.
From a signal-processing perspective, narrower bandwidth implies longer energy storage and therefore larger group delay.
The resulting temporal smearing shifts waveform features relative to the underlying excitation even when the model is correct.

Perfect sample-level alignment between source and speech waveforms is therefore not a physically well-posed expectation.

=== LPC and Local Entanglement of Gain, Phase, and Delay

In practice, vocal tract filters are estimated per frame using LPC or related AR models.
An all-pole filter

$
  H(z) = 1 / (1 - sum_(k=1)^p a_k z^(-k))
$

does not represent gain, phase, and delay independently.
Small changes in pole locations modify both magnitude and phase responses, inducing local variations in group delay.

Because LPC is estimated independently per frame, this entanglement is inherently local in time.
There is no physical or statistical justification for enforcing a single global delay across frames.
Any meaningful source evaluation must therefore quotient out gain and delay at the same temporal resolution at which the filter is estimated.

=== Equivalence Classes in Formant-Based Metrics

Formant metrics make these invariances explicit.
By operating on spectral envelopes rather than waveforms, they implicitly define the equivalence class

$
  x(t) ~ a x(t - tau) + text("fine structure"),
$

discarding absolute amplitude, timing, phase, and excitation-dependent detail.
Formant extraction compares signals only through the low-dimensional structure of resonances.

Importantly, formant analysis is performed per frame.
No attempt is made to enforce global phase or delay consistency across an utterance, because such quantities are neither stable nor meaningful under short-time analysis.

Equivalent invariances are standard throughout signal processing.
Radar, sonar, and seismic matched-filter detection routinely optimize over unknown time shifts before scoring similarity, explicitly treating delay as a nuisance parameter rather than an error @Oppenheim1999.

The absence of an analogous treatment in many source-evaluation pipelines is therefore historical rather than principled.


=== Source-Domain Analogy

The same reasoning applies to source evaluation.
Comparing glottal flow derivatives pointwise in time without accounting for affine and delay ambiguities is no more principled than comparing raw complex spectra including phase.

A source-domain metric should instead compare equivalence classes of the form

$
  [u'(t)] = { a u'(t - tau) + b | a in bb(R)^+, tau in bb(R), b in bb(R) },
$

evaluated per frame.

After quotienting out these nuisance degrees of freedom, similarity is naturally measured by the normalized inner product (cosine similarity) between aligned signals. Let

$
  tilde(u)'(t) = a^* u'_"est" (t - tau^*) + b^*
$

denote the affine and delay aligned estimate obtained by minimizing squared error over the equivalence class. The similarity score is then

$
  "score" = ( angle.l tilde(u)'_c , u'_("true",c) angle.r ) /
  ( ||tilde(u)'_c|| ||u'_("true",c)|| ),
$

where the subscript $c$ denotes mean centering over the overlapping support. This evaluates agreement in waveform shape while remaining invariant to global gain, bias, and constant timing shifts. In this sense, cosine similarity plays the same role in the source domain as log spectral distances do in envelope based spectral representations: both compare signals only along identifiable dimensions after quotienting out physically ambiguous transformations.

Failure to quotient out delay risks conflating physically induced dispersion with modeling error. This effect is expected to be strongest precisely in vowel classes known to exhibit narrow formants, providing a plausible explanation for systematic performance differences reported in the literature.


=== Why Per-Frame, Not Per-Cycle

Cycle-level normalization implicitly assumes a pitch-synchronous reference frame and discards inter-cycle variability.
This is appropriate only if the model itself is defined at the cycle level.

In contrast, LPC-based source–filter models estimate both source and filter per frame.
Delay, gain, and phase entanglement therefore occur at the frame level, not the cycle level.

Applying affine normalization per frame respects the resolution at which the inverse problem is posed and avoids conflating modeling assumptions with evaluation artifacts.

=== Conclusion

Formant-based filter metrics and delay-invariant source metrics operate on closely analogous equivalence classes.
Both explicitly discard absolute amplitude and timing to isolate identifiable structure.

Once the non-identifiability of the glottal source, the dispersive nature of vocal-tract propagation, and the local entanglement induced by LPC are acknowledged, delay-invariant source evaluation is not merely defensible but necessary.

Treating source equivalence classes with the same seriousness long afforded to spectral representations yields metrics that are physically grounded, statistically meaningful, and less likely to mistake propagation effects for inference failure.


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