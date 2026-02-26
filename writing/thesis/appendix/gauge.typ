#import "/writing/thesis/lib/prelude.typ": argmax, argmin, bm, expval, ncite, pcite, section-title-page
#import "/writing/thesis/lib/gnuplot.typ": gnuplot

#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/equate:0.3.2": equate, share-align
#show: equate.with()

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

= Gauge invariance in glottal inverse filtering evaluation
<chapter:gauge>

Every evaluation metric carries an implicit assumption about what constitutes a meaningful difference between two signals.
Spectral metrics make this assumption explicit: envelope-based measures deliberately ignore phase, ignore absolute timing, and ignore excitation fine structure.
The question is whether source-domain evaluation — comparing an estimated DGF waveform against a ground-truth one — should be held to the same standard, or whether it is legitimate to compare raw waveforms sample by sample.

This chapter argues that it is not legitimate, and that doing so systematically penalizes algorithms for choices that are physically unobservable — choices about what physicists call the _gauge_.
The argument is not a matter of taste.
It follows directly from the structure of the source-filter model and from the physics of acoustic propagation.
The practical consequence is a specific evaluation procedure, aligned at frame level with an affine correction, which we use throughout the experimental chapters.

== The source-filter gauge group

The speech production model is convolutional:
$
  y(t) = (u'(t) * h(t)) + epsilon(t),
$ <eq:source-filter>
where $u'(t)$ is the glottal flow derivative, $h(t)$ is the vocal tract impulse response including radiation, and $epsilon(t)$ is noise.
The inverse filtering problem asks: given $y(t)$, recover $u'(t)$ and $h(t)$.

The trouble is that the factorization is not unique.
Two transformations leave $y(t)$ invariant.

The first is _scale_: for any nonzero constant $alpha$, the pair $(alpha u'(t), h(t)/alpha)$ produces exactly the same observed signal as $(u'(t), h(t))$.
Absolute source amplitude is therefore not recoverable from acoustics alone; it is determined by convention.

The second is _time-translation_: for any constant delay $tau$, the pair $(u'(t - tau), h(t + tau))$ again produces the same convolution.
The two transformations together form a gauge group acting on $(u', h)$: the observable signal depends only on the equivalence class
$
  [(u', h)] = {(alpha thin u'(t - tau),; h(t + tau) \/ alpha) divides alpha in bb(R) without {0}, thin tau in bb(R)}.
$
Inverse filtering does not recover a unique $(u', h)$ pair.
It selects one representative from this class.

=== The implication for evaluation
is immediate: different algorithms can legitimately choose different representatives while producing identical acoustic reconstructions.
An evaluation that compares representatives directly — rather than comparing the equivalence classes they represent — is measuring gauge choice, not signal quality.
This is not a hypothetical concern.
The closed-phase LP-based algorithms (CPCA, SLP, WLP) anchor the filter by minimizing LP residual energy in the closed phase, which drives the gauge toward a particular delay convention.
Complex cepstrum decomposition (CCD) @Drugman2011 does something entirely different: it separates the maximum-phase anticausal component of the cepstrum, which selects a different gauge altogether.
The two families are not wrong in different ways; they are right in different gauges.
Evaluating them against the same raw waveform reference therefore introduces a systematic bias.

== Physical dispersion reinforces the ambiguity

The gauge ambiguity from the source-filter model is compounded by a physical one.
The vocal tract is a resonant system with a frequency-dependent phase response.
Its group delay
$
  tau_g(omega) = -dif / (dif omega) arg H(e^(j omega))
$
is not constant across frequency: narrow formants store and release energy over extended time windows, smearing the temporal relationship between the excitation event and the observed pressure waveform.
This smearing varies across vowels — close rounded vowels like /o/ and /u/ with their narrow first-formant bandwidths are empirically the most challenging cases for GIF @Chien2017 — and there is no single "correct" delay that compensates it.

The standard evaluation benchmark of #pcite(<Chien2017>) accounts for this with a fixed $0.65$~ms delay applied uniformly to all ground-truth waveforms, motivated by an assumed $22$~cm radiation distance.
This is a reasonable first-order correction, but it is a single number applied to a phenomenon that varies with vowel, pitch, voice quality, and fundamental frequency.
More importantly, it corrects for the mean delay of one physical effect while leaving the gauge ambiguity from the source-filter decomposition entirely unaddressed.
Applying a fixed shift does not constitute gauge-fixing; it shifts the entire evaluation into a particular gauge that may match the CPCA convention reasonably well but will systematically misalign with CCD estimates.

== Per-cycle alignment is the wrong granularity

The same benchmark evaluates performance in a pitch-synchronous, per-cycle fashion: ground-truth and estimated waveforms are aligned within each glottal cycle separately, using glottal closure instants extracted from the synthesizer.
This procedure is motivated by the desire to isolate the pulse shape from pitch-period-to-pitch-period variability.

The problem is that it introduces a reference frame — the glottal cycle — that is not implied by the generative model.
The source-filter interaction in @eq:source-filter occurs at frame scale: the filter $h(t)$ is estimated from a windowed segment spanning many cycles, and the gauge it selects is therefore a frame-level property, not a cycle-level one.
Different LPC-based algorithms apply their weighting functions (the closed-phase window in CPCA, the Gaussian weights around glottal closure in SLP, the piecewise-linear weights in WLP) at frame scale, and the effective delay they induce drifts slowly across frames as the filter estimate adapts.
Cycle-level alignment obscures this drift by correcting it separately at each cycle, which has the unintended effect of partially correcting for LPC-induced gauge variation on behalf of the algorithm.

Frame-level alignment is the granularity that matches the statistical assumptions of the model and the practical structure of the algorithms.

== Affine alignment at frame level

The physically motivated evaluation procedure aligns each estimated frame to the corresponding ground-truth frame by fitting an affine map and a nonnegative integer lag jointly,
$
  (a^*, b^*, tau^*) = argmin_(a, b, tau) || thin bm(y)_"est"[tau:] dot.op a + b - bm(y)_"true"[:n-tau] thin ||^2,
$ <eq:affine-lag>
where $tau in {0, 1, dots, tau_max}$ is a nonnegative integer lag measured in samples.#footnote[
  The lag is constrained to be nonnegative and to shift the estimate earlier in time, reflecting the physical direction of propagation delay: the acoustic observation $y(t)$ always lags the source, so a positive lag on the estimate corresponds to shifting it left to match the ground-truth timing.
  Allowing both directions would conflate physical delay correction with arbitrary time-warping.
]
The scalar $a$ absorbs scale and polarity; the scalar $b$ absorbs any mean offset introduced by pre-emphasis or measurement convention.
The lag $tau$ handles the frame-level delay that varies across algorithms and physical conditions.

The aligned estimate is
$
  tilde(bm(y))_"est" = a^* bm(y)_"est"[tau^*:] + b^*,
$
defined on the overlap $[:n - tau^*]$.
Performance is then measured by centered cosine similarity between $tilde(bm(y))_"est"$ and $bm(y)_"true"[:n-tau^*]$:
$
  "score" = (chevron.l tilde(bm(y))_"est" - overline(tilde(bm(y))_"est"), thin bm(y)_"true" - overline(bm(y))_"true" chevron.r) /
  (|| tilde(bm(y))_"est" - overline(tilde(bm(y))_"est") || thin || bm(y)_"true" - overline(bm(y))_"true" ||).
$ <eq:cosine-score>
Centering before taking the inner product removes the $b$ degree of freedom redundantly, making the score invariant to the bias correction.
Cosine similarity is a monotone function of RMSE after affine alignment and is bounded in $[-1, 1]$, which makes it convenient for visualization and for averaging across frames of different lengths.

The procedure is summarized in @table:gauge-sources, alongside the corresponding treatment in the EGIFA benchmark @Chien2017 for comparison.

#figure(
  tablem(
    columns: (1.2fr, 0.9fr, 0.9fr, 0.9fr),
    align: left,
    fill: (_, y) => if calc.odd(y) { rgb("#eeeeee89") },
    stroke: frame(rgb("21222C")),
  )[
    | *Gauge source* | *Origin* | *EGIFA treatment* | *This work* |
    | -------------- | -------- | ----------------- | ----------- |
    | Scale / polarity | Source-filter non-identifiability | Orthogonal projection per cycle | Affine fit per frame @eq:affine-lag |
    | Absolute delay | Source-filter non-identifiability | Fixed $0.65$~ms shift | Lag search per frame @eq:affine-lag |
    | Dispersive group delay | Vocal tract resonances | Fixed $0.65$~ms shift (partial) | Lag search per frame |
    | Algorithmic gauge | LP vs.\ cepstral estimation | Not corrected | Corrected uniformly |
    | Mean offset | Pre-emphasis / measurement | Not corrected | Affine fit ($b$ term) |
  ],
  placement: auto,
  caption: [
    *Gauge sources in GIF evaluation* and their treatment in the EGIFA benchmark of #pcite(<Chien2017>) versus this work.
    The EGIFA fixed delay of $0.65$~ms addresses one physical effect; it does not address the gauge ambiguity from the source-filter decomposition or the algorithm-dependent gauge drift.
    Affine alignment at frame level handles all sources uniformly.
  ],
) <table:gauge-sources>

== Summary

The source-filter model has a gauge group: scale and time-translation transformations that leave the observed signal invariant and therefore make the corresponding source properties unobservable from acoustics alone.
Different inverse filtering algorithms choose different representatives from this group — LP-based methods anchor on closed-phase energy, CCD anchors on cepstral phase — and any evaluation that compares representatives directly rather than correcting for this choice conflates algorithmic style with algorithmic error.

The same argument applies to the fixed-delay and per-cycle alignment choices in the EGIFA benchmark @Chien2017.
A fixed $0.65$~ms delay corrects for one specific physical effect at one specific propagation distance; it does not address the gauge freedom of the model, and it does not adapt to the algorithm being evaluated.
Per-cycle alignment introduces a reference frame that the generative model does not privilege and that partially absorbs algorithm-induced gauge drift.

Frame-level affine alignment with a lag search, followed by centered cosine similarity, addresses all of these uniformly.
It is the evaluation procedure used throughout the experimental chapters of this thesis.
