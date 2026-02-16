= Conclusion
<chapter:conclusion>

#image("3d.pdf")

/*
Nice introduction to importance of clinical speech science as a whole in @Andrade-Miranda2024
*/
A person speaks 16,000 words on average each day: glottal folds are used plenty, so disorders arise quickly when used inappropriately. @Andrade-Miranda2024

Possible improvements/future directions:
- $p(a)$ not as a delta function in IKLP
- work in spectral domain completely
- model $h(t)$ as a GP conditional on $e(t)$ given: alternate optimization
- interframe correlations as in @Mehta2012
- learn from more realistic glottal flow simulations like @Avhad2022 @Zhang2020a @Schoder2024; perhaps learn first a mapping from glottal flow area to glottal flow from physics and then use real life measured glottal flow area databases like @Andrade-Miranda2024


Recent article in Nature Parkison's disease
"Speech impairments, on the other hand, are particularly prevalent, affecting up to 90% of people with PD"
"Growing evidence indicates that speech and language alterations often precede the defining motor signs and PD diagnosis *by as much as a decade*" @Cao2025
"Recent research highlights the value of *objective acoustic speech markers* in detecting PD in the initial or even prodromal (e.g., RBD) stages23,25,26. This implicates a potential window for timely interventions, such as monoamine oxidase B inhibitors, anticholinergics and future novel therapies that may confer improved efficacy in early PD stages" @Cao2025

> The recommended and widely employed tests are 1) sustained phonation of the vowel /a/ in a single breath, which helps assess breath control, vocal fold function and vocal quality @Cao2025
> 2) rapid repetition of syllables such as /pa/, /ta/ and /ka/ to evaluate consonant and vowel articulation, articulation rate and regularity, coordination and speech timing;

#figure(
  image("holmberg.pdf"),
  placement: auto,
  caption: [*Test image* from Plotly.],
) <fig:plotly-test>

Important features: vocal fold oscillation irregularity, breathiness and noise, *and vocal tract resonance fluctuations* // https://pubmed.ncbi.nlm.nih.gov/22249592/

Detection rate in clinical lab conditions is already very high
But algorithms could be improved
And perhaps we could have apps on mobile phone to cut costs even more; this would mean nonideal conditions so better algos needed

// youtube 2012 news video on this: https://www.youtube.com/watch?v=ZcLOocMSWfE

/*
PD biomarkers detectable with GIF + source-filter modeling from @Cao2025

| Biomarker                      | Detectable | Notes                                                     |
| ------------------------------ | ---------- | --------------------------------------------------------- |
| Fundamental frequency (f0)     | Yes        | From glottal cycle timing                                 |
| Jitter                         | Yes        | Cycle-to-cycle f0 variation                               |
| Shimmer                        | Yes        | Cycle-to-cycle amplitude variation                        |
| Harmonics-to-noise ratio (HNR) | Yes        | Strong suit of GIF                                        |
| Cepstral peak prominence (CPP) | Yes        | Periodicity / harmonic structure                          |
| Voice tremor                   | Yes        | Low-frequency modulation of f0 or amplitude               |
| Breathy voice                  | Yes        | Via noise, spectral tilt, open quotient                   |
| Harsh voice                    | Partial    | Detectable via noise, irregularity, high-frequency energy |
| Asthenic voice                 | Partial    | Via reduced amplitude, weak excitation, low HNR           |
| Resonance                      | Yes        | From formant structure and spectral envelope              |
| Loudness                       | Yes        | From intensity / RMS energy (not perceptual loudness)     |
| Pitch                          | Yes        | Equivalent to f0                                          |
| Vowel articulation index (VAI) | Yes        | Derived from formant frequencies                          |
| Vowel space area (VSA)         | Yes        | From F1/F2 of corner vowels                               |
| Monopitch                      | Yes        | Low variance of f0 over time                              |
| Monoloudness                   | Yes        | Low variance of intensity over time                       |


Where a strong white-box source-filter model helps most:

Mechanistic, interpretable biomarkers
- Glottal source: f0, jitter, shimmer, tremor, HNR, spectral tilt, open quotient proxies (NAQ, etc.)
- Filter: formants, vowel space geometry (VSA, VAI), spectral envelope features
  This lets you say what changed (source vs filter) rather than just "the embedding moved."

Robust evaluation and confound control
- You can test what happens when you hold constant the filter and vary the source, and vice versa.
- That is a direct way to check whether a claimed biomarker is actually disease-related or just microphone / vowel / loudness.

Within-subject tracking
- For PD, longitudinal change within a person is often the clinically useful signal.
- A white-box model gives you stable summary parameters you can trend over time.

About adding uncertainty:

Adding uncertainty estimates to speech biomarkers would not primarily create a dramatic jump in headline accuracy, but it would substantially improve the trustworthiness and transportability of results. Most current speech features such as HNR, CPP, jitter, shimmer, spectral tilt, and formants are treated as point estimates derived from ad hoc signal processing pipelines. In reality, many segments are ill-posed: breath noise, creaky phonation, clipping, low SNR, vowel mismatch, or tracking failures all corrupt measurements in ways that are rarely made explicit. If each feature is reported together with an uncertainty estimate, unreliable frames or segments can be automatically downweighted or rejected, and coverage and failure modes can be reported instead of silently biasing the analysis. This alone often improves out-of-domain performance, where recording conditions and speaking style differ from the training data.

More importantly, uncertainty enables statistical models that are aligned with the underlying physics. Current practice typically treats measured features as ground truth and feeds them into classifiers, ignoring measurement error. With uncertainty-aware features, one can use errors-in-variables regression, hierarchical models that separate subject and session effects, and proper propagation of waveform noise through the source-filter model into biomarker uncertainty. Scoring can then be likelihood-based rather than driven by heuristic thresholds. This shift directly addresses the common pattern where models achieve very high accuracy on one dataset and collapse on another, because they have implicitly overfit to dataset-specific measurement artifacts.

Finally, uncertainty makes it possible to separate measurement noise from genuine biological variability. In Parkinson’s disease, within-subject variability due to fatigue, medication state, attention, or task effects can be large relative to between-subject differences. If measurement uncertainty is explicitly modeled, it becomes feasible to decompose observed variation into measurement noise, short-term state variability, and longer-term trait-like changes related to disease. This is particularly valuable for longitudinal tracking, where the goal is not binary classification but detecting meaningful change within an individual over time.

In short, turning ad hoc point estimates into calibrated distributions moves speech-based PD research from feature hacking toward measurement plus inference. That shift is unlikely to produce sensational accuracy numbers on paper, but it is exactly what is needed for results that generalize, survive re-evaluation, and are useful beyond a single dataset.


*/



/*
applications of a reliable white-box speech model:

speech and language sciences:
phonetics: articulatory–acoustic mapping and voice quality analysis
phonology: experimental study of speech sound systems
linguistics: prosody and phonological patterning

biomedical and clinical:
speech-language pathology: assessment and therapy of voice and motor speech disorders
laryngology: diagnosis and surgical planning for vocal fold disorders
neurology: biomarkers for neurodegenerative or motor control diseases
respiratory medicine: coupling between respiration and phonation
otolaryngology: quantitative evaluation of vocal performance and hearing

forensic and security:
forensic phonetics: speaker identification and authenticity testing
biometric security: voice-based identification and verification

engineering and technology:
speech technology: synthesis, coding, and generative speech models
robotics: natural voice generation and auditory feedback

comparative biology and evolution:
bioacoustics: modeling vocal production in animals
evolutionary linguistics: comparative study of vocal systems across species

*/


/*
From @Drugman2019

Glottal characterization has also been shown to be helpful in another
biomedical problem: the classification of clinical depression in speech. In
(Ozdas et al. (2004))
*/

