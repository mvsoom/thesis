= LBGID
<algo:lbgid>

// see notebooks/egifa/explore.py for plots
// and svi/aplawd/* for plots of how peaks are ignored

LBGID stands for level-based glottal instant detection.

Within the BNGIF framework, glottal closure and opening instants serve primarily as _alignment anchors_ for the closed-phase analysis used downstream in IKLP.
The task is therefore not to reconstruct vocal fold biomechanics or to match a stylized glottal model, but to extract consistent temporal landmarks from the airflow signals $u(t)$ produced by physics-based simulators such as VocalTractLab.

This changes the detection problem in an important way.
In a synthesizer, glottal closure and opening are not predefined symbolic events; they are emergent consequences of biomechanical dynamics, and the closed phase appears not as a hard discontinuity but as a soft, context-dependent region of suppressed airflow embedded in an otherwise continuous signal.
Traditional glottal instant detectors assume quasi-periodicity, predefined waveform shapes, or spectral heuristics derived from acoustic signals.
None of these assumptions transfer cleanly to synthetic airflow streams.

The guiding intuition behind LBGID is deliberately simple: closure corresponds to _low airflow_, and glottal instants correspond to strong opposing changes in airflow dynamics flanking those low-flow regions.
The algorithm therefore first identifies where the flow is small and then extracts the most energetic dynamic transitions within those regions.

== Adaptive envelope normalization

A central difficulty with airflow signals is that absolute amplitude is rarely interpretable on its own.
Slow baseline drift, variation in subglottal pressure, and simulator-specific scaling can all alter amplitude without changing the underlying temporal structure.
Rather than attempting to remove these effects explicitly, LBGID constructs a local coordinate system by tracking adaptive upper and lower envelopes, obtained by detecting peaks and troughs and interpolating between them.
These define a smoothly varying local amplitude range,
$
  A(t) = "roof"(t) - "floor"(t),
$
and all subsequent decisions are made relative to it.
The signal is interpreted relationally: what matters is not the absolute airflow value at any point, but where it sits within its current local dynamic range.

== Level-based closed phase detection

With the local amplitude frame in hand, candidate closed phases are identified via a relative threshold,
$
  "level"(t) = "floor"(t) + alpha ("roof"(t) - "floor"(t)),
$
for a small constant $alpha$, and points satisfying $u(t) <= "level"(t)$ are treated as belonging to a candidate closed phase.
This yields contiguous low-flow segments that naturally partition the signal into bounded candidate intervals.
The deliberate choice to isolate the closure regime first — before looking for any dynamics — is what distinguishes LBGID from direct peak or derivative detectors: the search for instants is local and physically constrained from the outset.

== Dynamic pairing inside the closed phase

Within each low-flow segment, LBGID searches for a pair of opposing dynamic events.
The derivative $u'(t)$ is computed by finite differences.
Rather than identifying a single extremum, the algorithm evaluates all pairs of candidate points $(i, j)$ within the segment by the signed product
$
  E(i, j) = u'(i) thin u'(j).
$ <eq:lbgid-energy>
Pairs with $E(i, j) < 0$ correspond to opposite-sign slopes, i.e., one point is on an ascending transition and the other on a descending one.
Among these, the pair maximizing $|E(i, j)|$ is selected as the most energetically opposed dynamic interaction within the closed phase, and its two members are reported as the glottal closure instant (GCI) and glottal opening instant (GOI).

No assumption is made about the waveform morphology between them.

== Closing remarks

LBGID performs three transformations in sequence: convert absolute airflow into a local amplitude frame via envelope tracking, identify the physical regime where closure must occur via the level threshold, and detect the onset and offset of that closure as the most energetically opposed derivative pair within the candidate segment.
The result is a pair of instants that emerge from the airflow's own amplitude and dynamic structure rather than from any externally imposed waveform model.

This makes LBGID well-suited to physics-based simulation environments, where the closed phase is gradual and the absence of a sharp GCI spike would cause direct peak detectors to fail.
In the BNGIF context it provides a stable alignment mechanism that is robust to the variation in phonatory effort, voice quality, and fundamental frequency present in the VocalTractLab data.
