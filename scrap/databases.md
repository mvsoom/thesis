Below is a short “menu” of corpora that are routinely used to benchmark F0, spectral-envelope and glottal-source estimation algorithms.  I’ve split them up by the kind of “ground truth” they provide so you can decide what best matches your evaluation goals.

| Category                                             | Corpus                      | What you get (ground-truth)                                                                                                                                                                 | Size / Speakers                                                                                | Licence & How to grab it                                                                                     |
| ---------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Synthetic & physical-model speech**                | **OPENGLOT (I–III)**        | Perfect, sample-synchronous LF or physical-model glottal flow + exact vocal-tract filter coefficients + rendered speech. Ideal when you want to know *both* the “true” envelope **and** F0. | Repos I–III: thousands of synthetic / FEM / 3-D-printed utterances (vowel and sentence-length) | MIT-style licence, free download ↗︎ research.spa.aalto.fi (see paper for links) ([University of Arizona][1]) |
|                                                      | EGIFA (you already have it) | Similar idea to OPENGLOT but a bit smaller; includes mixtures of synthetic and real EGG-recorded vowels for GIF benchmarking.                                                               | 23 vowel-type tokens × 9 SNR conditions                                                        | Free for research (Aalto)                                                                                    |
| **Natural speech with laryngograph ground truth**    | **PTDB-TUG**                | Studio mic + synchronous EGG; reference F0 tracks extracted from EGG and manually validated.                                                                                                | 4720 sentences, 20 speakers (10 ♂ / 10 ♀)                                                      | ODbL, direct download (4.4 GB zip) ([spsc.tugraz.at][2])                                                     |
|                                                      | **Keele Pitch DB**          | Classic reference corpus; short stories read by 10 speakers with EGG; still used for quick sanity checks.                                                                                   | ≈35 s per speaker                                                                              | Free for non-commercial use (Zenodo mirror in “Speech & Noise Corpora…” bundle) ([Zenodo][3])                |
|                                                      | MOCHA-TIMIT / FDA           | Mic + EGG (+ articulatory sensors) and hand-checked F0; handy if you also care about co-articulation.                                                                                       | 460 sentences / 2 speakers (MOCHA-TIMIT) ; 50 sentences / 1 speaker (FDA)                      | Same Zenodo bundle as above ([Zenodo][3])                                                                    |
| **Speech with validated formant / VTR trajectories** | **VTR Formants DB**         | Manual formant tracks (F1–F3) on 538 TIMIT sentences → use as “envelope” ground truth.                                                                                                      | 16 kHz audio + frame-level VTR                                                                 | UCLA licence (free for research) ([ee.ucla.edu][4])                                                          |
| **Singing voice (F0 & note-level labels)**           | **Annotated-VocalSet**      | Frame-wise F0 contours, note on/offsets, lyrics; 20 pro singers, diverse styles & techniques.                                                                                               | 2688 clips (≈6 h)                                                                              | CC-BY 4.0, Zenodo DOI 10.5281/zenodo.7061507 ([Zenodo][5])                                                   |
|                                                      | **MIR-1K**                  | Chinese pop karaoke clips; manual F0 contour + vocal/non-vocal masks.                                                                                                                       | 1000 clips, 17 singers                                                                         | Free research licence, NCTU mirror ([Zenodo][6])                                                             |

---

### Where your *local* folders fit

* **TIMIT / TIMIT-voiced** – great phonetic coverage but no gold-standard F0; the “voiced” subset just marks voiced frames.
* **VTRFormants** – you already have the formant ground truth; pair it with F0 from PTDB-TUG for joint tests.
* **APLAWDW, holmberg, klatt** – mostly vowel tokens with high-quality inverse-filtered flow; useful for controlled GIF tests but limited linguistic variety.

### A quick evaluation recipe

1. **Unit tests on synthetic data** – start with OPENGLOT I (pure LF source) to check that your JAX implementation reproduces envelopes/F0 exactly.
2. **Natural speech, controlled recording** – run on PTDB-TUG; compare your F0 track against the EGG reference (e.g., raw Hz RMSE + voicing error) and envelope against VTR if you merge with its sentences.
3. **Robustness** – add noise or reverberation (the PTDB-TUG site provides matched NOISEX/QUT mixes) to stress-test.
4. **Singing-voice stress test** – VocalSet or MIR-1K will reveal how the model copes with large pitch ranges & vibrato.

That gives you a ladder from “perfectly known” to “wild in the real world” without needing new proprietary corpora.  Good luck with the benchmarks—let me know if you need code snippets for alignment or metrics!

[1]: https://experts.arizona.edu/en/publications/openglot-an-open-environment-for-the-evaluation-of-glottal-invers "
        OPENGLOT – An open environment for the evaluation of glottal inverse filtering
      \-  University of Arizona"
[2]: https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html?utm_source=chatgpt.com "PTDB-TUG: Pitch Tracking Database from Graz University of ..."
[3]: https://zenodo.org/records/3920591 "Speech and Noise Corpora for Pitch Estimation of Human Speech"
[4]: https://www.ee.ucla.edu/~spapl/VTRFormants.html?utm_source=chatgpt.com "UCLA: Speech Processing & Auditory Perception Laboratory"
[5]: https://zenodo.org/records/7061507?utm_source=chatgpt.com "Annotated-VocalSet: A Singing Voice Dataset - Zenodo"
[6]: https://zenodo.org/records/3532216?utm_source=chatgpt.com "MIR-1K dataset - Zenodo"
