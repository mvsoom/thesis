Contact information:
Yu-Ren Chien <yrchien@ntu.edu.tw>
Jon Gudnason <jg@ru.is>

EGIFA performs evaluation of glottal inverse filtering algorithms.  For detailed description, see:
Y.-R. Chien, D. D. Mehta, J. Gudnason, M. Zanartu, and T. F. Quatieri, "Performance evaluation of glottal inverse filtering algorithms using a physiologically based articulatory speech synthesizer," IEEE/ACM Transactions on Audio, Speech, and Language Processing, submitted.

The evaluation requires Matlab R2016b, and a data set generated with VocalTractLab.  Two readily applicable data sets are available also from the authors.

To test an inverse filtering algorithm on a data set, use test_vtl3.m.  Type 'help test_vtl3' in Matlab to get help.  To measure the performance, use score_vtl12.m for sustained vowels, and score_vtl_continuous1.m for continuous speech.

# Own notes
Matlab 2025b with {Signal Processing, System Identification, Wavelet} Toolbox

TL;DR
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


## Bridges
data/EGIFA/python_matlab_test.py
data/EGIFA/matlab_python_test.m

We have to bring our algo to this testing env, whose score metric is flawed
and bring their algos to our testing env, which has a better scoring metric

## Data
Data: both vowel/ (~0.6 s) and speech/ (~3 s) are monotone/fixed-F0 VTL synths; speech is a sentence with flat pitch, not prosodic speech.
All signals at 44.1 kHz

/vowel
consists of pairs of (.wav, .mat) files where
- .wav: mono speech, single channel; length ~ 0.6 sec
- .mat: contains `glottal_flow`, `lower_area`, `upper_area` timeseries

/speech
- .wav: same as in vowel; length ~ 3 sec
- .mat: same as above

### GF vs DGF

Input .mats: contain GF
Output .mats: contain DGF (saved as `uu`)
Scoring: differences the ground truth GF before comparing with inferred GFs

## GIF methods via test_vtl3.m

GIF methods & flags (for test_vtl3.m, which calls inverse_filter9.m)

- Closed-phase LPC (default): no flag (uses projParam('cp')).
- Weighted covariance 1 (rgauss): --wca1.
- Weighted covariance 2 (ame): --wca2.
- Iterative Adaptive Inverse Filtering (IAIF): --iaif.
- Complex Cepstrum Decomposition (CCD): --ccd.
- Null model: --whitenoise (outputs white noise with length and scale matched to input signal).
- Oracle: --oracle (outputs ground-truth derivative flow, resampled and scaled to match processing).
- Oracle (delayed): --oracle-delayed (same as oracle, but prepends 13 zeros so, after the scorer’s fixed drop of the first 14 samples, it aligns perfectly and scores ~0).

All of these rely on GCIs EXCEPT --iaif.
- For --ccd: this is pitch-synchronuous, so needs good analysis windows, and this is choosen from GCI events 

These GCIs are estimated with DYPSA.
Supply --sg for ground truth GCIs via the input .mat files with optional offset to check error degradation.

To run:
```matlab
% cd .
mkdir('vowel/res_iaif');
test_vtl3('--data','vowel/', '--res','vowel/res_iaif/', '--iaif');
% running time: 3 minutes
```

## Scoring

Note trailing backslash at end: 'vowel/res_iaif/'. REQUIRED

/vowel
Just operates on a /res_*/ folder which has to be a child folder of vowel/

```matlab
score_vtl12('vowel/res_iaif/');
Number of samples: 750
Average normalized median absolute error: 0.32116
(standard deviation: 0.20104)
Average normalized median error: -0.0081173
(standard deviation: 0.069188)
% running time: <1 min
% output:
% Number of samples: 750
% Average normalized median absolute error: 0.32116
% (standard deviation: 0.20104)
% Average normalized median error: -0.0081173
% (standard deviation: 0.069188)
```

/speech
Just operates on a /res_*/ folder which has to be a child folder of speech/

```matlab
% example
score_vtl_continuous1('speech/res/');

% different metric
score_vtl_continuous1('/home/marnix/pro/science/thesis/data/EGIFA/speech/res_iaif/', '--naq'); 
% output
% Number of samples: 125
% Average median absolute error: 0.032604
% (standard deviation: 0.014665)
% Average median error: 0.02359
(standard deviation: 0.023921)

% running time: <5 sec
```

Optional scoring flag: `--best_affine` applies a best affine + limited lag alignment (lag range capped by inferred F0) on the middle segment before computing the default errors.

## Frame sizes and other hyperparams

- Common fs: inputs resampled to 20 kHz in `inverse_filter9`; ground truth (flow/area) resampled from 44.1 kHz to 20 kHz when used (`--sg`, scoring). Outputs `uu` at 20 kHz.

- DYPSA (GCIs/GOIs): runs at 20 kHz; Voicebox defaults (LPC frame 20 ms, group-delay smoothing ~0.45 ms, cross-corr window ~200 samples ~10 ms).

- CP/WCA (`default`, `--wca1`, `--wca2` → `weightedlpc3`): frame/window 32 ms, hop 16 ms; LPC order ≈ fs/1000 (20th order at 20 kHz); preemphasis 10 Hz; closed-phase weighting from GCIs/GOIs (cp/wca1/wca2 differ only in weighting parameters).

- IAIF (`--iaif`): `lpcauto` frames 32 ms with 16 ms hop; default LPC orders p=10, g=4, r=10; high-pass at 60 Hz (FIR 1024 taps) unless `h=0` when calling `iaif` directly.

- CCD (`--ccd`): SRH pitch tracker resamples to 16 kHz (25 ms frame, 5 ms hop for LPC residual; 100 ms SRH window, 10 ms hop). GCIs from `--sg` if provided; otherwise SEDREAMS with mean f0. CCD estimation runs at 20 kHz input from `inverse_filter9`.

## Null model

--whitenoise switch:
```
test_vtl3('--data','vowel/', ...
          '--res','vowel/res_whitenoise/', ...
          '--whitenoise');
```

/vowels
```
>> score_vtl12('/home/marnix/pro/science/thesis/data/EGIFA/vowel/res_whitenoise/');
Number of samples: 750
Average normalized median absolute error: 0.48321
(standard deviation: 0.27745)
Average normalized median error: -0.030572
(standard deviation: 0.081573)
```

/speech
```
>> score_vtl_continuous1('/home/marnix/pro/science/thesis/data/EGIFA/speech/res_whitenoise/');
Number of samples: 125
Average normalized median absolute error: 0.58323
(standard deviation: 0.2836)
Average normalized median error: -0.051649
(standard deviation: 0.083671)
```
