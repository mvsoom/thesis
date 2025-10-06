= Glottal inverse filtering
<chapter:gif>

In this Chapter we sketch the problem in detail, highlight challenges and solutions given, and current state of the art.

Things that should be mentioned (refer to it in other Chapters):
- Crucial that the DGF model has low rolloff; or equivalently, that it can represent sharp GCI
- High F0 domain: time to break it open: infants, woman, elderly (important for Parkinson's disease application: @Ma2020)
- Distinction between parametric, nonparametric (signal processing), nonparametric (Bayesian -- this approach) solutions
- Applications: scientific, medical, forensic. Related problem, formant estimation expected to be positively influenced if this works
- Radiation modeling: do the canonical derivative operation (small tweeter). This can be sidestepped in IBIF: radiation transfer function is replaced by neck tissue transfer function: more stable for a calibration on a single person
- Importance of closure condition (null integral constraint)

Good introduction to GIF is in @Alku2002

Review: @Chien2017 @Freixes2023
Classic: @Alku2011 @Drugman2019a

Our method:
- Can handle "connected speech" automatically: not only comfortable sustained vowels
- Probabilistic: simulate for error bars on downstream metrics
- Nonparametric in the probablistic sense: still a probablistic model so controllable and queriable and interpretable, still flexibility of "signal processing" nonparametric approaches like almost all today
- Can be seen as a surrogate learner
- Drop in generalization of classic AR model, so unifies formant estimation and inverse filtering: joint inverse filtering
- Efficient
- Pitch synchronuos by design -- higher F0 possible
- Ideal for HNR biomarkers (harmonics-to-noise ratio) as used for Parkinson's @Ma2020

Recent parametric approaches: @Lin2025

New forensics (see presentation for older ones):
- Evidential value of voice quality acoustics in _forensic voice comparison_ // @Chan2023

Related to GIF but without radiation issue: IBIF (Subglottal Impedance-Based Inverse Filtering)
- Our model can be used there too (can switch derivative order)

Voice orders in US affect 7% of working population // https://pmc.ncbi.nlm.nih.gov/articles/PMC9615581/

