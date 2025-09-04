# OPENGLOT-I
Noise part O(1e-1) picks up everything and is then shaped moderately by spectrum
* nu_e/nu_w = O(20)! => low pitchedness
* Inferred very strongly -- no sample variation

Signal part O(1e-3) does not look periodic, but lengthscales are approx correct
* Large sample variation

I checked and this is not due to SVD noise floor -- same result for -60 dB and -100 dB

For now it seems like the ExpSqPeriodic kernel is just a really bad model for OPENGLOT I
* ***Let's check the prior model -- how does it look? Periodic at least?***
* Model fails completely at estimating F0 but the inferred noise component does have the correct periodicity
* Things might improve for REAL speech (less harsh)
* F3 and F4 are not terribly off
* The model manages to put weight on F0 ~ 100 Hz

I tried a [Matern12 model](code_matern.py) with the same results -- except that the signal component now indeed looks like Matern12 draws
* Inferred spectrum is also very similar

Conclusion: **for OPENGLOT I our source model is way too smooth** and we can **check final pitchedness to see if a run failed or not**

# OPENGLOT-II

Another run on OPENGLOT-II with a handpicked sample that had very high final pitchedness
* Here we see that the inferred source signal is periodic and very smooth, and at the right F0
* However the inferred spectrum is completely off