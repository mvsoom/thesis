# `relaxation` experiment

We generate 5 LF waveforms with R_d values in [0.5; 2.7] and see which kernel has best evidence

This is a followup from the `gfm` experiment because the "discretization" of the LF examplars according to `d` was found to influence the fitting process too much; though the results were positive, it is more interesting to directly fit the LF waveforms
Additional changes made here wrt. `gfm`:

- Automatically finding `t_c`: the point where total flow exceeds (1-1e-5)

  - All waveforms regardless of the value of R_d are restretched to span 256 samples of an open phase of [0,t_c) where t_c = 6.0 msec
  - This is necessary because low Rd values define an "effective open phase"; Fant is not consistent in using the "T" param as a period or open phase. Basically he set T := t_c for this reason.

- The open phase is endpoint exclusive: [0,t_c) which should give `periodickernel` a better shot at fitting the data

- There is no hard GCI as in an effective jump -- all of the waveforms have a sharp "turning point"

## More about Rd

The recommended @Fant1995 Rd range [0.3; 2.7] ranges spans from pressed phonation (harder GCI, longer closed phase; ie tight adducted phonation) to lax phonation (softer GCI, longer open phase; ie breathy abducted phonation).
In this context, Rd is also called the "relaxation coefficient", hence the experiment name.
This range is also shown in a Figure in @Degottex2010, p. 38

Rd is also proportional to NAQ:

> The NAQ measure (Alku et al., 2002) has been proposed as a global parameter, which correlates with the tense/lax dimension of vocal quality and, when scaled by 0.11, is essentially the same as the Rd parameter used in this study. A high NAQ value is indicative of lax voice, and a low NAQ value indicative of pressed or tense voice. NAQ has gained considerable popularity as a measure of the tense/lax dimension of voice variation. @Yanushevskaya2022

# Nested sampling

- nlive = 500
- using 'rwalk' as 'uniform' method could sometimes get stick in a bootstrapping problem loop ("The enlargement factor for the ellipsoidal bounds determined from bootstrapping is very large.")
  * in simple tests this outputted same logz as 'uniform' sampling, just in mins rather than hrs

# Observations

[ongoing]
