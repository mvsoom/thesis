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

Top three log Z scores per Rd and kernels:
```
       Rd    kernel centered normalized      logz   logzerr information
    <ord>    <fctr>   <lgcl>     <lgcl>     <num>     <num>       <num>
 1:   0.3 matern:32    FALSE      FALSE  313.7534 0.1118094    8.896086
 2:   0.3 matern:52    FALSE      FALSE  298.7225 0.1160840    9.652219
 3:   0.3    tack:1     TRUE      FALSE  277.1214 0.1379075   14.049976
 4:   0.9    tack:1     TRUE       TRUE  748.8229 0.1522010   17.403057
 5:   0.9    tack:1     TRUE      FALSE  727.4155 0.1404274   14.759115
 6:   0.9 matern:32    FALSE      FALSE  696.2593 0.1235674   11.163743
 7:   1.5 matern:32    FALSE      FALSE  872.3859 0.1271889   11.844000
 8:   1.5    tack:1     TRUE       TRUE  866.8301 0.1482321   16.375094
 9:   1.5    tack:1     TRUE      FALSE  866.6617 0.1370326   13.971396
10:   2.1 matern:32    FALSE      FALSE 1030.3269 0.1313533   12.771993
11:   2.1    tack:1     TRUE      FALSE 1021.5357 0.1400237   14.556715
12:   2.1    tack:1     TRUE       TRUE 1017.6724 0.1534838   17.669475
13:   2.7 matern:32    FALSE      FALSE 1196.4620 0.1399400   14.553160
14:   2.7 matern:52    FALSE      FALSE 1193.9682 0.1443756   15.434895
15:   2.7    tack:1     TRUE      FALSE 1191.6537 0.1450761   15.710742
```

I want to try again with `centered` meaning: actually infer the center.
It might really matter and explain subpar tack:1 performance.
==> experiments/relaxation_centered

## `errored` runs

Due to "kernel died": nested sampling inference remains stuck, kernel unresponsive and is then killed after (?) one hour.

Only happens with some tack:2 kernels and more tack:3 kernels; in total ~10 out of 950, so OK.