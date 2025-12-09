---
title: "[TODO] Title"
author: Marnix Van Soom [VUB]
event: GP lunch seminar 111225

theme:
  name: terminal-dark
---


Speech inversion [1]
===================

<!-- speaker_note:
Our brains are really good at "inverting speech"
What I mean is: we reconstruct the meaning of what someone said from a pressure waveform hitting our ears 

Note that there is a many-to-one phenomenon here:
Many utterances, or speech waveforms, all map to the same meaning
Our brains filter out room reverb and delay automatically

We can show this very crudely by throwing random math functions on speech waveforms and playing them out loud
-->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    many -> one [label="to"];
}
EOF
```

<!-- pause -->

<!-- new_lines: 1 -->
<!-- column_layout: [1, 1] -->

<!-- column: 1 -->
<!-- alignment: center -->

> I know the human being and fish can coexist peacefully!

<!-- pause -->

<!-- column: 0 -->

```python
x = load("bush.wav")
```

```python +exec
/// from play import *
/// from numpy import exp, log, abs
play(x)

play(x**2)

play(exp(1 + 5*x - x**2))
```

Speech inversion [2]
===================

<!-- speaker_note
There is another level of many-to-one mapping.

Given a waveform, there are infinitely many pairs of excitation and filter that produce exactly the same signal.

This is a much harder ambiguity.
-->

<!-- column_layout: [1, 3] -->

<!-- column: 0 -->

Problem:

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    "u * h = d"
}
EOF
```

given d, find (u, h)

# a blind inversion problem

# Ill-posed

## Needs regularization

<!-- pause -->

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "∞ many" -> one [label="to"];
}
EOF
```

![](assets/blind_demo_smooth.gif)


Ill-posed problems
==================
<!-- speaker_note:

i am workign on "speech inversion"
this is a blind deconvolution problem (or whatever its called): many to one
=> need prior information to constrain to whatevers plausibe

[now this is a crowd of engineers so they think in costs, so ill talk about assigning initial costs and that bayesian inference is a way to correctly work with these costs: bayesian inference is ... bookkeeping]
-->

then: zoom out

many problems are of this kind: regression etc => we got many good off the shelf models we can use (matlab algs, python libs), but how do we "get them where we want them": be general enough but kinda do our bidding, kinda "know" what we want?

this is a hard question and very general

we can start by thinking about science's workhorse: _linear regression_



Linear regression
=================
<!-- speaker_note:
-->

so, linear regression (meaning basis functions etc)

O(N) inference: fastest you can get

```typst +render +width:40%
$ "argmin"_a ||y - f(x; a)||^2 + lambda ||a||^2 $
```

ridge regression solved a lot of stuff

and has a strong bearing on interpolation problems [WE MAKE ANOTHER INTERACTIVE DEMO HERE: got toy data and see what ridge regression does based on weights; can be ordinary LR or with basis functions, dont know whats best; try also random covariance matrices to see impact on fit]

but it is arbitraruy: which weights/alpha/ridge do we set?

why not a general covariance matrix?

how do we pick it?

also: how do pick the basis functions? they aobviously matter a lot

```bash +exec +acquire_terminal
python liveridge.py
```

> How do we choose Phi and lambda?

Gaussian processes
==================

<!-- speaker_note

bam, insight: use GPs and quantify them to BLR (regression models)
this ives you Phi (basis functions) AND the covariance matrix (the costs) in one go
and we did it by just suplying HIGH LEVEL INFORMATION!
=> let the model work out the algebraic consequences.
so this really does what we wanti th

-->


```bash +exec +acquire_terminal
python livegp.py
```



Interesting fact about GPs:

High level information made quantitative
- Lengthscale
- Differentiability class
- (Quasi)-periodicity
- Which covariates have a bearing on the target function

They do exactly what we like, but are O(N³) and esoteric

No need to pick Phi and lambda

Linear regression <=> Gaussian processes
========================================

Well known that linear models are GPs

But GPs can also be transformed into BLRs!

You can go _either_ way

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Gaussian process" -> "BLR" [label="quantize"];
}
EOF
```

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

extra benfits:
- very easy representation for any stationary kernel
- O(N) inference, not N^3, generally reasonable approximation
- reveals the structure of the GP because we can rotate Phi x Sigma^(1/2) which gives us equally weighted basis functions, a bit like PCA
- all sparse VIs are compatible/based on this, so we get arbitrary likelihoods, batching, etc
- (any ones you can think of here?)

<!-- column: 1 -->

downsides:
- still an approx, which might break down
- might require many basisfunctions in higher dims : O(10) max, tho many clever ways try to go around this [MCMC : fourier features, reorthogonalizing: Eleftheriadis2023, ...]
- nonstationary kernels are generally hard to get BLR repr from
- (any ones you can think of here?)

Hilbert-GP
==========

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Stationary GP" -> "BLR" [label="Hilbert"];
}
EOF
```

best method for this currently in low dim is Hilbert-GP

which is basically quadrature of bochners theorem

fixed basis functions , but we get the diagonal weights, which is insae

here we should do another interactive demo, perhaps in 2D now?

we should show the spectrum at least

```typst +render +no_background +width:20%
$Phi = sin(x)$

$lambda_k = S(omega_k)$
```

[TODO: cite paper]

Learning from examples: levels 1 & 2
====================================

bonus: this allows "level 1" learning

say you have examplars of what your prior should try to emulate from a simulation

with ordinary GP: optimize hyperparams and you get something more like it: level 2 learned

but usually not quite refined enough if your examplars have definite shape

and here we go back to my use case

say we have a glottal flow model that can generate samples

if i quantize my prior, i can learn the shape from it both level 1 (amplitude level) and level 2 (high level)

this results in another BLR model where amps have defintie prior mean and covariance

here we show level 1 and level 2 adaptation for whitenoise kernel, periodickernel and my spack kernel (custom derived) which should be a plot containing: col 1: prior, no conditioning, col 2: prior, level 1 learnt, col 3: prior, level 1 + level 2 learnt and rows are the kernels. each cell contains samples from the prior and pressing a key adds another examplar to the learning examplar pool

Results
=======

OpenGLOTI benchmark and graphs for the three kernels

then i show benchmark results for whitenoise, periodic kernel and spack and show that examplar learning really does pay off

Summary
=======

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

# Takeaway:

- Many problems need regularization, or "cost control"
- Use GPs to specify high-level information about costs
- Convert them to quick linear regression models which thusly acquire sophistication
- Set hyperparameters, or fit to examples
  * Level 1
  * Level 1 & 2
- Enjoy better inference

## Advantages:

- O(N) inference, not O(N^3)
- Principled cost control
- Any likelihood

## Disadvantages:

- Still an approximation
- Lower dimensions: O(10)
  * Though clever tricks exist

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    "Gaussian process" -> "Linear regression" [label="quantize!"];
    
    "Examples" -> "Linear regression" [label="level 1 and 2"]
    "Examples" -> "Gaussian process" [label="level 2 only"]
}
EOF
```

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    "Stationary GP" -> "Linear regression" [label="Hilbert!"]
}
EOF
```

References
==========

Ways to get to higher dimensions:
- MCMC, eg. Fourier Features
- Eleft. 2023


Hilbert GP:
- og paper
- that other paper

