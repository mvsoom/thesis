---
title: "Quantization of Gaussian processes"
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

> *I know the human being and fish can coexist peacefully!*
>
> -- George W. Bush

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

Given d, find (u, h).

This a blind identification problem:

- Ill-posed
- Needs regularization

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "∞ many" -> one [label="to"];
}
EOF
```

<!-- pause -->

![](assets/blind_demo_smooth.gif)


Ill-posed problems
==================
<!-- speaker_note:

i am workign on "speech inversion"
this is a blind identification problem problem (or whatever its called): many to one
=> need prior information to constrain the set of mathematically possible solutions: they must adhere to what we already know to be plausible or true

[now this is a crowd of engineers so they think in costs, so ill talk about assigning initial costs and that bayesian inference is a way to correctly work with these costs: bayesian inference is ... bookkeeping]
-->

<!-- column_layout: [1,1] -->

<!-- column: 0 -->

# Ill-posed inverse problems are everywhere

- Many-to-one mappings
- Repeated measurements
- Any interpolation problem!
- Any extrapolation problem!

The data admit many solutions.

=> Which solution(s) do we want?

<!-- newlines: 1 -->

# Simplest ill-posed problem

```
y = a₁ + a₂
```

Suppose we observe: y = 1.

=> What is the solution space?

<!-- column: 1 -->

<!-- pause -->

```bash +image
gnuplot <<'GP'
set term pngcairo size 560,560 background rgb "black" enhanced

set size square
unset key

set border lc rgb "#4c566a"
set tics textcolor rgb "#4c566a"
set xlabel "a_1" textcolor rgb "#d8dee9"
set ylabel "a_2" textcolor rgb "#d8dee9"
set zeroaxis lc rgb "#434c5e"

set label "a_1 + a_2 = 1" at -1.3,1.2 tc rgb "#88c0d0"
# set label "ridge cost" at 0.7,0.6 tc rgb "#5e81ac"


set xrange [-1.5:1.5]
set yrange [-1.5:1.5]

set parametric
set trange [-2*pi:2*pi]

# data constraint: a1 + a2 = 1
x_line(t) = t
y_line(t) = 1 - t

# ridge circle radii
r1 = 0.2
r2 = 0.5
r3 = 1.0

# ridge solution (closest point to origin)
a_star = 0.5

plot \
    x_line(t), y_line(t) w l lw 4 lc rgb "#88c0d0", \
    r1*cos(t), r1*sin(t) w l lc rgb "#5e81ac" dt 2, \
    r2*cos(t), r2*sin(t) w l lc rgb "#5e81ac" dt 2, \
    r3*cos(t), r3*sin(t) w l lc rgb "#5e81ac" dt 2, \
    a_star, a_star w p pt 7 ps 2 lc rgb "#bf616a"

unset parametric
GP
```

Ridge regression
=================
<!-- speaker_note:

Ridge regression was introduced to stabilize least squares under near-singularity.

A practical, engineered fix for instability in least squares caused by near collinearity.

Collinearity

In linear regression collinearity means some columns of are nearly linear combinations of others.
=> many different coefficients give almost the same fitted values

ill-conditioned matrix Phi^T Phi
When it happens

Not exotic at all:
    Redundant measurements: same physical quantity measured in slightly different ways.

    Polynomial features x, x², x³ on a narrow domain.

    High dimension: more basisfunctions than n

-->

<!-- column_layout: [1,1] -->
<!-- column: 0 -->

# Linear regression

Given m basis functions Φ = [ϕ₁ | ϕ₂ | ... | ϕₘ], model a function f(x):
```
f(x) = a₁ϕ₁(x) + a₂ϕ₂(x) + ... + aₘϕₘ(x)
```
and fit the coefficients a to data y by least squares:
```
â = argminₐ ‖Φa − y‖²
  = (ΦᵀΦ)⁻¹Φᵀy
```
=> This crashes numerically for ill-posed problems

<!-- column: 1 -->
<!-- pause -->

# Ridge regression

Introduce a minimal engineering fix:
```
â = argminₐ ‖Φa − y‖² + λ‖a‖²
  = (ΦᵀΦ + λI)⁻¹ Φᵀy
```
This stabilizes inversion by making large coefficients costly.
Eigenvalues shifted by +λ.

<!-- newlines: 1 -->
<!-- pause -->

## Anisotropic ridge regression

Assign different costs to different coefficients:
```
â = argminₐ ‖Φa − y‖² + λ aᵀΣ⁻¹a
  = (ΦᵀΦ + λΣ⁻¹)⁻¹ Φᵀy
```

<!-- pause -->

```bash +exec +acquire_terminal
/// python live/ridge.py
```

Gaussian processes
==================

<!-- speaker_note

bam, insight: use GPs and quantify them to BLR (regression models)
this ives you Phi (basis functions) AND the covariance matrix (the costs) in one go
and we did it by just suplying HIGH LEVEL INFORMATION!
=> let the model work out the algebraic consequences.
so this really does what we wanti th



Ridge regression is what you do when you know you need regularization but don’t know what kind.

Gaussian processes are what you use when you do know the kind.

-->



```bash +exec +acquire_terminal
python live/gp.py
```

<!-- pause -->

Choosing Σ looks a lot like Gaussian process regression!

## Good

Gaussian processes are ill-posed problem killers.

They specify **high-level structure** in function space:

* lengthscale (smoothness)
* differentiability
* periodicity
* relevance of input dimensions

They compose: k1 * k2, etc. Kernel algebra.

No need to choose Φ or Σ by hand!

## Bad

O(N³) inference cost and sometimes black-box.

Ridge regression <=> Gaussian processes
=======================================

Well known that linear models are GPs

But GPs can also be transformed into BLRs!

You can go _either_ way

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Gaussian process" -> "ridge" [label="quantize!"];
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

Quantization [1]
================

<!-- column_layout: [1,1] -->

<!-- column: 0 -->

<!-- newlines: 2 -->

# Hilbert-GP

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Stationary GP" -> "ridge" [label="Hilbert!"];
}
EOF
```

<!-- reset_layout -->

best method for this currently in low dim is Hilbert-GP

which is basically quadrature of bochners theorem

fixed basis functions , but we get the diagonal weights, which is insae

here we should do another interactive demo, perhaps in 2D now?

we should show the spectrum at least

```typst +render +no_background +width:20%
$Phi = sin(x)$

$lambda_k = S(omega_k)$
```

Quantization [2]
================

<!-- speaker_note:

Learning camera calibration on a test set of 30k+ samples takes 1 minute, with a clean learning curve

-->

<!-- column_layout: [1,1] -->

<!-- column: 0 -->

<!-- newlines: 2 -->

# Spherical harmonics

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Stationary/NN GP" -> "ridge" [label="Spherical!"];
}
EOF
```

<!-- reset_layout -->

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

```bash +image
convert -density 300 assets/embedding.png -background none  -channel RGB -negate png:-
```

[Dutordoir+ 2020]

<!-- column: 1 -->


```bash +image
convert -density 300 assets/spherical_harmonics.png -background none  -channel RGB -negate png:-
```

Quantization [3]
================

<!-- speaker_note:

Learning camera calibration on a test set of 30k+ samples takes 1 minute, with a clean learning curve

-->

<!-- column_layout: [1,1] -->

<!-- column: 0 -->

<!-- newlines: 2 -->

# Spherical harmonics

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Stationary/NN GP" -> "ridge" [label="Spherical!"];
}
EOF
```

<!-- reset_layout -->

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

```bash +image
convert -density 300 assets/uvx_learning_curve.png -background none  -channel RGB -negate png:-
```

<!-- column: 1 -->

```bash +image
convert -density 300 assets/uvx.png -background none  -channel RGB -negate png:-
```

Learning from examples: levels 1 & 2
====================================

Another thing that ridge regression analogy makes clear:

- Level 2 learning = GP hyperparam learning = learning the covariance matrix
- Level 1 learning = learning the coefficients

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
- Enjoy better inference when lucky

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    "Gaussian process" -> "Ridge regression" [label="quantize!"];
    "Ridge regression" -> "Gaussian process" [label="special\ncase of"];
}
EOF
```

<!-- reset_layout -->
<!-- newlines: 1 -->
<!-- column_layout: [1, 1] -->
<!-- column: 0 -->

## Advantages:

- O(N) inference, not O(N³)
- Level 1 & 2 learning
- Principled cost control
- Any likelihood

<!-- column: 1 -->

## Disadvantages:

- Still an approximation
- Requires lower input dimensions: O(10)
  * Though variations exist that go up to O(200)

References
==========

<!-- speaker_note:

Important application of doing BLR in higher dimensions is BO

-->

<!-- column_layout: [2,2,1] -->

<!-- column: 0 -->

# References [1]
Hilbert GP:

1. Solin, A. & Särkkä, S. **Hilbert space methods for reduced-rank Gaussian process regression**. Stat Comput 30, 419–446 (2020)
2. Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A. & Vehtari, A. **Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming**. arXiv:2004.11408

Spherical harmonics:
1. Dutordoir, V., Durrande, N. & Hensman, J. **Sparse Gaussian Processes with Spherical Harmonic Features.** in Proceedings of the 37th International Conference on Machine Learning 2793–2802 (PMLR, 2020).

<!-- column: 1 -->

# References [2]
Ways to higher input dimensions:

1. Rahimi, A. & Recht, B. **Random Features for Large-Scale Kernel Machines**. in Advances in Neural Information Processing Systems (2007)

2. Eleftheriadis, S., Richards, D. & Hensman, J. **Sparse Gaussian Processes with Spherical Harmonic Features Revisited**. arXiv:2303.15948

3. Mutny, M. & Krause, A. **Efficient High Dimensional Bayesian Optimization with Additivity and Quadrature Fourier Features**. in Advances in Neural Information Processing Systems (2018)

<!-- column: 2 -->

# Any questions or contact for further discussing or colab:
> marnix@ai.vub.ac.be