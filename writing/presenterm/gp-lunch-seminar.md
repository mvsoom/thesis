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
=> many different amplitudes give almost the same fitted values

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
and fit the amplitudes a to data y by least squares:
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
This stabilizes inversion by making large amplitudes costly.
Eigenvalues shifted by +λ.

<!-- newlines: 1 -->
<!-- pause -->

## Anisotropic ridge regression

Assign different costs to different amplitudes:
```
â = argminₐ ‖Φa − y‖² + λ aᵀΣ⁻¹a
  = (ΦᵀΦ + λΣ⁻¹)⁻¹ Φᵀy
```

<!-- pause -->

```bash +exec +acquire_terminal
python live/ridge.py
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

***Choosing Σ looks a lot like Gaussian process regression!***

Notice: we also moved in the nullspace, but we didn't do it by assigning costs.

We did it in terms of a high-level "lengthscale" concept, not on the individual coefficient level.

No need to choose Φ or Σ by hand!


-->

<!-- column_layout: [1, 1] -->
<!-- column: 0 -->

```bash +exec +acquire_terminal
python live/gaussian.py
```

<!-- column: 1 -->
<!-- pause -->


Gaussian process | ridge regression
--|--
condition on data | move along the nullspace
change hyperparameters | change costs Σ

<!-- newlines: 3 -->
<!-- column_layout: [1, 1] -->
<!-- column: 0 -->
<!-- pause -->

# Good

Gaussian processes are naturally born ill-posed problem killers because hyperparameter optimization regularizes automatically.
No need to choose Σ!

Instead, costs are specified at a high functional level, not at the amplitude level, and determine model properties such as:

* lengthscale (smoothness)
* differentiability
* periodicity
* relevance of input dimensions

They compose in a kernel algebra: k₁ * k₂ + k₃, etc.

<!-- column: 1 -->
<!-- pause -->

# Bad

- O(N³) inference cost
- Inference quality depends on kernel...
- ... but this can feel like black-box magic at times

Ridge regression <=> Gaussian processes
=======================================

<!-- speaker_note:

It is well known that linear models are GPs

But GPs can also be transformed into ridge regression models!

You can go _either_ way

-->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Gaussian process" -> "ridge" [label="quantize!"];
}
EOF
```

<!-- newlines: 2 -->
<!-- column_layout: [1, 1] -->
<!-- column: 0 -->
<!-- pause -->

# Good
- O(N) inference, not O(N³)
- Works really well in practice
- Almost all kernels used in daily life are easily quantifiable
- Plugs into VI
- Turns it into a white-box
  - Level 1 and level 2 learning

<!-- column: 1 -->
<!-- pause -->

# Bad
- Still an approximation, which might or might not be appropriate
- Nonstationary kernels are harder to quantize
- Might require many basisfunctions m in higher dims
- Kernel algebra rapdily increases m

Quantization [1]
================

<!-- column_layout: [1,1] -->
<!-- column: 0 -->
<!-- newlines: 2 -->

# Hilbert-GP

A stationary kernel is translation invariant, so it admits a spectral representation:

```
k(r) = ∫ S(ω) e^{iωr} dω
```

Hilbert-GP approximates this integral by regular quadrature.
This results in the following ridge regression model:

```
f(x) = a₁ϕ₁(x) + a₂ϕ₂(x) + ... + aₘϕₘ(x)


ϕₖ(x) = √(2/L) · sin(ωₖ x),   ωₖ = kπ / L
  Σₖₖ = S(ωₖ)
```

So higher frequencies have higher cost => determines lengthscale and smoothness.

<!-- column: 1 -->

```bash +exec_replace
graph-easy --from=dot --as_boxart << 'EOF'
digraph {
    rankdir = LR;
    "Stationary GP" -> "ridge" [label="Hilbert!"];
}
EOF
```

```bash +image
gnuplot <<'GP'
set term pngcairo size 600,450 background rgb "black" enhanced

set border lc rgb "#4c566a"
set tics textcolor rgb "#4c566a"

set xlabel "frequency ω" tc rgb "#d8dee9"
set ylabel "S(ω)" tc rgb "#d8dee9"

set xrange [0:10]
set yrange [1e-5:2]
set logscale y

# Legend
set key top right
set key tc rgb "#d8dee9"
set key spacing 1.2
set key box lc rgb "#4c566a"

set title "Kernel magnitude spectra (ℓ = 1)" tc rgb "#d8dee9"

# lengthscale
ell = 1.0

# Matérn parameters
k12 = sqrt(2*0.5)/ell
k32 = sqrt(2*1.5)/ell
k52 = sqrt(2*2.5)/ell

# Spectral densities (up to constants)
S12(w) = 1.0 / (w*w + k12*k12)**(1.0)
S32(w) = 1.0 / (w*w + k32*k32)**(2.0)
S52(w) = 1.0 / (w*w + k52*k52)**(3.0)

# Squared exponential spectrum
SSE(w) = exp(-0.5 * ell*ell * w*w)

plot \
    S12(x) w l lw 3 lc rgb "#bf616a" title "Matérn ν=1/2", \
    S32(x) w l lw 3 lc rgb "#d08770" title "Matérn ν=3/2", \
    S52(x) w l lw 3 lc rgb "#a3be8c" title "Matérn ν=5/2", \
    SSE(x) w l lw 3 lc rgb "#81a1c1" dt 2 title "SE"
GP
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

Learning on levels 1 & 2
========================

level | Gaussian process | ridge regression
--|--|--
0 | prior | prior
1 | hyperparam learning | learning the basis Φ and costs Σ
2 |   | learning the amplitudes a


# Quantization creates a more white-box model, which allows for a bonus: ***learning from examples***

Level 0: no learning

Level 1: learning the geometry itself

Level 2: learning the coordinates inside the geometry



```bash +exec +acquire_terminal
python live/levels.py
```

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