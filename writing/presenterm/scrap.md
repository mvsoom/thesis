Speech waveforms
================

<!-- speaker_note:
The presentation is about how to represent speech waveforms as Gaussian processes

Speech waveforms are 1D pressure waveforms hitting our eardrums and being converted to electric signals

So how would "that" look like?
-->

<!-- pause -->

```bash +image
convert -density 300 assets/that.png -resize 80%x -background none  -channel RGB -negate png:-
```

What can we do?
===============

![](assets/keepcalm.png)

Once you assign initial costs to your liking, the algebra for consistently reasoning with them is unique and known as Bayesian inference

> Bayesian inference is just: bookkeeping.

Assigning initial costs is known as wielding a _prior_

Priors depend on your personality and haircut

Just kidding, in many cases you can assign them objectively

Unifing principle behind modern GPs
===================================

> Any explicit feature map 𝜙(𝑥) ϕ(x) — harmonic, random Fourier, neural, interpolation-based, learned, whatever — is just a way to write the GP prior as a Bayesian linear regression in a low-dimensional latent 𝑧. And once you have that, VI becomes the canonical inference method.

- harmonic features
- Fourier features
- spherical harmonics
- splines
- Mercer eigenfunctions
- learned neural features
- a truncated Mercer decomposition
- inducing points projected via kernel interpolation

All approximations of GPs fall into two buckets:

# (A) covariance-space approximations

- inducing points
- Nyström
- etc.

## These eventually reduce to an implicit feature representation.

# (B) feature-space approximations

- random Fourier features
- deterministic harmonic features
- Mercer eigenfunctions

## These explicitly give you 𝜙(𝑥)

Both categories reduce to the same BLR form.
And once you're in BLR, VI is the canonical inference method.


---

here’s a phrasing that tends to land really well:

“least squares asks:
can I fit the data?
ridge asks:
can I fit the data without doing something extreme?”

notice: no matrices, no distributions, just taste.

later, when you go GP/BLR, you’ll be able to say:

“we’ve just been manually designing what ‘extreme’ means.”

that’s the hing

---








# Ill-posedness

<!-- speaker_note:
This problem is ill-posed.

There are infinitely many valid solutions,
and the data alone cannot tell them apart.

So without further assumptions,
the problem cannot be solved.
-->

a blind inversion problem
ill-posed
needs regularization

---

# Priors as costs

<!-- speaker_note:
At this point,
we need prior information to constrain solutions
to what we consider plausible.

Engineers often think in terms of costs or penalties.

Bayesian inference is a way to work with such costs
consistently.
-->

assign initial costs
→ update with data

---

# Zooming out

<!-- speaker_note:
This is not unique to speech inversion.

Many problems in science and engineering
have this structure:
regression, interpolation, inverse problems.

We often use off-the-shelf models for them.
The hard part is getting them to do what we want.
-->

many problems
same structure

---

# A familiar tool

<!-- speaker_note:
A common starting point is linear regression.

We represent functions using basis functions
and infer weights.

This is a workhorse of science.
-->

linear regression
basis functions
weights

---

# Regularization

<!-- speaker_note:
Ridge regression solved many practical issues.

It stabilizes estimation
and improves generalization.

But it introduces new choices.
-->

penalties
weights
regularization strength

---

# Arbitrariness

<!-- speaker_note:
Which weights should be penalized more?

Why only diagonal penalties?
Why these basis functions?

The choices matter,
but they often feel arbitrary.
-->

why these costs?
why these bases?

---

# Key insight

<!-- speaker_note:
Gaussian processes offer a way out.

Instead of choosing basis functions and penalties directly,
we specify high-level assumptions
about correlations between function values.

Everything else follows from that.
-->

specify correlations
derive the rest

---

# Gaussian processes

<!-- speaker_note:
A GP is a prior over functions.

For any finite collection of inputs,
the function values are jointly Gaussian,
with covariance given by a kernel.
-->

joint Gaussian
covariance = kernel

---

# From GPs to regression

<!-- speaker_note:
Many GP priors can be rewritten
as Bayesian linear regression models
in a suitable feature space.

This gives us both
the basis functions
and the weight covariance.
-->

GP
→ BLR

---

# What this gives us

<!-- speaker_note:
This representation is very powerful.

Inference becomes efficient.
Structure becomes explicit.
Approximate methods fit naturally.
-->

features
covariance
structure

---

# A unifying view

<!-- speaker_note:
Harmonic features,
Fourier features,
splines,
Mercer eigenfunctions,
learned features.

These are all ways of writing phi(x).
-->

different features
same model

---

# Limits

<!-- speaker_note:
This is still an approximation.

It can break down,
especially in higher dimensions.

Some kernels are harder to represent this way.
-->

approximation
limits

---

# Hilbert-GPs

<!-- speaker_note:
In low dimensions,
there is a particularly nice construction.

Hilbert-GPs give fixed basis functions
and learn only diagonal weights.
-->

fixed basis
learned spectrum

---

# Looking at the spectrum

<!-- speaker_note:
Viewing the prior in the spectral domain
reveals what it believes.

This is often more informative
than looking in the time domain.
-->

spectrum
structure

---

# Learning priors

<!-- speaker_note:
Sometimes we know what plausible solutions look like
because we can simulate them.

We can use these exemplars
to learn a prior.
-->

learn from exemplars

---

# Two levels

<!-- speaker_note:
There are two levels of learning.

Level 1:
amplitude and scale.

Level 2:
shape and structure.
-->

level 1
level 2

---

# Back to speech

<!-- speaker_note:
In speech,
we often have glottal flow models
that can generate samples.

We can use them
to learn priors
before observing real data.
-->

domain knowledge
as prior

---

# Summary

<!-- speaker_note:
We started with an ill-posed inverse problem.

We introduced priors as costs.
Gaussian processes organize these costs.
Feature-space views let us learn them.
-->

ill-posed
→ priors
→ GPs
→ BLR

---

# Takeaway

<!-- speaker_note:
Many problems are underdetermined.

Priors are unavoidable.

Gaussian processes provide
a principled way to construct and learn them.
-->

structure
first-class