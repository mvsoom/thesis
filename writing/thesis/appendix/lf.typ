= The Liljencrants-Fant model for glottal flow
<appendix:lf>

LF is widely used, though models of similar if not better quality exist

Mainly computationably cumbersome due to non-analytical tractablilty: requires solving a bisection routine for each numerical sample.

Merit: baseline for comparison

We also built a jax-compatible library which can differentiate through this, and which can simulate realistic changes in amplitude (shimmer), fundamental frequency (jitter), open quotient and others.

Differentiable and batchable.
