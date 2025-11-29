- [ ] Mercer
    - [x] Needs no awareness of ndim
    - [x] +, * : always stack/kathri rao
    - [ ] Transforms always possible
    - [ ] Differentiation possible
    - [x] Scaling via weights
    - [ ] Subspace is an ordinary Transform

- [ ] Hilbert
    - [x] Dont need axis info, only ndim
    - [x] ndim via post init: need to specify if M, L scalar or assume 1. Make len M and len L ndim info.
    - [ ] Can always assume same domain: +, *: sum and convolution spectral densities. M = max(M1, M2), can differ!
    - [ ] Arbitrary transform: jax.Bijector and logdet. Rewrite Cholesky in that way. Spectral density is known via inverse and logdet of Bijectors!
    - [ ] Go hard on numerical convolution, massive gains: O(1) vs O(M1*M2)!

- [ ] Subspace(Hilbert, transforms.Subspace)
    - [ ] This has axes prop built in
    - [ ] Define sum and mul here: then we know everything.
        - [ ] If other is another Hilbert Subspace, check if identical and if so use Hilbert for +,* above
        - [ ] Else fallback to Mercer
    - [ ] Convolution possibly still salvageable even for different subspaces: absent axes just produce deltas

- [ ] solvers/mercer.py
    - Need to implement a Solver() to take full advantage of Mercer structure!
    - But tinygp only allows solvers to represent K = LL^t, not K = RR^t + sigma^2 I where R is low-rank and L is full rank triangular
    - Already a hacky, NON TESTED implementation à la Hilbert-GP
    - Basically we need to implement our MercerOp here


##


kx = Hilbert(Matern32(), M, L, D=2)
ky = Hilbert(Matern32(), M, L, D=2)
kxy = Hilbert(Matern32(), M, L, D=2)

# Hilbert addition
kx + ky

# Hilbert multiplication
kx * ky

# Mercer stack of 1D kernels via Subspace.__add__
Subspace(0, kx) + Subspace(1, ky)

# Mercer stack via Hilbert.__add__ and/or Subspace.__add__
Subspace(0, kx) + kxy

# Mercer product
Subspace(0, kx) * kxy
