alright, thanks! i have to think a bit about the msg i want to tell
ill hardly use math and try to keep it very simple
lets brainstorm together (no code or lists etc)
things i want to say, in rough order

i am workign on "speech inversion"
this is a blind deconvolution problem (or whatever its called): many to one
=> need prior information to constrain to whatevers plausibe

[now this is a crowd of engineers so they think in costs, so ill talk about assigning initial costs and that bayesian inference is a way to correctly work with these costs: bayesian inference is ... bookkeeping]

then: zoom out
many problems are of this kind: regression etc => we got many good off the shelf models we can use (matlab algs, python libs), but how do we "get them where we want them": be general enough but kinda do our bidding, kinda "know" what we want?
this is a hard question and very general
we can start by thinking about science's workhorse: linear regression

so, linear regression (meaning basis functions etc)
ridge regression solved a lot of stuff
and has a strong bearing on interpolation problems [WE MAKE ANOTHER INTERACTIVE DEMO HERE: got toy data and see what ridge regression does based on weights; can be ordinary LR or with basis functions, dont know whats best; try also random covariance matrices to see impact on fit]
but it is arbitraruy: which weights/alpha/ridge do we set?
why not a general covariance matrix?
how do we pick it?
also: how do pick the basis functions? they aobviously matter a lot

bam, insight: use GPs and quantify them to BLR (regression models)
this ives you Phi (basis functions) AND the covariance matrix (the costs) in one go
and we did it by just suplying HIGH LEVEL INFORMATION!
=> let the model work out the algebraic consequences.
so this really does what we wanti th

extra benfits:
- very easy representation for any stationary kernel
- O(N) inference, not N^3, generally reasonable approximation
- reveals the structure of the GP because we can rotate Phi x Sigma^(1/2) which gives us equally weighted basis functions, a bit like PCA
- all sparse VIs are compatible/based on this, so we get arbitrary likelihoods, batching, etc
- (any ones you can think of here?)

downsides:
- still an approx, which might break down
- might require many basisfunctions in higher dims : O(10) max, tho many clever ways try to go around this [MCMC : fourier features, reorthogonalizing: Eleftheriadis2023, ...]
- nonstationary kernels are generally hard to get BLR repr from
- (any ones you can think of here?)

best method for this currently in low dim is Hilbert-GP
which is basically quadrature of bochners theorem
fixed basis functions , but we get the diagonal weights, which is insae
here we should do another interactive demo showing whatsup
any ideas what we would do? we should show the spectrum at least

then we do a summary slide of the GP-BLR connection

bonus: this allows "level 1" learning
say you have examplars of what your prior should try to emulate from a simulation
with ordinary GP: optimize hyperparams and you get something more like it: level 2 learned
but usually not quite refined enough if your examplars have definite shape
and here we go back to my use case
say we have a glottal flow model that can generate samples
if i quantize my prior, i can learn the shape from it both level 1 (amplitude level) and level 2 (high level)
this results in another BLR model where amps have defintie prior mean and covariance
here we show level 1 and level 2 adaptation for whitenoise kernel, periodickernel and my spack kernel (custom derived) which should be a plot containing: col 1: prior, no conditioning, col 2: prior, level 1 learnt, col 3: prior, level 1 + level 2 learnt and rows are the kernels. each cell contains samples from the prior and pressing a key adds another examplar to the learning examplar pool

then i show benchmark results for whitenoise, periodic kernel and spack and show that examplar learning really does pay off


