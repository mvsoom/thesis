# Kernel identification experiment

> Use IKLP to identify kernel hyperparameters through laying them out on a grid


Works well both for Hilbert as SVD Mercer features
- Important: alpha *= 0.1 to really have a single component dominate
- Often ell is found but at wrong nu
- Good: (4 x 5), (1 x 20): OK at N_data = 1024
- Used N_data = 1024 for N_ell = 1, 5, 10, 20



