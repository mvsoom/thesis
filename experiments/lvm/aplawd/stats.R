library(ggplot2)
library(plotly)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lvm/aplawd/runs.csv")
runs <- data.table(fread(runs_file))

# Any nans in D_KL?
runs[is.nan(results.mean_gmm_loglikelihood), .N]

quick <- runs[, .(kernelname, M, Q, results.K, results.mean_gmm_loglikelihood, svi_lengthscale, svi_obs_std)]
quick <- unique(quick)
setorder(quick, -results.mean_gmm_loglikelihood)

View(quick)

# Check cost of SVI vs LVM
runs[, .(mean(svi_walltime) / 60, mean(lvm_walltime) / 60)] # minutes

# drop LVM means ~2 times speedup

# This entire run was ~20 hrs
