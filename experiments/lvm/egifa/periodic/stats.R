library(ggplot2)
library(plotly)
library(data.table)
library(errors)
options(errors.digits = 1)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lvm/egifa/periodic/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(M, Q, results.K, results.mean_loglike_test, results.std_loglike_test, svi_lengthscale, svi_obs_std, svi_variance)]
df <- unique(df)

df[, score := set_errors(results.mean_loglike_test, results.std_loglike_test), ]
df[, compute := M * results.K, ]

df <- df[, .(compute, score, svi_lengthscale, M, Q, results.K)]

setorder(df, -score)

df[]
