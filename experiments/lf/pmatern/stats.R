library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lf/pmatern/runs.csv")
runs <- data.table(fread(runs_file))

neffs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lf/pack/neffs.csv")
neff <- data.table(fread(neffs_file))

runs <- merge(runs, neff, by = "sample_idx", all.x = TRUE)

df <- runs[, .(
    nu,
    M,
    log_prob_u,
    logz,
    logzerr,
    information,
    neff,
    mean.sigma_a_log10,
    mean.sigma_c_log10,
    mean.sigma_noise_log10
)]

df[
    ,
    `:=`(
        log_prob_u_eff = log_prob_u / neff,
        logz_eff = logz / neff,
        information_eff = information / neff
    )
]

means <- df[, c(.N, lapply(.SD, mean, na.rm = TRUE)), by = .(nu, M)]

means[, D_KL := log_prob_u - logz]
means[, D_KL_eff := log_prob_u_eff - logz_eff]

# set sorting index
setorder(means, D_KL)

means
