library(ggplot2)
library(plotly)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lvm/pack_01/runs.csv")
runs <- data.table(fread(runs_file))

# Any nans in D_KL?
runs[is.nan(results.D_KL_bans_per_sample)]

quick <- runs[!is.na(results.D_KL_bans_per_sample), .(d, M, Q, results.K, results.D_KL_bans_per_sample)]
quick <- unique(quick)
setorder(quick, results.D_KL_bans_per_sample)
quick

View(quick)
