library(ggplot2)
library(plotly)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lvm/qpack_test/runs.csv")
runs <- data.table(fread(runs_file))

# Any nans in D_KL?
runs[is.nan(results.mean_gmm_loglikelihood), .N]

quick <- runs[, .(d, M, Q, results.K, results.mean_gmm_loglikelihood)]
quick <- unique(quick)
setorder(quick, -results.mean_gmm_loglikelihood)

View(quick)
