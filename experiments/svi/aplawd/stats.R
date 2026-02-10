library(ggplot2)
library(plotly)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "svi/aplawd/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(kernelname, M, mean_loglike_test, std_loglike_test, svi_lengthscale, svi_obs_std, mean_loglike_test_null, std_loglike_test_null)]
setorder(df, -mean_loglike_test)

View(df)

# We have null models for each (kernel, M)
# We can choose best null model amongst these as global null model to set scale
best_model <- df[1]
best_null <- df[which.max(mean_loglike_test_null)]

View(rbind(best_model, best_null))
