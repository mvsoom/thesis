library(ggplot2)
library(plotly)
library(data.table)

# Get experiment dir from environment variable
runs_file <- file.path(
    Sys.getenv("PROJECT_EXPERIMENTS_PATH"),
    "lvm/egifa/pack/runs.csv"
)

runs <- data.table(fread(runs_file))

df <- runs[, .(
    J,
    M,
    Q,
    results.K,
    results.mean_loglike_test,
    results.std_loglike_test,
    svi_obs_std,
    svi_variance
)]

df <- unique(df)

# order numerically (always sort on the raw mean)
setorder(df, -results.mean_loglike_test)

# build formatted score string manually cos errors() chokes for some bullshit R reason
df[, score := sprintf(
    "%.1f(%.1f)",
    results.mean_loglike_test,
    results.std_loglike_test
)]

# final table
df <- df[, .(
    compute = M * results.K,
    score,
    J,
    M,
    Q,
    results.K
)]

df[]

# %%
# print weights of the best model
best <- df[1]
best_run <- runs[
    M == best$M &
        Q == best$Q &
        results.K == best$results.K &
        J == best$J
]

best_run
