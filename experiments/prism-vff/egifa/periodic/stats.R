# %%
library(ggplot2)
library(plotly)
library(data.table)
library(errors)
options(errors.digits = 1)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "prism-vff/egifa/periodic/runs.csv")
runs <- data.table(fread(runs_file))

# %%
# inspect distribution of pack weights, then drop them

# select only pack kernels in kernelname
pack <- runs[grepl("pack", kernelname)]

# plot smoothed density of pack weights according to kernelname
ggplot(pack, aes(x = log10(svi_lengthscale_or_weights), color = kernelname)) +
    geom_density() +
    theme_minimal() +
    facet_wrap(~J) +
    labs(title = "Density of pack log10(weights) by kernel")

# looks healthy

# %%
#
periodic <- runs[grepl("periodic", kernelname)]

# plot density of lengthscales
ggplot(periodic, aes(x = (svi_lengthscale_or_weights), color = kernelname)) +
    geom_density() +
    theme_minimal() +
    labs(title = "Density of periodic lengthscale for periodic kernel")

# looks good

# %%

# We have null models for each (kernel, M)
# We can choose best null model amongst these as global null model to set scale

# Which model will score best on average for a random draw from the test distribution?
N_TEST <- 1000L

agg <- runs[
    ,
    {
        # shorthand
        mu <- mean_loglike_test
        s <- std_loglike_test
        mu0 <- mean_loglike_test_null
        s0 <- std_loglike_test_null

        # helper
        se_seed <- function(x) sd(x) / sqrt(.N)

        # test score uncertainty (seed + test sampling)
        v_seed <- var(mu)
        v_test <- mean(s^2) / N_TEST
        se_tot <- sqrt(v_seed / .N + v_test / .N)

        # same for null
        v_seed0 <- var(mu0)
        v_test0 <- mean(s0^2) / N_TEST
        se_tot0 <- sqrt(v_seed0 / .N + v_test0 / .N)

        .(
            mean_loglike_test = mean(mu),
            se_loglike_test = se_tot,
            mean_loglike_test_null = mean(mu0),
            se_loglike_test_null = se_tot0,
            svi_obs_std = mean(svi_obs_std),
            se_obs_std = se_seed(svi_obs_std),
            N = .N
        )
    },
    by <- .(kernelname, M, J)
]

agg[, `:=`(
    score_mean      = mean_loglike_test,
    score_se        = se_loglike_test,
    score95_half    = 2 * se_loglike_test,
    score_null_mean = mean_loglike_test_null,
    score_null_se   = se_loglike_test_null,
    obs_std_mean    = svi_obs_std,
    obs_std_se      = se_obs_std
)]

fmt <- function(m, s, k = 1) {
    sprintf("%.3f ± %.3f", m, k * s)
}

agg[, `:=`(
    score      = fmt(score_mean, score_se),
    score95    = fmt(score_mean, score_se, 2),
    score_null = fmt(score_null_mean, score_null_se),
    obs_std    = fmt(obs_std_mean, obs_std_se)
)]

# best model
best <- agg[
    order(-score_mean),
    .(kernelname, M, J, score, score95, obs_std)
]

best[1:10]

# best null
best0 <- agg[
    order(-score_null_mean),
    .(kernelname, M, J, score_null)
]

best0[1:10]

# %%
# extract best hyper parameters per kernelname

best_cfg <- agg[
    order(-score_mean),
    .SD[1],
    by = kernelname
][, .(kernelname, M, J)]

best_cfg