library(ggplot2)
library(plotly)
library(data.table)
library(errors)
options(errors.digits = 1)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "svi/aplawd_rational_quadratic/runs.csv")
runs <- data.table(fread(runs_file))

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
            svi_lengthscale = mean(svi_lengthscale),
            se_lengthscale = se_seed(svi_lengthscale),
            svi_obs_std = mean(svi_obs_std),
            se_obs_std = se_seed(svi_obs_std),
            svi_alpha = mean(svi_alpha),
            se_alpha = se_seed(svi_alpha),
            N = .N
        )
    },
    by = .(kernelname, M)
]


agg[, `:=`(
    score       = set_errors(mean_loglike_test, se_loglike_test),
    score95     = set_errors(mean_loglike_test, 2 * se_loglike_test),
    score_null  = set_errors(mean_loglike_test_null, se_loglike_test_null),
    lengthscale = set_errors(svi_lengthscale, se_lengthscale),
    alpha       = set_errors(svi_alpha, se_alpha),
    obs_std     = set_errors(svi_obs_std, se_obs_std)
)]

# best model
agg[, .(kernelname, M, score, score95, lengthscale, alpha, obs_std)][order(-score)]

# best null
agg[, .(kernelname, M, score_null)][order(-score_null)]
