library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "modal/pmatern/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    modality,
    nu,
    M,
    logz,
    logzerr,
    information,
    iteration,
    mean.sigma_a_log10,
    mean.sigma_c_log10,
    mean.sigma_noise_log10,
    te
)]

df[
    ,
    `:=`(
        modality = as.factor(modality),
        nu = factor(nu, levels = sort(unique(nu)), ordered = TRUE)
    )
]

# aggregate over iterations
df <- df[
    ,
    {
        n <- .N
        m_logz <- mean(logz)
        var_mean <- sum(logzerr^2) / (n * n)
        m_logzerr <- sqrt(var_mean)
        m_info <- mean(information)
        m_sigma_a_log10 <- mean(mean.sigma_a_log10)
        m_sigma_c_log10 <- mean(mean.sigma_c_log10)
        m_sigma_noise_log10 <- mean(mean.sigma_noise_log10)
        list(
            logz = m_logz,
            # logzerr = m_logzerr,
            information = m_info,
            mean.sigma_a_log10 = m_sigma_a_log10,
            mean.sigma_c_log10 = m_sigma_c_log10,
            mean.sigma_noise_log10 = m_sigma_noise_log10,
            te = mean(te)
        )
    },
# styler: off
    by = .(modality, nu, M)
    # styler: on
]

# set sorting index
setorder(df, modality, -logz)

# look at results manually (see README.md)
View(df)

# display top 5 results per modality
df[
    ,
    .SD[1:10],
    by <- modality
]

# best kernel per modality by logz
kernel_scores <- df[
    ,
    .(
        n_modalities = uniqueN(modality),
        sum_logz = sum(logz),
        sum_information = sum(information)
    ),
    by = .(nu, M)
]

kernel_scores <- kernel_scores[n_modalities == 4]

rank_logz <- kernel_scores[
    order(-sum_logz)
]

rank_info <- kernel_scores[
    order(sum_information)
]

View(rank_logz)
View(rank_info)

# pareto on normalized
df[
    ,
    `:=`(
        logz_z = (logz - mean(logz)) / sd(logz),
        information_z = (information - mean(information)) / sd(information)
    ),
    by = modality
]

kernel_scores <- df[
    ,
    .(
        n_modalities = uniqueN(modality),
        sum_logz_z = sum(logz_z),
        sum_information_z = sum(information_z)
    ),
    by = .(nu, M)
][
    n_modalities == 4
]

rank_logz <- kernel_scores[order(-sum_logz_z)]
rank_info <- kernel_scores[order(sum_information_z)]

View(rank_logz)
View(rank_info)

setorder(kernel_scores, -sum_logz_z, sum_information_z)

kernel_scores[, best_info := cummin(sum_information_z)]

pareto <- kernel_scores[
    (seq_len(.N) == 1L) | (sum_information_z < shift(best_info))
]

pareto[, best_info := NULL]

View(pareto)

# view distribution of walltimes
summary(df$walltime)
