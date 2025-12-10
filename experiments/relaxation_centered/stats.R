library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "relaxation_centered/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    Rd,
    kernel,
    logz,
    logzerr,
    information,
    centered,
    normalized,
    iteration,
    mean.sigma_a_log10,
    mean.sigma_b_log10,
    mean.sigma_c_log10,
    mean.center,
    te
)]

df[
    ,
    `:=`(
        Rd = factor(Rd, ordered = TRUE),
        kernel = as.factor(kernel)
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
        m_sigma_b_log10 <- mean(mean.sigma_b_log10)
        m_sigma_c_log10 <- mean(mean.sigma_c_log10)
        m_center <- mean(mean.center)
        list(
            logz = m_logz,
            # logzerr = m_logzerr,
            information = m_info,
            mean.sigma_a_log10 = m_sigma_a_log10,
            mean.sigma_b_log10 = m_sigma_b_log10,
            mean.sigma_c_log10 = m_sigma_c_log10,
            mean.center = m_center,
            te = mean(te)
        )
    },
    by = .(Rd, kernel, centered, normalized)
]

# set sorting index
setorder(df, Rd, -logz)

# look at results manually (see README.md)
View(df)

# display top 5 results per Rd
df[
    ,
    .SD[1:10],
    by = Rd
]

# investigate centered & normalized interaction with logz => consistent improvement for logz and H?
acks <- df[grepl("ack", kernel)]

m <- lm(logz ~ kernel + centered + normalized, data = acks)
summary(m) # For this subset of kernels, neither centering nor normalizing shows any consistent improvement in logz, and whatever differences exist across tack kernels are weak and unstable.

# information H?
m <- lm(information ~ kernel + centered + normalized, data = acks)
summary(m) # normalization tends to INCREASE H, centering tends to DECREASE H
