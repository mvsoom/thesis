library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "pack_modal/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    modality,
    kernel,
    effective_num_harmonics,
    logz,
    logzerr,
    information,
    normalized,
    mean.sigma_noise_log10,
    mean.sigma_a_log10,
    mean.sigma_b_log10,
    mean.center,
    mean.tc,
    te,
    walltime
)]

df[
    ,
    `:=`(
        modality = as.factor(modality),
        kernel = as.factor(kernel)
    )
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

ggplot(df) +
    facet_wrap(~modality, scales = "free") +
    geom_density(aes(x = mean.tc)) +
    geom_density(aes(x = mean.center), color = "red") +
    geom_vline(aes(xintercept = te), color = "blue", linetype = "dashed")
summary(df)
