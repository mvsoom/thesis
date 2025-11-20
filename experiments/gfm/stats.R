library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "gfm/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    examplar_name,
    d,
    kernel,
    logz,
    logzerr,
    information,
    centered,
    normalized
)]

df[
    ,
    `:=`(
        examplar_name = as.factor(examplar_name),
        d = factor(d, ordered = TRUE),
        kernel = as.factor(kernel)
    )
]

# set sorting index
setorder(df, examplar_name, d, -logz)

# look at results manually (see README.md)
View(df)

# investigate centered & normalized interaction with logz => consistent improvement for logz and H?
acks <- df[grepl("ack", kernel)]

m <- lm(logz ~ kernel + centered + normalized, data = acks)
summary(m) # For this subset of kernels, neither centering nor normalizing shows any consistent improvement in logz, and whatever differences exist across tack kernels are weak and unstable.

# information H?
m <- lm(information ~ kernel + centered + normalized, data = acks)
summary(m) # normalization tends to INCREASE H, centering tends to DECREASE H
