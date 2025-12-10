library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "pack/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    pitch,
    kernel,
    P,
    gauge,
    prior_pi,
    scale_dgf_to_unit_power,
    results.dgf_aligned_nrmse,
    results.dgf_nrmse
)]

df[
    ,
    `:=`(
        kernel = as.factor(kernel),
        P = factor(P, ordered = TRUE)
    )
]

# distribution of invalid results.dgf_nrmse?
invalid <- df[is.na(results.dgf_nrmse) | is.infinite(results.dgf_nrmse)]

# => neglible <2%, tho usually periodic kernels, so spack kernels much more robust
summary(invalid[, .(kernel)])
summary(df[, .(kernel)])

df <- df[!is.na(results.dgf_nrmse) & !is.infinite(results.dgf_nrmse)]

# show density for all combinations
ggplot(df) +
    geom_density(aes(x = results.dgf_aligned_nrmse), fill = "black") +
    facet_grid(pitch ~ kernel, scales = "free")

# quick regression to see what factors contribute
# => very little
model <- lm(
    log10(results.dgf_aligned_nrmse) ~ kernel + pitch + P + gauge + prior_pi + scale_dgf_to_unit_power,
    data = df
)
summary(model)

### select only kernel=="periodickernel" to see pitch and P effects
summary(lm(
    (results.dgf_nrmse) ~ pitch + P + gauge + prior_pi + scale_dgf_to_unit_power,
    data = df[kernel == "spack:1"]
))


# what are best scoring instances?
df[order(results.dgf_aligned_nrmse)][1:10]

View(df[order(results.dgf_nrmse)])

# average scores per kernel across all other factors
# => spack:d ~= periodickernel
df[
    ,
    .(
        aligned = median(results.dgf_aligned_nrmse),
        non_aligned = median(results.dgf_nrmse)
    ),
    by = .(kernel)
][order(aligned)]

# see contributions to score per kernel from factors
ggplot(runs) +
    geom_density(aes(x = results.posterior_pi, fill = factor(prior_pi)))
