library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "pack_refine/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    pitch,
    kernel,
    gauge,
    beta,
    refine,
    scale_dgf_to_unit_power,
    results.dgf_both_aligned_nrmse,
    results.vowel,
    results.modality,run,
    results.elbo
)]

df[
    ,
    `:=`(
        kernel = as.factor(kernel),
        beta = factor(beta, ordered = TRUE),
        score = sqrt(pmax(0, 1 - results.dgf_both_aligned_nrmse^2)), # cosine similarity
        vowel = as.factor(results.vowel),
        results.modality = as.factor(results.modality)
    )
]

# there are NO nan elbos!
# could this be due to results.posterior_pi = 0.5 initialized?
summary(df$results.elbo)


# test fx
df[, pitch_c := pitch - mean(pitch)]

model2 <- lm(
    score ~
        kernel * pitch_c +
        beta +
        gauge +
        refine + # singular: linear combination of others
        scale_dgf_to_unit_power +
        results.modality,
    data = df
)

summary(model2)

coefs <- coef(summary(model2))

# beta has no effect
# whitenoise kernel is worst
# gauge ON worsens fit
# kernelspack:3 dominates
coefs[order(abs(coefs[, "t value"]), decreasing = TRUE), ]


geom_density(aes(x = results.dgf_nrmse, fill = interaction(P, scale_dgf_to_unit_power, gauge, window_type)), alpha = 0.1) +
    facet_grid(pitch ~ kernel, scales = "free") +
    guides(fill = "none")




ggplot(df) +
    geom_density(aes(x = results.dgf_nrmse, fill = window_type), alpha = 0.3) +
    facet_grid(pitch ~ kernel, scales = "free")



ggplot(df) +
    geom_density(aes(x = score, fill = window_type), alpha = 0.3) +
    facet_grid(pitch ~ kernel, scales = "free")


ggplot(df[window_type == "adaptive" & gauge == FALSE]) +
    geom_density(aes(x = score, fill = results.modality), alpha = 0.3) +
    facet_grid(pitch ~ kernel, scales = "free")


ggplot(df[window_type == "adaptive" & gauge == FALSE & kernel %in% c("periodickernel", "spack:3")]) +
    geom_density(aes(x = score, fill = kernel), alpha = 0.3) +
    facet_grid(pitch ~ results.modality, scales = "free_y")

# & results.modality == "whispery"] & kernel %in% c("periodickernel")

## at 100 Hz spack dominate clearly

ggplot(df[refine == FALSE & pitch == 100]) +
    geom_density(aes(x = score, fill = kernel), alpha = 0.3)
