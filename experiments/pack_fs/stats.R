library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "pack_fs/runs.csv")
runs <- data.table(fread(runs_file))

df <- runs[, .(
    pitch,
    kernel,
    P,
    gauge,
    scale_dgf_to_unit_power,
    window_type,
    results.dgf_aligned_nrmse,
    results.dgf_nrmse, run,
    results.vowel,
    results.modality,
    results.E_nu_w,
    results.E_nu_e
)]


df[
    ,
    `:=`(
        kernel = as.factor(kernel),
        window_type = as.factor(window_type),
        P = factor(P, ordered = TRUE),
        score = sqrt(pmax(0, 1 - results.dgf_aligned_nrmse^2)),
        vowel = as.factor(results.vowel),
        results.modality = as.factor(results.modality)
    )
]

df[, pitch_c := pitch - mean(pitch)]

model2 <- lm(
    log10(score) ~
        kernel * pitch_c +
        P + gauge + scale_dgf_to_unit_power + window_type + results.modality + log10(results.E_nu_w) + log10(results.E_nu_e),
    data = df
)
summary(model2)


##


ggplot(df) +
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

ggplot(df[window_type == "adaptive" & gauge == FALSE & pitch == 100]) +
    geom_density(aes(x = score, fill = kernel), alpha = 0.3)
