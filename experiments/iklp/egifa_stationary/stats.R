library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)
library(plotly)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_SCRAP_EXPERIMENTS_PATH"), "pack_refine/runs.csv")
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
    results.modality, run,
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

# there are only 10 nan elbos!
# could this be due to results.posterior_pi = 0.5 initialized?
summary(df$results.elbo)

df <- df[!is.nan(results.elbo)]


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

plot <- ggplot(df[refine == FALSE & kernel %in% c("whitenoise", "periodickernel", "spack:3")]) +
    geom_density(aes(x = score, fill = kernel), alpha = 0.3)


ggplotly(plot)


############################################################
## CDF-based null calibration
############################################################

# --- 1) build empirical null CDF from whitenoise
null_scores <- df[kernel == "whitenoise", score]

# empirical CDF function
F0 <- ecdf(null_scores)

# --- 2) apply calibration
df[, u_cal := F0(score)]

# optional: avoid exact 0/1 for plotting/log transforms
eps <- 1e-6
df[, u_cal := pmin(pmax(u_cal, eps), 1 - eps)]

# optional z transform (not required, but useful sometimes)
df[, z_cal := qnorm(u_cal)]

############################################################
## Diagnostics
############################################################

# null should be uniform
p_null <- ggplot(df[kernel == "whitenoise"]) +
    geom_density(aes(x = u_cal), fill = "steelblue", alpha = 0.4) +
    ggtitle("Null after calibration (should look flat/uniform-ish)")

print(p_null)

# compare calibrated distributions across kernels
p_cal <- ggplot(df[refine == FALSE &
    kernel %in% c("whitenoise", "periodickernel", "spack:3")]) +
    geom_density(aes(x = u_cal, fill = kernel), alpha = 0.3) +
    ggtitle("CDF-calibrated score (u-space)")

print(p_cal)

# interactive version
ggplotly(p_cal)

############################################################
## Simple scalar comparison (optional)
############################################################

# mean(u) — null expectation = 0.5
df_summary <- df[
    refine == FALSE &
        kernel %in% c("whitenoise", "periodickernel", "spack:3"),
    .(
        mean_u = mean(u_cal),
        tail_95 = mean(u_cal > 0.95),
        tail_99 = mean(u_cal > 0.99)
    ),
    by = kernel
]

print(df_summary)

############################################################
## D_KL( p_model(u) || Uniform )
## “how far is this model from null behaviour?”
############################################################

# helper: KL divergence using histogram estimate
kl_to_uniform <- function(u, nbins = 50) {
    # histogram density estimate
    h <- hist(u, breaks = nbins, plot = FALSE)

    p <- h$density
    binwidth <- diff(h$breaks)[1]

    # avoid log(0)
    eps <- 1e-12
    p <- p + eps

    # uniform density on [0,1] is 1
    q <- 1

    # discrete approx to integral
    sum(p * log(p / q)) * binwidth
}

kl_results <- df[
    refine == FALSE &
        kernel %in% c("whitenoise", "periodickernel", "spack:3"),
    .(D_KL = kl_to_uniform(u_cal)),
    by = kernel
]

print(kl_results)

############################################################
## Pseudo likelihood ratio / energy score
############################################################

df[, energy_score := -log(1 - u_cal)]

energy_summary <- df[
    refine == FALSE &
        kernel %in% c("whitenoise", "periodickernel", "spack:3"),
    .(
        mean_energy = mean(energy_score),
        median_energy = median(energy_score)
    ),
    by = kernel
]

print(energy_summary)

############################################################
## Practical equivalence scale
############################################################

# pairwise model comparison
models <- c("periodickernel", "spack:3")

df_compare <- df[
    refine == FALSE &
        kernel %in% models,
    .(kernel, energy_score)
]

# compute mean and sd
summary_stats <- df_compare[
    ,
    .(
        mean_energy = mean(energy_score),
        sd_energy = sd(energy_score)
    ),
    by = kernel
]

print(summary_stats)

# effect size (Cohen-like)
mean_diff <- diff(summary_stats$mean_energy)
pooled_sd <- sqrt(mean(summary_stats$sd_energy^2))

effect_size <- mean_diff / pooled_sd

cat("Effect size (intrinsic units):", effect_size, "\n")

############################################################
## dominance probability
############################################################

a <- df_compare[kernel == "spack:3", energy_score]
b <- df_compare[kernel == "periodickernel", energy_score]

# Monte Carlo estimate
n <- min(length(a), length(b))

dom_prob <- mean(sample(a, n) > sample(b, n))

cat("Dominance probability (spack3 > periodic):", dom_prob, "\n")


############################################################
## free energy gap
############################################################

free_energy <- df[
    refine == FALSE &
        kernel %in% c("whitenoise", "periodickernel", "spack:3"),
    .(
        F = -log(mean(1 - u_cal))
    ),
    by = kernel
]

print(free_energy)

# pairwise differences
free_energy[, F_centered := F - F[kernel == "whitenoise"]]

print(free_energy)
