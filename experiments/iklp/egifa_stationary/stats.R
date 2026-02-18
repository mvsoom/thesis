# %%
library(ggplot2)
library(data.table)
library(plotly)
library(GGally)
theme_set(theme_minimal())

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "iklp/egifa_stationary/runs.csv")
runs <- data.table(fread(runs_file))

summary(runs)

id_cols <- c(
    "results.wav",
    "results.frame_index",
    "results.index",
    "results.restart_index"
)

runs[, id := do.call(paste, c(.SD, sep = "|")), .SDcols = id_cols]

df <- runs[
    ,
    .(
        collection = as.factor(collection),
        kernel = as.factor(kernel),
        name = as.factor(results.name),
        egifa_f0 = egifa_f0,
        I_eff = results.I_eff,
        SNR_db = results.SNR_db,
        lag_est = results.lag_est,
        oq_true = results.oq_true,
        pitch_true = results.pitch_true,
        pitch_wrmse = results.pitch_wrmse,
        score = sqrt(pmax(0, 1 - results.source_aligned_nrmse^2)), # cosine similarity
        id
    ),
]

# FIXME: only test completed so far
# df <- df[collection == "vowel"]


# %%

# Plots

(
    ggplot(df) +
        geom_density(aes(x = score, fill = kernel), alpha = 0.3)
) |> ggplotly()

# (score, oq_true) distribution per kernel
# little influence
(
    ggplot(df) +
        geom_point(aes(x = score, y = oq_true, color = kernel), alpha = 0.5) +
        ggtitle("Score vs OQ_true, colored by kernel")
) |> ggplotly()

# Looks like bimodality might be an artifact from affine+shift equivalence
# CONFIRMED via null score below

# %%

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

# compare calibrated distributions across kernels
(
    ggplot(df) +
        geom_density(aes(x = u_cal, fill = kernel), alpha = 0.3) +
        ggtitle("CDF-calibrated score (u-space)")
) |> ggplotly()

# %%
# A p-value less than 0.05 indicates that the row kernel tends to yield a higher score than the column kernel at the 5% significance level (paired one-sided Wilcoxon signed-rank test)
# Same comparison as in Chien+ (2017) Tables II & III
wide <- dcast(df, id ~ kernel, value.var = "score")

scores <- as.matrix(wide[, -1])

kernels <- colnames(scores)

p_mat <- matrix(1, length(kernels), length(kernels),
    dimnames = list(kernels, kernels)
)

for (i in seq_along(kernels)) {
  for (j in seq_along(kernels)) {

    if (i == j) next

    p_mat[i, j] <- wilcox.test(
        scores[, i],
        scores[, j],
        paired = TRUE,
        alternative = "greater"  # A > B
    )$p.value
  }
}

print(round(p_mat, 3))

# %%
# dominance curve
# periodickernel is dominated everywhere by pack:1 and pack:2
# pack:1 and pack:2 cross, so
# pack:1 is more reliable overall [better typical performance]
# pack:2 produces more extreme top performances (starting from 65% threshold) [better best-case performanceþ
# on wilcoxon pack:1 wins because it improves more often.
# "While pack:1 exhibits statistically stronger overall performance, pack:2 achieves superior performance in the extreme high-quality regime."

thresholds <- seq(0, 1, length.out = 200)

dom <- df[, .(
    threshold = thresholds,
    dominance = sapply(thresholds, function(t) {
        mean(u_cal > t)
    })
), by = kernel]

(
    ggplot(dom, aes(threshold, dominance, colour = kernel)) +
        geom_line() +
        labs(
            x = "u_cal threshold",
            y = "P(u_cal > threshold)",
            title = "Dominance curves relative to null"
        )
) |> ggplotly()
