# %%
library(ggplot2)
library(data.table)
library(plotly)
theme_set(theme_minimal())

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "iklp/egifa_matlab/runs.csv")
runs <- data.table(fread(runs_file))

summary(runs)

id_cols <- c(
    "results.wav",
    "results.frame_index",
    "results.voiced_group"
)

runs[, id := do.call(paste, c(.SD, sep = "|")), .SDcols = id_cols]

df <- runs[
    ,
    .(
        collection = as.factor(collection),
        method = as.factor(method),
        name = as.factor(results.name),
        egifa_f0 = egifa_f0,
        lag_est = results.lag_est,
        oq_true = results.oq_true,
        pitch_true = results.pitch_true,
        score = sqrt(pmax(0, 1 - results.excitation_aligned_nrmse^2)), # cosine similarity
        utterance = results.wav,
        id
    ),
]

# %%

# Select collection %in% ["vowel", "speech"]
# df <- df[collection == "speech"]


# %%

# Plots

(
    ggplot(df) +
        geom_density(aes(x = score, fill = method), alpha = 0.3) +
        ggtitle("Scores per method")
) |> ggplotly()

# %%
# vowel vs speech
(
    ggplot(df) +
        geom_density(aes(x = score, fill = collection), alpha = 0.3) +
        facet_wrap(~method) +
        ggtitle("Scores per collection")
) |> ggplotly()


# %%

# (score, oq_true) distribution per method
# Strong influence compared to egifa_stationary kernels!
df_small <- df[sample(.N, min(.N, 5000))]

(
    ggplot(df_small) +
        geom_point(aes(score, oq_true, color = method), alpha = 0.5) +
        ggtitle("Score vs OQ_true (downsampled)")
) |> ggplotly()

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

# %%
# A p-value less than 0.05 indicates that the row kernel tends to yield a higher score than the column kernel at the 5% significance level (paired one-sided Wilcoxon signed-rank test)
# Pair test on (frame level)
wide <- dcast(df, id ~ method, value.var = "score")

scores <- as.matrix(wide[, -1])

methods <- colnames(scores)

p_mat <- matrix(1, length(methods), length(methods),
    dimnames = list(methods, methods)
)

for (i in seq_along(methods)) {
    for (j in seq_along(methods)) {
        if (i == j) next

        p_mat[i, j] <- wilcox.test(
            scores[, i],
            scores[, j],
            paired = TRUE,
            alternative = "greater" # A > B
        )$p.value
    }
}

print(round(p_mat, 3))

# %%
# Pair test on (median aggregate per utterance) like Chien+ (2017) Tables II & III
agg <- df[, .(score = median(score)), by = .(method, utterance)]

agg_wide <- dcast(agg, utterance ~ method, value.var = "score")

agg_scores <- as.matrix(agg_wide[, -1])

methods <- colnames(scores)

agg_p_mat <- matrix(1, length(methods), length(methods),
    dimnames = list(methods, methods)
)
for (i in seq_along(methods)) {
    for (j in seq_along(methods)) {
        if (i == j) next

        agg_p_mat[i, j] <- wilcox.test(
            agg_scores[, i],
            agg_scores[, j],
            paired = TRUE,
            alternative = "greater" # A > B
        )$p.value
    }
}

print(round(agg_p_mat, 3))

# %%
# very stable for pack:1,2,3 and periodickernel
thresh <- 0.99
(
    ggplot(
        df[u_cal > thresh, .(lag_est), by = kernel]
    ) +
        geom_density(aes(x = lag_est, fill = kernel), alpha = 0.3) +
        ggtitle("lag_est distribution for different kernels")
) |> ggplotly()
