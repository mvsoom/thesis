# %%
library(ggplot2)
library(data.table)
library(plotly)
theme_set(theme_minimal())

# Get experiment dir from environment variable
experiments_dir <- Sys.getenv("PROJECT_EXPERIMENTS_PATH", unset = "experiments")
if (experiments_dir == "") experiments_dir <- "experiments"
matlab_runs_file <- file.path(experiments_dir, "iklp/egifa_matlab/runs.csv")
stationary_runs_file <- file.path(experiments_dir, "iklp/egifa_stationary/runs.csv")

matlab_runs <- data.table(fread(matlab_runs_file))
stationary_runs <- data.table(fread(stationary_runs_file))

matlab_df <- matlab_runs[
    ,
    .(
        collection = as.factor(collection),
        method = as.factor(method),
        score = sqrt(pmax(0, 1 - results.excitation_aligned_nrmse^2)), # cosine similarity
        utterance = results.wav,
        dataset = "egifa_matlab"
    )
]

stationary_df <- stationary_runs[
    ,
    .(
        collection = as.factor(collection),
        method = as.factor(kernel),
        score = sqrt(pmax(0, 1 - results.excitation_aligned_nrmse^2)), # cosine similarity
        utterance = results.wav,
        dataset = "egifa_stationary_previous"
    )
]

df <- rbindlist(list(matlab_df, stationary_df), use.names = TRUE)
df[, method := as.factor(as.character(method))]
df <- df[is.finite(score)]

# Select collection %in% ["vowel", "speech"]
# df <- df[collection == "vowel"]

# %%
# Score densities for all methods (combined datasets)
(
    ggplot(df) +
        geom_density(aes(x = score, fill = method), alpha = 0.3) +
        ggtitle("Scores per method (combined datasets)")
) |> ggplotly()

# %%
# A p-value less than 0.05 indicates that the row method tends to yield a higher score than the column method at the 5% significance level (paired one-sided Wilcoxon signed-rank test)
# Pair test on (median aggregate per utterance) like Chien+ (2017) Tables II & III
agg <- df[, .(score = median(score)), by = .(method, utterance)]

agg_wide <- dcast(agg, utterance ~ method, value.var = "score")

agg_scores <- as.matrix(agg_wide[, -1])
methods <- colnames(agg_scores)

agg_p_mat <- matrix(1, length(methods), length(methods),
    dimnames = list(methods, methods)
)

for (i in seq_along(methods)) {
    for (j in seq_along(methods)) {
        if (i == j) next

        x <- agg_scores[, i]
        y <- agg_scores[, j]
        ok <- is.finite(x) & is.finite(y)

        if (sum(ok) == 0) {
            agg_p_mat[i, j] <- NA_real_
            next
        }

        agg_p_mat[i, j] <- wilcox.test(
            x[ok],
            y[ok],
            paired = TRUE,
            alternative = "greater" # A > B
        )$p.value
    }
}

# compute performance statistic per method
perf <- agg[, .(perf = median(score)), by = method]

# order methods from best to worst
ord <- perf[order(-perf)]$method

# reorder matrix
agg_p_mat_sorted <- agg_p_mat[ord, ord]

print(round(agg_p_mat_sorted, 3))

# This survives all kinds of extra conservative assumptions
# So pack:1 is best
