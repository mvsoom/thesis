library(ggplot2)
library(plotly)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "lvm/pack/runs.csv")
runs <- data.table(fread(runs_file))



quick <- runs[!is.na(results.D_KL_bans_per_sample), .(M, Q, results.K, results.D_KL_bans_per_sample)]
quick <- unique(quick)
setorder(quick, results.D_KL_bans_per_sample)
quick

View(quick)

# along line you can see influence of increasing M => has local optima
(
    ggplot(
        quick,
        aes(
            M * results.K,
            results.D_KL_bans_per_sample,
            color = as.factor(Q),
            text = paste0("M = ", M, "<br>K = ", results.K)
        )
    ) +
        geom_point() +
        geom_line(aes(group = interaction(Q, results.K)), alpha = 0.3) +
        scale_x_log10() +
        xlab("compute (M x K)") +
        ylab("normalized D_KL (bans/sample)") +
        ggtitle(
            "Performance on test set vs compute (M x K) [M increases along lines]"
        )
) |> ggplotly(tooltip = "text")


# along line you can see influence of increasing K => small influence
(
    ggplot(
        quick,
        aes(
            M * results.K,
            results.D_KL_bans_per_sample,
            color = as.factor(Q),
            text = paste0("M = ", M, "<br>K = ", results.K)
        )
    ) +
        geom_point() +
        geom_line(aes(group = interaction(Q, M)), alpha = 0.3) +
        scale_x_log10() +
        xlab("compute (M x K)") +
        ylab("normalized D_KL (bans/sample)") +
        ggtitle(
            "Performance on test set vs compute (M x K) [K increases along lines]"
        )
) |> ggplotly(tooltip = "text")


## Increasing Q is a "free" knob
(
    ggplot(
        quick,
        aes(
            M * results.K,
            results.D_KL_bans_per_sample,
            text = paste0("M = ", M, "<br>K = ", results.K)
        )
    ) +
        geom_point(aes(color = as.factor(Q))) +
        geom_line(aes(group = interaction(results.K, M)), alpha = 0.3) +
        scale_x_log10() +
        xlab("compute (M x K)") +
        ylab("normalized D_KL (bans/sample)") +
        ggtitle(
            "Performance on test set vs compute (M x K)"
        )
) |> ggplotly(tooltip = "text")
