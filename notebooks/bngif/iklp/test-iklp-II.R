library(ggplot2)
library(ggnuplot)
library(data.table)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "iklp-openglot-II/runs.csv")
runs <- data.table(fread(runs_file))


ggplot(runs, aes(x = file_stats.pitchedness, 
         y = file_stats.pitch_wmae/true_pitch, 
         color = file_stats.num_iters)) +
  geom_point(size=4) +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_viridis_c(option = "C", guide = guide_colorbar(title = "Num Iters")) +
  theme_gnuplot()

ggplot(runs, aes(x = file_stats.pitchedness, 
                 y = true_pitch, 
                 color = modality)) +
  scale_x_log10() +
  scale_y_log10() +
  theme_gnuplot()

ggplot(runs, aes(x = true_pitch, 
                 y = file_stats.pitch_wmae, 
                 color = modality)) +
  geom_jitter(width = 0.05, alpha = 0.3, size=4) +
  scale_x_log10() +
  scale_y_log10() +
  theme_gnuplot()

ggplot(runs, aes(x = file_stats.pitchedness)) +
  stat_ecdf(geom = "step") +
  ylab("Cumulative Probability") +
  theme_gnuplot()


ggplot(runs, aes(x = file_stats.pitch_wmae/true_pitch)) +
  stat_ecdf(geom = "step") +
  ylab("Cumulative Probability") +
  theme_gnuplot()
