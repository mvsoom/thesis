library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "iklp-openglot-I-test/runs.csv")
runs <- data.table(fread(runs_file))

ggplot(runs, aes(x = frame_stats.pitchedness, fill=wav_file)) +
  stat_bin() +
  theme_gnuplot()

ggplot(runs, aes(x = frame_stats.pitchedness, y=frame_stats.initial_pitchedness, color=wav_file)) +
  geom_jitter(width = 0.05, size=4) +
  theme_gnuplot()

# std
runs[, pitch_std := sqrt(frame_stats.estimated_pitch_var)]
runs[, `:=`(pitch_min = frame_stats.estimated_pitch_mean - pitch_std,
           pitch_max = frame_stats.estimated_pitch_mean + pitch_std)]

# calculate error
runs[, pitch_error := abs(frame_stats.estimated_pitch_mean - true_pitch)]

ggplot(runs, aes(x = frame_stats.pitchedness, 
         y = pitch_error, color=wav_file)) +
  geom_jitter() +
  geom_point(size=4) +
  theme_gnuplot()

# Get

ggplot(runs, aes(x = file_stats.pitchedness, 
                 y = true_pitch, 
                 color = modality)) +
  geom_jitter(width = 0.05, alpha = 0.3, size=4) +
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
