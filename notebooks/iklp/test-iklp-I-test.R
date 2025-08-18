library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "iklp-openglot-I-test/runs.csv")
runs <- data.table(fread(runs_file))

runs[, pitch_std := sqrt(frame_stats.estimated_pitch_var)]
runs[, `:=`(pitch_min = frame_stats.estimated_pitch_mean - pitch_std,
           pitch_max = frame_stats.estimated_pitch_mean + pitch_std)]

# calculate error
runs[, pitch_error := abs(frame_stats.estimated_pitch_mean - true_pitch)]

runs[, best_of := frame_stats.score == max(frame_stats.score), by = .(wav_file, frame_stats.frame_index)]


# When initial_pitchedness is > 0.5, they all converge to pitchedness of 1
# When initial_pitchedness is < 0.5, some still converge to pitchedness of 1
ggplot(runs, aes(x = frame_stats.pitchedness, y = initial_pitchedness, color=wav_file)) +
geom_jitter(width = 0.05, size=4) +
  theme_gnuplot()

ggplot(runs, aes(x = initial_pitchedness, 
         y = pitch_error/true_pitch, color=wav_file, size=best_of, shape=best_of)) +
  geom_jitter() +
  geom_point(size=4) +
  scale_x_continuous(minor_breaks = waiver()) +
  scale_y_continuous(minor_breaks = waiver()) +
  theme_gnuplot()

ggplot(runs, aes(x = frame_stats.pitchedness, y=frame_stats.initial_pitchedness, color=wav_file)) +
  geom_jitter(width = 0.05, size=4) +
  theme_gnuplot()

# std


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
