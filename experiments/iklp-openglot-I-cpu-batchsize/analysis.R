library(ggplot2)
library(ggnuplot)
library(data.table)
library(ggrepel)

# Get experiment dir from environment variable
runs_file <- file.path(Sys.getenv("PROJECT_EXPERIMENTS_PATH"), "iklp-openglot-I-cpu-batchsize/runs.csv")
runs <- data.table(fread(runs_file))



df = runs[, .(time_per_iter, batch_size, jax_enable_x64, jax_platform_name, r)]

View(df)

ggplot(df, aes(x = batch_size, y = time_per_iter, color = factor(r), shape=jax_enable_x64, group=jax_platform_name)) +
  geom_point(size=4) +
  geom_line(aes(group=r)) +
  facet_wrap(~ jax_enable_x64) +
    scale_y_log10() +
  theme_gnuplot()


# Check x32 vs x64 speedup
# compute speedup x32/x64, aggregating (mean) over all columns except r and batch_size
agg_cols <- c("r", "batch_size", "jax_enable_x64")
df_agg <- df[, .(time_per_iter = mean(time_per_iter)), by = agg_cols]

# pivot so FALSE/TRUE become columns, then rename and compute speedup
df_wide <- dcast(df_agg, r + batch_size ~ jax_enable_x64, value.var = "time_per_iter")
setnames(df_wide, old = c("FALSE", "TRUE"), new = c("time_x32", "time_x64"))
df_wide[, speedup := time_x64/time_x32]

print(df_wide)


# Regression: log10(time_per_iter) ~ batch_size + r + jax_enable_x64
# prepare data.table, compute log10 response and ensure jax_enable_x64 is a factor
dt <- copy(df)
dt[, log_time := log10(time_per_iter)]
dt[, jax_enable_x64 := factor(jax_enable_x64, levels = c("FALSE", "TRUE"))]

# fit linear model using data.table-prepared data
model <- lm(log_time ~ batch_size + r + jax_enable_x64, data = dt)

# show standard summary
print(summary(model))

# present coefficients in a data.table and add multiplicative interpretation on original scale:
# a coefficient b on log10 scale corresponds to a multiplicative factor 10^b on time_per_iter
coefs <- coef(summary(model))
coef_dt <- data.table(
  term = rownames(coefs),
  estimate = coefs[, "Estimate"],
  std_error = coefs[, "Std. Error"],
  t_value = coefs[, "t value"],
  p_value = coefs[, "Pr(>|t|)"]
)
coef_dt[, multiplicative := 10^estimate]

print(coef_dt)
