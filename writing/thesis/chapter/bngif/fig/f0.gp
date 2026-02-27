eval set_plot_size(400, 230)

unset key
set samples 2000

# ---- model parameters ----
mu_T = log(0.0067)     # mean log-period (T0 ≈ 6.7 ms → 150 Hz)
sigma = 0.25          # std of log-period

# since f0 = 1/T:
# log f0 ~ Normal(-mu_T, sigma^2)
mu_f = -mu_T

# ---- axes ----
set xrange [50:350]
set yrange [0:*]

set ylabel "density"
unset ytics
unset mytics
set format y ""

set xlabel "fundamental frequency [Hz]" offset 0,0.9
set xtics 50

set tics scale 0

lognormal_f0(x) = (x > 0) \
  ? (1.0 / (x * sigma * sqrt(2*pi))) \
    * exp(-(log(x) - mu_f)**2 / (2*sigma**2)) \
  : 0

plot lognormal_f0(x) lw 2