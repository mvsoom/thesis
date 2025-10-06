eval set_plot_size(200, 175)

unset key
set samples 1000
set xrange [-1.1:1.1]
set yrange [-.1:1.1]

set xtics ("-1" -1, "0" 0, "1" 1)
set ytics ("0" 0, "0.5" 0.5, "1" 1)

set tics scale 0     # no tick marks, labels only

set lmargin screen 0.15
set rmargin screen 0.85
set bmargin screen 0.20
set tmargin screen 0.98

plot (x>0 ? x**3 : 0) lw 2