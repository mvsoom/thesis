# This is read and simply prepended before any other commands
set encoding utf8

fontspec = "Libertinus Serif,16"

set_plot_size(w,h) = sprintf("set term svg size %d,%d font '%s'", w, h, fontspec)

set linetype 1 lc rgb "blue" pt 3
set linetype 2 lc rgb "red" pt 4
set linetype 3 lc rgb "green" pt 6
set linetype 4 lc rgb "black" pt 12
set linetype 5 lc rgb "blue" pt 5
set linetype 6 lc rgb "red" pt 1
set linetype 7 lc rgb "green" pt 2
set linetype 8 lc rgb "black" pt 7
set linetype cycle 8
set style data lines

set key noautotitle

set auto fix

# offsets
set offsets graph .1, graph .1, graph .1, graph .1

set key samplen 2

set xlabel "time" offset 0, 1
set ylabel "amplitude" offset 1,0

set tics in scale 0.5,0.3
set xtics offset 0,0.3 autofreq
set ytics offset 0.1,0 autofreq

set grid xtics ytics lc rgb "#cccccc" dt 3