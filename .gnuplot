# From Gaston.jl docs <https://mbaz.github.io/Gaston.jl/stable/#Gnuplot-configuration-1>
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
set offsets graph .05, graph .05, graph .05, graph .05

set grid
set termoption dashed

# Can't set font for each terminal, has to be done on invoke
FONT = "JuliaMono,12"