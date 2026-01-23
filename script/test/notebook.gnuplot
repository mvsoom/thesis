set term svg size 900,600 font "Libertinus Serif,16"
set output "test.svg"

set encoding utf8
set grid xtics ytics lc rgb "#cccccc" dt 3

set tics in
set key top right

set xlabel "x"
set ylabel "y"

plot "< ./readplotly notebook.ipynb 0 0 x y"