#import "@preview/neoplot:0.0.3" as gp

#let _gp_style = read("style.gnuplot")

#let gnuplot(script) = gp.exec("reset\n" + _gp_style + "\n" + script)
