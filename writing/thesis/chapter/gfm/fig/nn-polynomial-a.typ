```dot
  digraph {
    graph [rankdir=LR, splines=false, bgcolor="transparent"];
    node  [shape=circle, fixedsize=true, width=0.30];
    edge  [dir=none, penwidth=0.8];

    h1 [label="t_+^0"];
    h2 [label="t_+^0"];
    u [label="u'"];

    1 -> h1 [label="-t_o"]
    t -> h1

    1 -> h2 [label="-t_m"]
    t -> h2

    h1 -> u [label="a_1"]
    h2 -> u [label="a_2"]
  }
```
