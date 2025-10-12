```dot
  digraph {
    graph [rankdir=LR, splines=false, bgcolor="transparent", ranksep=0.7];
    node  [shape=circle, fixedsize=true, width=0.30];
    edge  [dir=none, penwidth=0.8];

    h1 [label="t_+^0"];
    h2 [label="t_+^0"];
    u [label="u'"];

    1 -> h1 [headlabel="0", labelangle=-16, labeldistance=3.2];
    t -> h1 [headlabel="1", labelangle=9, labeldistance=4.5];

    1 -> h2 [headlabel="-t_p", labelangle=-9, labeldistance=4.3];
    t -> h2 [headlabel="1", labelangle=13, labeldistance=3.5];

    h1 -> u [label="a_1"]
    h2 -> u [label="a_2"]
  }
```
