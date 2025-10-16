```dot
  digraph {
    graph [rankdir=LR, splines=false, bgcolor="transparent"];
    node  [shape=circle, fixedsize=true, width=0.30];
    edge  [dir=none, penwidth=0.8];

    h1 [label="t_+^n"];
    h2 [label="t_+^n"];
    h3 [label="..."];
    hK [label="t_+^n"];
    u [label="u'"];

    1 -> h1 [label="b_1"]
    t -> h1

    1 -> h2
    t -> h2 [label="c_H"]

    1 -> h3
    t -> h3 [label="c_h"]

    1 -> hK [label="b_2"]
    t -> hK

    h1 -> u [label="a_1"]
    h2 -> u [headlabel="a_H", labelangle=17, labeldistance=2.75];
    h3 -> u [headlabel="a_h", labelangle=14, labeldistance=2.75];
    hK -> u [label="a_2"]
  }
```
