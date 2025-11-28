// typst watch ./slides.typ --root ..
#import "@preview/polylux:0.4.0": *
#import "@preview/jotter-polylux:0.1.0": framed-block, post-it, setup, title-slide

#import "../thesis/lib/gnuplot.typ": gnuplot
#import "../thesis/lib/prelude.typ": bm

#set text(
  size: 21pt,
  font: "Kalam",
  fill: blue.darken(50%),
)

#show math.equation: set text(
  font: ("Pennstander Math", /* as a fallback: */ "New Computer Modern Math"),
  weight: "light",
)

#show raw: set text(font: "Fantasque Sans Mono")

#show: setup.with(
  header: [Linear surrogates via GPs],
  highlight-color: red,
  binding: true,
  dots: false, // set to true later, otherwise destroys preview
)

/*
ONS VERHAAL GEBRUIKEN toch

heel goede modellen nodig


ANKERPUNTEN om de zoveel slides

we willen van hier naar daar

combinatie van Rd = .3 tem 2.7



gotta tame the prior
too wild
but not too tame either



duidelijk in en uit verhaal


beeldjes van mensen die spreken
waveforms


laatste 5 minutes aan voorbeelden
- robotica
- signal processing
- chemie: batterij laad en oplaad curves
- vision
- echo's, sonar
curves, overal curves

hoeveel dimensions?


======

[
hyperparams: prior experience or data
then data comes: set your hyperparams and go
]

here

[
hyperparams: learn from examples
then data comes: lets go
]

*/

#title-slide[My interesting title][
  A subtitle

  The speaker

  Date, Place

  #place(
    horizon + right,
    post-it[
      #set align(horizon + left)
      #set text(size: .6em)
      Don't miss this talk!
    ],
  )
]

#slide[
  #image("./fig/test.svg", height: 100%)
]


#slide[
  = Problem: learn a surrogate

  #toolbox.side-by-side[
    Problem: got examples, want to learn the distribution for use in regression with "similar data"

    // is this unsupervised learning?

    Examples:
    - Surrogates for expensive simulations
    - others

    *NOT* the same as:
    - Extrapolation or interpolation: assume different data, not same
  ][
    #show: later
     == Approaches

    - Normalized flows
    - Few-shot learning in LLMs for very complex
    - (others)
    - This talk: cheap *linear* surrogates!
  ]
]

#slide[ 
  = Bayesian linear regression

  Got $N$ data tuples: ${y_(1:N), bold(x)_(1:N)}$

  Predict $y = f(bold(x)) + epsilon$, $epsilon ~ mono("Normal")(0, sigma^2)$: interpolate and extrapolate

  Model:
  $
    f(bold(x)) = bm(Phi) thin bm(a)
  $

  - Here $bm(Phi) in bb(R)^(N times M)$ are the $M$ _basis functions_ indexed at $bold(x)_n$:
  $
    [bm(Phi)]_(n m) = phi.alt_m (bold(x)_n)
  $

  - And $bm(a) in bb(R)^M$ are the _amplitudes with prior_ $bm(a) ~ mono("Normal")(bm(0), bm(Sigma_a))$

  - Inference is well-known: // @Murphy

  == Problem: how to choose?

  Assume we agree on the model and the white noise, how do we choose $bm(Phi)$ and $bm(Sigma_a)$?

  #post-it[Actually you can always choose $ bm(Sigma_a) := I_M $]

  This has a large influence!

  Examples:
  /*
    Same dataset with (2 phi's) times (2 Sigma_a's)
  */
]

#slide[
  == This slide changes!

  You can always see this.

  #show: later
  swerpifleps

  #show: later
  shweks
]

#slide[
  = Typography

  #toolbox.side-by-side[
    Style your content beautifully!

    Some text is *bold*, some text is _emphasized_.
  ][
    - a bullet point
    - another bullet point

    + first point
    + second point
  ]
]

#slide[
  = Maths and Code

  #toolbox.side-by-side[
    Maxwell says:
    $
      integral.surf_(partial Omega) bold(B) dot dif bold(S) = 0
    $
  ][
    Compute the answer:
    ```rust
    pub fn main() {
        dbg!(42);
    }
    ```
  ]
]

#slide[
  = Highlighting content

  // #framed-block and #post-it accept a sloppiness parameter that determine how
  // randomized they are.
  // #framed-block also accepts inset, width, and height like the standard
  // #block.

  #toolbox.side-by-side[
    #grid(
      columns: 2,
      gutter: 1em,
      framed-block[a], framed-block[couple],
      framed-block[of], framed-block[randomized],
      framed-block[framed], framed-block[boxes],
    )
  ][
    #box(post-it[a post-it])
    #box(post-it[another post-it])
  ]
]

#slide[
  #post-it[
    #image("../thesis/chapter/gfm/fig/glottal-cycle.png", height: 50%)
    Textit
  ]
]

#slide[
  #post-it(sloppiness: 0.2)[
    #v(1.1cm)
    #figure(
      gnuplot(read("../thesis/chapter/gfm/fig/lf.gp")),
      placement: auto,
    ) <fig:lf>
  ]

  last slide is here
  and here

  fast update
]

