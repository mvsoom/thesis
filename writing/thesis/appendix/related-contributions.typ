#import "/writing/thesis/lib/prelude.typ": bm

= Related contributions
<chapter:related-contributions>

*#cite(<VanSoom2019a>, form: "prose").* Measuring formant frequency $bm(F)$ and bandwidth $bm(B)$ from
steady-state vowel waveforms, taking into account the low-frequency
effects of the DGF $u' (t)$ during the open phase of the pitch period.
Inference is implemented using a maximum-a-posteriori (MAP)
approximation. In BNGIF these low-frequency effects are taken into
account automatically.

*#cite(<VanSoom2020>, form: "prose").*
Same as the above, but now using robust nested sampling inference
instead of a MAP approximation. A new heuristic derivation of the model
function from source-filter theory is added. In BNGIF nested sampling is
the preferred inference method.

*#cite(<VanSoom2022>, form: "prose").*
Theoretical derivation of a new weakly informative prior for resonance
frequencies. Shown to have superior performance in predicting
steady-state vowel waveforms compared to the usual priors in inferring
the pole frequency $bm(x)$ and bandwidth $bm(y)$, while being roughly
equally (un)informative.
