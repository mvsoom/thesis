#import "lib/bye-ubc.typ": thesis

#import "@preview/diagraph:0.3.6"
#show raw.where(lang: "dot"): it => diagraph.render(it.text)

#import "lib/gnuplot.typ": gnuplot
#import "lib/prelude.typ": bm

#set figure.caption(separator: [ | ])

#show heading: set block(above: 1.4em, below: 1em)

// Latex \par{} style
#show heading.where(level: 4): it => {
  // inline, bold, body-sized; add trailing period and a little space
  text(size: 1em, weight: "bold")[
    #it.body
  ]
  h(0.5em)
}

#set outline(depth: 2)

#set cite(style: "american-psychological-association")

#show: thesis.with(
  // Avoid using scientific symbols or Greek letters; spell out the words.
  title: [
    Bayesian Nonparametric Glottal Inverse Filtering
  ],
  // Should be the name under which you are registered at UBC.
  author: "Marnix Van Soom",
  // Optional to list these. If listed, must have a degree (abbreviated, e.g.
  // MSc, BSc), institution and graduation year.
  previous-degrees: (
    /*
    (
      abbr: "MPhys",
      institution: "University of Manchester",
      year: "2020",
    ),
    (
      abbr: "BSc",
      institution: "Another University",
      year: "2017",
    ),
    */
  ),
  // Spell out degree in full e.g. Doctor of Philosophy, Master of Science,
  // Master of Arts.
  degree: "Doctor of Philosophy",
  // The specific graduate program in which you are registered. Check
  // SSC/Workday to confirm your program name.
  program: "Physics",
  // Vancouver or Okanagan.
  campus: "Vancouver",
  // The month and year in which your thesis is accepted.
  month: "February",
  year: "2026",
  // Include all committee members. For supervisory committee members who were
  // not part of the examining committee, include them below under
  // `additional-committee`.
  // Adding the external examiner is optional. Ask them whether or not they wish
  // to be listed in the committee page.
  examining-committee: (
    (
      name: "John Doe",
      title: "Research Scientist",
      department: "Physical Sciences Division",
      institution: "TRIUMF",
      role: "Research Co-supervisor",
    ),
    (
      name: "Jane Doe",
      title: "Professor",
      department: "Department of Chemistry",
      institution: "UBC",
      role: "Academic Co-supervisor",
    ),
  ),
  additional-committee: (),
  // Feel free to do these in whatever way works better for you. You can even
  // write these sections directly in [...] content blocks here instead (like
  // the title above).
  abstract: include "./meta/abstract.typ",
  lay-summary: include "./meta/lay_summary.typ",
  preface: include "./meta/preface.typ",
  // These are optional. You can:
  // - Delete these lines if you want to omit them.
  // - Write them by hand either here in a content block or in a separate file.
  // - Use e.g. the `glossarium` or similar packages to handle these for you and
  //   include the generating functions here.
  list-of-symbols: none,
  glossary: none,
  acknowledgments: include "./meta/acknowledgments.typ",
  dedication: none,
  bibliography: bibliography("library.bib", style: "apa"),
  // Also optional. If you don't have any appendices, you can delete this.
  // Same as all other sections, you can just include the content here from a
  // separate file.
  appendices: [
    #include "./appendix/regularization.typ"
    #include "./appendix/parametric.typ"
    #include "./appendix/gps.typ"
    #include "./appendix/related-contributions.typ"
  ],
)

#set heading(numbering: "1.1.1")

#set heading(numbering: "1.", supplement: "Chapter")

#set math.equation(
  numbering: "(1)",
  supplement: none,
)
#show ref: it => {
  // provide custom reference for equations
  if it.element != none and it.element.func() == math.equation {
    // optional: wrap inside link, so whole label is linked
    link(it.target)[(#it)]
  } else {
    it
  }
}

#include "./chapter/overview/main.typ"

#include "./chapter/iklp/main.typ"

#include "./chapter/gfm/main.typ"

#include "./chapter/spectral/main.typ"

= Learning a nonparametric glottal flow prior

= But what about $p(bm(a))$?
<chapter:arprior>

= Evaluation
<chapter:evaluation>

/*
Gridding over OQ also done in @Fu2006
*/

= Conclusion
<chapter:conclusion>

/*
From @Drugman2019

Glottal characterization has also been shown to be helpful in another
biomedical problem: the classification of clinical depression in speech. In
(Ozdas et al. (2004))
*/
