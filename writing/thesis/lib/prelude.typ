#let bm(x) = math.bold(math.upright(x))

#let pcite(key, ..args) = cite(key, form: "prose", ..args)

#let section-title-page(label) = context {
  let els = query(label)
  if els.len() == 0 { "" } else {
    let el = els.first()
    if el.func() == heading {
      [#link(el.location(), text(el.body, weight: "bold")) (#ref(label, form: "page"))]
    }
  }
}