import os
import time
from pathlib import Path

import dill as pickle
import matplotlib as mpl
from IPython.display import display

# Enforce PGF globally (must happen before importing pyplot)
mpl.use("pgf")

from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("png")  # notebook display as PNG

import matplotlib.pyplot as _plt
import matplotlib.style as _style

_style.use("utils.plots_sans")  # loads fonts/preamble/size/etc.
plt = _plt

_FIGDIR = Path(os.environ.get("PROJECT_FIGURES_PATH", ".")).resolve()
_FIGDIR.mkdir(parents=True, exist_ok=True)


def _ts_stem():
    t = time.time()
    s = time.strftime("%Y%m%d%H%M%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)
    return f"{s}{ms:03d}"


def _any_exists(stem, exts):
    for ext in exts:
        if (_FIGDIR / ext / f"{stem}.{ext}").exists():
            return True
    if (_FIGDIR / "pkl" / f"{stem}.pkl").exists():
        return True
    return False


def _uniq_stem(base, exts):
    stem = base
    n = 1
    while _any_exists(stem, exts):
        stem = f"{base}-{n:02d}"
        n += 1
    return stem


def _show_links(paths):
    try:
        from IPython.display import HTML, display
    except Exception:
        return
    items = []
    cwd = os.getcwd()
    for k in ("pdf", "png", "svg", "pickle"):
        p = paths.get(k)
        if not p:
            continue
        rel = os.path.relpath(p, cwd)
        items.append('<li><b>%s</b>: <a href="%s">%s</a></li>' % (k, rel, rel))
    if items:
        display(HTML("<ul>%s</ul>" % "".join(items)))


# size presets (width, height) in inches
_FIG_SIZES = {
    "1col": (5.8, 3.0),  # paragraph width, ~20% shorter height
    "2col": (3.0, 1.8),  # half width
    "3col": (2.0, 1.3),  # third width
}


def retain(
    fig,
    stem=None,
    *,
    formats=("pdf", "png", "svg"),
    keep_pickle=True,
    show=True,
    links=True,
    col=None,  # '1col'|'2col'|'3col'
    width=None,  # explicit width in inches
    height=None,  # explicit height in inches
    transparent=True,
):
    """Save to PROJECT_FIGURES_PATH/<ext>/<stem>.<ext> (+ pkl/<stem>.pkl)

    Args:
      col: choose a preset size ('1col','2col','3col').
      width, height: explicit overrides (in inches).
    """
    # adjust figure size if requested
    if col in _FIG_SIZES:
        fig.set_size_inches(*_FIG_SIZES[col])
    if width or height:
        w, h = fig.get_size_inches()
        fig.set_size_inches(width or w, height or h)

    base = _uniq_stem(stem or _ts_stem(), formats)
    out = {}

    for ext in formats:
        (_FIGDIR / ext).mkdir(parents=True, exist_ok=True)
        path = _FIGDIR / ext / f"{base}.{ext}"
        fig.savefig(path, transparent=transparent)
        out[ext] = str(path)

    if keep_pickle:
        (_FIGDIR / "pkl").mkdir(parents=True, exist_ok=True)
        pkl = _FIGDIR / "pkl" / f"{base}.pkl"
        with pkl.open("wb") as f:
            pickle.dump(fig, f)
        out["pickle"] = str(pkl)

    if links:
        _show_links(out)
    if show:
        display(fig)

    return out


def reopen(stem_or_path):
    p = Path(stem_or_path)
    if p.suffix == ".pkl" and p.exists():
        target = p
    else:
        target = _FIGDIR / "pkl" / (p.stem + ".pkl")
    with target.open("rb") as f:
        fig = pickle.load(f)
    return fig
