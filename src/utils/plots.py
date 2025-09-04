import os
import time
from pathlib import Path

import dill as pickle
import matplotlib as mpl
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("png")  # display in notebook as PNG

import matplotlib.pyplot as _plt
import matplotlib.style as _style

_style.use("utils.plots")

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
        if (_FIGDIR / f"{stem}.{ext}").exists():
            return True
    if (_FIGDIR / f"{stem}.pkl").exists():
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


def retain(
    fig,
    stem=None,
    *,
    formats=("pdf", "png", "svg"),
    keep_pickle=True,
    show=True,
    links=True,
):
    """
    Save to PROJECT_FIGURES_PATH with identical stem across {pdf,png,svg} (+ .pkl).
    - pdf/png: render with LaTeX (text.usetex=True).
    - svg: no usetex, keep text as text (svg.fonttype='none') for Typst to substitute.
    - DPI/tight/etc. come from your style (plots.mplstyle).
    """
    base = _uniq_stem(stem or _ts_stem(), formats)
    out = {}

    for ext in formats:
        (_FIGDIR / ext).mkdir(parents=True, exist_ok=True)
        path = _FIGDIR / ext / f"{base}.{ext}"
        if ext in ("pdf", "png"):
            with mpl.rc_context(
                {"text.usetex": True, "text.parse_math": False}
            ):
                fig.savefig(path)
        elif ext == "svg":
            with mpl.rc_context(
                {
                    "text.usetex": False,
                    "text.parse_math": False,  # disable mathtext so text stays text
                    "svg.fonttype": "none",
                }
            ):
                fig.savefig(path)
        else:
            fig.savefig(path)
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
        plt.show()

    return out


def reopen(stem_or_path):
    p = Path(stem_or_path)
    if p.suffix != ".pkl":
        p = _FIGDIR / (p.stem + ".pkl")
    with p.open("rb") as f:
        fig = pickle.load(f)
    return fig
