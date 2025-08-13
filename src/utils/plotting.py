"""Simple figure helper using gnuplotlib for Jupyter notebooks and exporting.

Hardcoded policy (simple & opinionated):
- Always export all formats: .gp, .pdf, .svg, .png
- Inline display uses the EXPORTED .svg
- Font is hardcoded to JuliaMono,12
- DPI is hardcoded to 300 for raster sizing
- Dimensions are hardcoded for thesis-friendly single-column figures:
    PDF: width = 120 mm, aspect = 0.62  (height = width*aspect)
    SVG/PNG: same physical size mapped to pixels via 300 DPI

Path logic:
- Assumes this is called from a Jupyter notebook.
- Uses os.getcwd() (the notebook directory) as the caller location.
- Writes to: $PROJECT_FIGURES_PATH / (cwd relative to $PROJECT_ROOT_PATH) / <export>
- Overwrites existing files with the same stem.

Env (required, no fallbacks):
    PROJECT_ROOT_PATH, PROJECT_FIGURES_PATH
"""

from __future__ import annotations

import os
from pathlib import Path

import gnuplotlib as gp
import jax
import numpy as np
from IPython.display import SVG, display

# Required env vars (no fallbacks)
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT_PATH"]).resolve()
FIGURES_PATH = Path(os.environ["PROJECT_FIGURES_PATH"]).resolve()
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
RC_FILE = (PROJECT_ROOT / ".gnuplot").resolve()

# Hardcoded style
_FONT = "JuliaMono,12"
_DPI = 300
_WIDTH_MM = 120.0
_ASPECT = 0.62


def iplot(
    *args,
    export: str
    | None = None,  # short plot name / stem; REQUIRED for uniqueness
    tmp: bool = False,  # write under /tmp/...
    **kwargs,
) -> None:
    """Plot with gnuplotlib, export to GP/PDF/SVG/PNG, and display SVG inline."""
    # Load rc (keeps styles centralized)
    if RC_FILE.exists():
        loadcmd = f"load '{RC_FILE.as_posix()}'"
        cmds = kwargs.get("cmds")
        kwargs["cmds"] = [loadcmd] + (
            [] if cmds is None else ([cmds] if isinstance(cmds, str) else cmds)
        )

    # JAX -> NumPy for array-likes (leave dicts/strings/etc. alone)
    converted_args = [
        np.asarray(arg) if isinstance(arg, jax.numpy.ndarray) else arg
        for arg in args
    ]

    # Geometry (keep identical across all terminals)
    w_in = _WIDTH_MM / 25.4
    h_in = w_in * _ASPECT
    px_w, px_h = int(round(w_in * _DPI)), int(round(h_in * _DPI))

    # Destination base from CWD (notebook directory)
    cwd = Path(os.getcwd()).resolve()
    try:
        rel_dir = cwd.relative_to(PROJECT_ROOT)
    except Exception:
        rel_dir = Path("unknown")

    root = Path("/tmp").resolve() if tmp else FIGURES_PATH
    dest_dir = (root / rel_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = export or "plot"
    base = dest_dir / stem

    # ---- Export bundle: GP, PDF, SVG, PNG ----
    gp.plot(*converted_args, hardcopy=str(base.with_suffix(".gp")), **kwargs)

    gp.plot(
        *converted_args,
        hardcopy=str(base.with_suffix(".pdf")),
        terminal=f"pdfcairo size {w_in}in,{h_in}in font '{_FONT}' enhanced",
        **kwargs,
    )

    gp.plot(
        *converted_args,
        hardcopy=str(base.with_suffix(".svg")),
        terminal=f"svg size 600,300 dynamic font '{_FONT}' enhanced",
        **kwargs,
    )

    gp.plot(
        *converted_args,
        hardcopy=str(base.with_suffix(".png")),
        terminal=f"pngcairo size 600,300 font '{_FONT}' enhanced",
        **kwargs,
    )

    # Inline preview: use the EXPORTED SVG (closest to thesis layout, vector)
    display(SVG(filename=str(base.with_suffix(".svg"))))


__all__ = ["iplot"]
