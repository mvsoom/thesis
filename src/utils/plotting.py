"""Simple figure helper using gnuplotlib for Jupyter notebooks and exporting"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gnuplotlib as gp
import jax
import numpy as np
from IPython.display import SVG, display

FIGURES_PATH = Path(os.getenv("PROJECT_FIGURES_PATH", "figures")).resolve()
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

RC_FILE = (Path(os.getenv("PROJECT_ROOT_PATH")) / ".gnuplot").resolve()


def iplot(
    *args,
    export: str | None = None,
    **kwargs,
) -> None:
    """
    Inline plot for Jupyter using gnuplotlib. Uses `.gnuplot` file at project root for config and tries to convert Jax arrays to numpy arrays.

    Args:
        *args, **kwargs: forwarded to gnuplotlib.plot
        export: optional path + stem for saving files under PROJECT_FIGURES_PATH/{export}.{pdf,gp}
    """
    if RC_FILE.exists():
        loadcmd = f"load '{RC_FILE.as_posix()}'"
        cmds = kwargs.get("cmds")
        if cmds is None:
            kwargs["cmds"] = loadcmd
        else:
            # Prepend to any existing cmds (string or list)
            if isinstance(cmds, str):
                kwargs["cmds"] = [loadcmd, cmds]
            else:
                kwargs["cmds"] = [loadcmd, *cmds]

    converted_args = [
        np.asarray(arg) if isinstance(arg, jax.numpy.ndarray) else arg for arg in args
    ]

    if export:
        base = FIGURES_PATH / Path(export)
        base.parent.mkdir(parents=True, exist_ok=True)
        print(base)

        # Save as PDF and GP
        gp.plot(*converted_args, hardcopy=str(base.with_suffix(".gp")), **kwargs)
        gp.plot(
            *converted_args,
            hardcopy=str(base.with_suffix(".pdf")),
            terminal="pdfcairo font FONT",
            **kwargs,
        )

    fd, path = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    gp.plot(*converted_args, hardcopy=path, terminal="svg font FONT", **kwargs)
    display(SVG(filename=path))


__all__ = ["iplot"]
