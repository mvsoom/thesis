#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
from jupyter_cache import get_cache
from rich.console import Console
from rich.table import Table
from rich.text import Text

CACHE_DIRNAME = ".jupyter_cache"
RUNS_DIRNAME = "runs"


def ascii_hist_log10(x, bins=10, width=30):
    counts, edges = np.histogram(x, bins=bins)
    m = counts.max() if counts.size else 1

    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(width * c / m) if m > 0 else ""
        lines.append(f"{lo:5.2f}–{hi:5.2f} | {bar}")
    return lines


def main(exp_dir):
    exp = Path(exp_dir).resolve()
    runs = exp / RUNS_DIRNAME
    cache_dir = exp / CACHE_DIRNAME

    if not cache_dir.exists():
        raise SystemExit("no .jupyter_cache found")

    cache = get_cache(cache_dir)
    console = Console()

    notebooks = sorted(runs.glob("*.ipynb"))
    proj_records = {
        Path(r.uri).resolve(): r for r in cache.list_project_records()
    }

    table = Table(title="Notebook execution status", show_lines=False)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Notebook", overflow="fold")
    table.add_column("Status", justify="center")
    table.add_column("Time (s)", justify="right")

    times = []
    n_done = n_todo = n_fail = 0

    for nb in notebooks:
        nb = nb.resolve()
        pr = proj_records.get(nb)

        nb_link = Text.from_markup(f"[link=file://{nb}]{nb.name}[/link]")

        if pr is None:
            table.add_row("-", nb_link, "-", "")
            n_todo += 1
            continue

        if pr.traceback:
            table.add_row(
                str(pr.pk),
                nb_link,
                "[red]✗[/red]",
                "",
            )
            n_fail += 1
            continue

        cr = cache.get_cached_project_nb(str(nb))
        if cr is None or not cr.data or "execution_seconds" not in cr.data:
            table.add_row(
                str(pr.pk),
                nb_link,
                "[yellow]-[/yellow]",
                "",
            )
            n_todo += 1
            continue

        secs = cr.data["execution_seconds"]
        times.append(secs)
        table.add_row(
            str(pr.pk),
            nb_link,
            f"[green]✓[/green] [{cr.pk}]",
            f"{secs:,.2f}",
        )
        n_done += 1

    console.print(table)

    total = len(notebooks)
    console.print()
    console.print(
        f"[bold]Summary[/bold]: "
        f"executed={n_done}, todo={n_todo}, failed={n_fail}, total={total}"
    )
    console.print("Legend: ✓ cached (hash match), - todo, ✗ failed\n")

    if not times:
        return

    t = np.asarray(times)
    logt = np.log10(t)

    med = np.median(logt)
    lo, hi = np.percentile(logt, [10, 90])

    eta = None
    if n_todo > 0:
        eta = 10 ** (med + math.log10(n_todo))

    console.print("[bold]Timing statistics (log10 domain)[/bold]")
    console.print(f"  median time     : {10**med:,.2f} s")
    console.print(f"  10–90% interval : [{10**lo:,.2f}, {10**hi:,.2f}] s")

    if eta is not None:
        console.print(
            f"  ETA ({n_todo} remaining) : {eta:,.2f} s (~{eta / 60:,.1f} min)"
        )

    console.print("\n[bold]log10 execution time histogram[/bold]")
    for line in ascii_hist_log10(logt):
        console.print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_folder", type=Path)
    args = ap.parse_args()
    main(args.experiment_folder)
