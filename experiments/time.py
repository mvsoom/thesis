#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
from jupyter_cache import get_cache
from rich.console import Console
from rich.table import Table

CACHE_DIRNAME = ".jupyter_cache"
RUNS_DIRNAME = "runs"


def ascii_hist_seconds(logx, bins=10, width=30):
    counts, edges = np.histogram(logx, bins=bins)
    m = counts.max() if counts.size else 1

    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(width * c / m) if m > 0 else ""
        lo_s = 10**lo
        hi_s = 10**hi
        lines.append(f"{lo_s:8.0f}–{hi_s:8.0f} s | {bar}")
    return lines


def pretty_path(p: Path) -> str:
    try:
        return str(p).replace(str(Path.home()), "~", 1)
    except Exception:
        return str(p)


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

    rows = []
    times = []
    n_done = n_todo = n_fail = 0

    for nb in notebooks:
        nb = nb.resolve()
        pr = proj_records.get(nb)

        if pr is None:
            rows.append((float("inf"), "-", nb, "-", None))
            n_todo += 1
            continue

        if pr.traceback:
            rows.append((pr.pk, pr.pk, nb, "fail", None))
            n_fail += 1
            continue

        cr = cache.get_cached_project_nb(str(nb))
        if cr is None or not cr.data or "execution_seconds" not in cr.data:
            rows.append((pr.pk, pr.pk, nb, "todo", None))
            n_todo += 1
            continue

        secs = cr.data["execution_seconds"]
        times.append(secs)
        rows.append((pr.pk, pr.pk, nb, f"ok:{cr.pk}", secs))
        n_done += 1

    rows.sort(key=lambda r: r[0])

    table = Table(title="Notebook execution status", show_lines=False)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Notebook", overflow="fold")
    table.add_column("Status", justify="left")
    table.add_column("Time (s)", justify="right")

    for _, pid, nb, status, secs in rows:
        path_str = pretty_path(nb)

        if status == "-":
            table.add_row("-", path_str, "-", "")
        elif status == "fail":
            table.add_row(str(pid), path_str, "[red]✗[/red]", "")
        elif status == "todo":
            table.add_row(str(pid), path_str, "[yellow]-[/yellow]", "")
        else:
            _, cache_pk = status.split(":")
            table.add_row(
                str(pid),
                path_str,
                f"[green]✓[/green] [{cache_pk}]",
                f"{secs:,.2f}",
            )

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

    console.print("[bold]Timing statistics[/bold]")
    console.print(f"  median time     : {10**med:,.2f} s")
    console.print(f"  10–90% interval : [{10**lo:,.2f}, {10**hi:,.2f}] s")

    if n_todo > 0:
        eta = 10 ** (med + math.log10(n_todo))
        console.print(
            f"  ETA ({n_todo} remaining) : {eta:,.2f} s (~{eta / 60:,.1f} min)"
        )

    console.print("\n[bold]execution time histogram[/bold]")
    for line in ascii_hist_seconds(logt):
        console.print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_folder", type=Path)
    args = ap.parse_args()
    main(args.experiment_folder)
