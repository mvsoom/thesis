#!/usr/bin/env python3
import argparse
import ast
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import nbformat
import papermill as pm
from jupyter_cache import get_cache
from tqdm import tqdm


def load_cfg(exp: Path):
    spec = importlib.util.spec_from_file_location("config", exp / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stable_params(d: dict):
    # Make papermill's injected parameters cell text stable across runs.
    return OrderedDict(sorted(d.items(), key=lambda kv: kv[0]))


def gen_notebooks(src: Path, cfg_module, runs: Path):
    runs.mkdir(exist_ok=True)
    notebooks = []
    for cfg_ in tqdm(cfg_module.configurations(), desc="Preparing notebooks"):
        idx = len(notebooks) + 1
        name = cfg_.get("name", f"{idx:06d}")
        out = runs / f"{name}.ipynb"

        assert out not in notebooks, (
            f"Duplicate notebook name: {out}. "
            "Ensure unique 'name' per configuration or remove 'name'."
        )

        pm.execute_notebook(
            str(src),
            str(out),
            parameters=stable_params(cfg_),
            prepare_only=True,
            kernel_name="python3",
        )
        notebooks.append(out)
    return [Path(nb) for nb in notebooks]


def append_export_cell(notebooks):
    for nb_path in tqdm(notebooks, desc="Appending export cell"):
        nb = nbformat.read(nb_path, as_version=4)

        # Remove any previous export cell to avoid duplicates.
        nb.cells = [
            c for c in nb.cells if getattr(c, "id", None) != "export-glue"
        ]

        exports = set()
        for cell in nb.cells:
            if (
                cell.cell_type == "code"
                and "export" in cell.metadata.get("tags", [])
                and isinstance(cell.source, str)
            ):
                try:
                    tree = ast.parse(cell.source)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name):
                                exports.add(tgt.id)
                            elif isinstance(tgt, ast.Tuple):
                                for elt in tgt.elts:
                                    if isinstance(elt, ast.Name):
                                        exports.add(elt.id)

        if exports:
            lines = ["import scrapbook as sb"] + [
                f"sb.glue({name!r}, {name})" for name in sorted(exports)
            ]
            nb.cells.append(
                nbformat.v4.new_code_cell(
                    source="\n".join(lines),
                    id="export-glue",
                )
            )

        nbformat.write(nb, nb_path)


def fullpath(p: Path) -> Path:
    return Path(p).resolve()


def prune_stale(runs: Path, notebooks):
    keep = {fullpath(nb) for nb in notebooks}
    for f in runs.glob("*.ipynb"):
        if fullpath(f) not in keep:
            f.unlink()


def prune_project_records(cache, notebooks):
    valid = {str(fullpath(nb)) for nb in notebooks}
    for rec in cache.list_project_records():
        if str(fullpath(rec.uri)) not in valid:
            cache.remove_nb_from_project(rec.pk)


def add_to_project(cache, notebooks):
    for nb in notebooks:
        cache.add_nb_to_project(nb)


def execute_and_cache(notebooks, cache):
    for nb_path in tqdm(notebooks, desc="Executing"):
        nb_path = fullpath(nb_path)
        nb_node = nbformat.read(str(nb_path), as_version=4)

        try:
            cache.match_cache_notebook(nb_node)
            print(f"Cached: {nb_path} (skip)")
            continue
        except KeyError:
            print(f"Miss: {nb_path} (execute)")

        # Execute in place with the same kernel_name used at prepare time.
        pm.execute_notebook(
            str(nb_path),
            str(nb_path),
            kernel_name="python3",
            log_output=True,
            progress_bar=False,
            stdout_file=sys.stdout,
            stderr_file=sys.stderr,
        )

        # Store cache record; validate against project metadata.
        cache.cache_notebook_file(
            path=str(nb_path),
            overwrite=True,
            check_validity=True,
        )


def main(args):
    exp = args.experiment_folder
    cfg = load_cfg(exp)
    src_nb = exp / "code.ipynb"
    runs = exp / "runs"

    cache = get_cache(exp / ".jupyter_cache")

    if args.regenerate:
        notebooks = gen_notebooks(src_nb, cfg, runs)
        append_export_cell(notebooks)
        prune_stale(runs, notebooks)
        prune_project_records(cache, notebooks)
        cache.change_cache_limit(max(8, len(notebooks) * 2))
        add_to_project(cache, notebooks)
    else:
        notebooks = sorted(runs.glob("*.ipynb"))
        if not notebooks:
            print("No notebooks found. Regenerating...")
            args.regenerate = True
            return main(args)
        add_to_project(cache, notebooks)  # idempotent

    execute_and_cache(notebooks, cache)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment_folder", type=Path)
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate prepared notebooks from the experiment's config.py and code.ipynb and rerun experiment from scratch",
    )
    args = p.parse_args()

    exit(main(args))
