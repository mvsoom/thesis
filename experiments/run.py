#!/usr/bin/env python3
import argparse
import ast
import importlib.util
import logging
from pathlib import Path

import nbformat
import papermill as pm
from jupyter_cache import get_cache
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("exp_folder", type=Path)
    return p.parse_args()


def load_cfg(exp):
    spec = importlib.util.spec_from_file_location("config", exp / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gen_notebooks(src, cfg, runs):
    runs.mkdir(exist_ok=True)
    notebooks = []
    for cfg_ in tqdm(cfg.configurations()):
        idx = len(notebooks) + 1
        out = runs / f"{idx:06d}.ipynb"
        pm.execute_notebook(
            src, out, parameters=cfg_, prepare_only=True, kernel_name="python3"
        )
        notebooks.append(out)
    return notebooks


def append_export_cell(notebooks):
    for nb_path in tqdm(notebooks):
        nb = nbformat.read(nb_path, as_version=4)
        exports = set()
        for cell in nb.cells:
            if "export" in cell.metadata.get("tags", []):
                tree = ast.parse(cell.source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name):
                                exports.add(tgt.id)
                            elif isinstance(tgt, ast.Tuple):
                                exports.update(
                                    elt.id
                                    for elt in tgt.elts
                                    if isinstance(elt, ast.Name)
                                )
        if exports:
            lines = ["import scrapbook as sb"] + [
                f"sb.glue({name!r}, {name})" for name in sorted(exports)
            ]
            nb.cells.append(nbformat.v4.new_code_cell(source="\n".join(lines)))
            nbformat.write(nb, nb_path)


def prune_stale(runs, notebooks):
    keep = {str(nb) for nb in notebooks}
    for f in runs.glob("*.ipynb"):
        if str(f) not in keep:
            f.unlink()


def prune_cache(cache, notebooks):
    valid = {str(nb) for nb in notebooks}
    for rec in cache.list_project_records():
        if rec.uri not in valid:
            cache.remove_nb_from_project(rec.pk)


def execute_and_cache(notebooks, cache):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("papermill")
    logger.setLevel(logging.INFO)

    for nb_path in tqdm(notebooks):
        nb_node = nbformat.read(str(nb_path), as_version=4)
        try:
            cache.match_cache_notebook(nb_node)
            continue  # up-to-date
        except KeyError:
            pass

        pm.execute_notebook(
            str(nb_path), str(nb_path), kernel_name="python3", log_output=True
        )
        cache.cache_notebook_file(
            path=nb_path, overwrite=True, check_validity=False
        )


def main():
    exp = parse_args().exp_folder
    cfg = load_cfg(exp)
    src_nb = exp / "code.ipynb"
    runs = exp / "runs"

    notebooks = gen_notebooks(src_nb, cfg, runs)
    append_export_cell(notebooks)
    prune_stale(runs, notebooks)

    cache = get_cache(exp / ".jupyter_cache")
    cache.change_cache_limit(len(notebooks) * 2)
    prune_cache(cache, notebooks)

    for nb in notebooks:
        cache.add_nb_to_project(nb)

    execute_and_cache(notebooks, cache)


if __name__ == "__main__":
    main()
