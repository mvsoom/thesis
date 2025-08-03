#!/usr/bin/env python3
import argparse
import ast
import importlib.util
import logging
from pathlib import Path

import nbformat
import papermill as pm
from jupyter_cache import get_cache
from jupyter_cache.executors import load_executor
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("exp_folder", type=Path)
    p.add_argument(
        "--executor",
        choices=["local-serial", "local-parallel"],
        default="local-serial",
    )
    p.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Don't clear existing cache; continue from previous runs",
    )
    return p.parse_args()


def load_cfg(exp):
    spec = importlib.util.spec_from_file_location("config", exp / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gen_notebooks(src, cfg, runs):
    runs.mkdir(exist_ok=True)

    notebooks = []
    for config in tqdm(cfg.configurations(), "Generating notebooks"):
        id = len(notebooks) + 1
        out = runs / f"{id:06d}.ipynb"
        pm.execute_notebook(src, out, parameters=config, prepare_only=True)
        notebooks.append(out)

    return notebooks


def append_export_cell(notebooks):
    for nb_path in tqdm(notebooks, desc="Appending export cell"):
        nb = nbformat.read(nb_path, as_version=4)
        exports = set()

        # collect all LHS names in cells tagged "export"
        for cell in nb.cells:
            if "export" in cell.metadata.get("tags", []):
                tree = ast.parse(cell.source)
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
            lines = ["import scrapbook as sb"]
            for name in sorted(exports):
                lines.append(f"sb.glue({name!r}, {name})")
            nb.cells.append(nbformat.v4.new_code_cell(source="\n".join(lines)))
            nbformat.write(nb, nb_path)



def init_cache(exp, n, cont):
    cache = get_cache(exp / ".jupyter_cache")
    if not cont:
        cache.clear_cache()
    cache.change_cache_limit(n * 2)
    return cache


def stage_and_run(notebooks, cache, runs, executor_name):
    for nb in tqdm(notebooks, "Staging notebooks"):
        cache.add_nb_to_project(nb)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )

    execr = load_executor(executor_name, cache=cache)
    return execr.run_and_cache()


def merge_executed(cache, uri):
    pk, merged_nb = cache.merge_match_into_file(uri)
    nbformat.write(merged_nb, uri)


def main(args):
    exp = args.exp_folder
    cfg = load_cfg(exp)
    src_nb = exp / "code.ipynb"
    runs = exp / "runs"

    notebooks = gen_notebooks(src_nb, cfg, runs)
    append_export_cell(notebooks)

    cache = init_cache(exp, len(notebooks), args.continue_run)
    result = stage_and_run(notebooks, cache, runs, args.executor)

    print(
        f"Succeeded: {len(result.succeeded)}, "
        f"Excepted: {len(result.excepted)}, "
        f"Errored: {len(result.errored)}"
    )

    for uri in tqdm(result.succeeded, "Merging outputs"):
        merge_executed(cache, uri)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp_folder", type=Path)
    p.add_argument(
        "--executor",
        choices=["local-serial", "local-parallel"],
        default="local-serial",
    )
    p.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Don't clear existing cache; continue from previous runs",
    )

    main(p.parse_args())
