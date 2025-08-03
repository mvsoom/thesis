#!/usr/bin/env python3
import argparse
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

def patch_exports(notebooks):
    for nb_path in tqdm(notebooks, "Patching export cells"):
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            tags = cell.metadata.get("tags", [])
            if "export" in tags:
                vars_ = [
                    line.strip()
                    for line in cell.source.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                glue = ["import scrapbook as sb", ""]
                for v in vars_:
                    glue.append(f"sb.glue({v!r}, {v})")
                cell.source = "\n".join(glue)
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
    patch_exports(notebooks)

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
