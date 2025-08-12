#!/usr/bin/env python3
"""
Usage:
  python -m experiments.run generate  <experiment-folder>
  python -m experiments.run execute   <experiment-folder> [-- ...jcache-args]
  python -m experiments.run merge     <experiment-folder>
  python -m experiments.run collect <experiment-folder> [--output runs.csv]
  python -m experiments.run list      <experiment-folder> [-- ...jcache-args]
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Number
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from jupyter_cache import get_cache
from tqdm import tqdm

KERNEL_NAME = "python3"
CACHE_DIRNAME = ".jupyter_cache"
RUNS_DIRNAME = "runs"
EXPORT_CELL_ID = "export-glue"


def exp_paths(exp):
    exp = exp.resolve()
    return exp, exp / RUNS_DIRNAME, exp / CACHE_DIRNAME


def load_cfg(exp):
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", exp / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stable_params(d):
    return OrderedDict(sorted(d.items(), key=lambda kv: kv[0]))


def list_runs(runs):
    return sorted(p for p in runs.glob("*.ipynb") if p.is_file())


def jcache_cli(
    *args,
    cwd=None,
    cache_path=None,
    suppress_stdout=False,
    suppress_stderr=False,
):
    exe = shutil.which("jcache")
    if not exe:
        raise RuntimeError("Could not find 'jcache' on PATH")
    env = os.environ.copy()
    if cache_path is not None:
        env["JUPYTERCACHE"] = str(cache_path)
    stdout = subprocess.DEVNULL if suppress_stdout else None
    stderr = subprocess.DEVNULL if suppress_stderr else None
    res = subprocess.run(
        [exe] + list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )
    return res.returncode


def prepare_from_config(exp, runs):
    cfg = load_cfg(exp)
    src = exp / "code.ipynb"
    runs.mkdir(exist_ok=True)
    notebooks = []

    for cfg_ in tqdm(cfg.configurations(), desc="prepare"):
        idx = len(notebooks) + 1
        name = cfg_.pop("name", f"{idx:06d}")
        out = runs / f"{name}.ipynb"
        if out in notebooks:
            raise RuntimeError("Duplicate notebook name: %s" % out)
        pm.execute_notebook(
            str(src),
            str(out),
            parameters=stable_params(cfg_),
            prepare_only=True,
            kernel_name=KERNEL_NAME,
            log_output=False,
            progress_bar=False,
        )
        notebooks.append(out)

    keep = {p.resolve() for p in notebooks}
    for f in list_runs(runs):
        if f.resolve() not in keep:
            print("remove stale:", f)
            f.unlink()

    append_export_cell(notebooks)
    return notebooks


def append_export_cell(notebooks):
    for nb_path in tqdm(list(notebooks), desc="export-cell"):
        nb = nbformat.read(nb_path, as_version=4)
        nb.cells = [
            c for c in nb.cells if getattr(c, "id", None) != EXPORT_CELL_ID
        ]

        exports = set()
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            if "export" not in cell.metadata.get("tags", []):
                continue
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
            glue_lines = "\n".join(
                f"sb.glue({name!r}, _walk({name}))" for name in sorted(exports)
            )
            code = (
                """import scrapbook as sb
import numpy as np

def _to_py(x):
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, np.generic):
        return x.item()
    return x

def _walk(x):
    x = _to_py(x)
    if isinstance(x, dict):
        return {k: _walk(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_walk(v) for v in x]
    return x

# glue exports
"""
                + glue_lines
            )
            nb.cells.append(
                nbformat.v4.new_code_cell(
                    source=code,
                    id=EXPORT_CELL_ID,
                )
            )
        nbformat.write(nb, nb_path)


def reset_cache(cache_dir):
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def register_project(cache_dir, notebooks):
    cache = get_cache(cache_dir)
    for nb in notebooks:
        cache.add_nb_to_project(nb)


def set_cache_limit(cache_dir, n_records):
    cache = get_cache(cache_dir)
    cache.change_cache_limit(max(8, n_records))


def cmd_generate(exp_dir):
    exp, runs, cache_dir = exp_paths(exp_dir)
    print("generate:", exp)
    notebooks = prepare_from_config(exp, runs)
    print("reset cache:", cache_dir)
    reset_cache(cache_dir)
    print("register project")
    register_project(cache_dir, notebooks)
    print("set cache limit:", len(notebooks) * 2)
    set_cache_limit(cache_dir, len(notebooks) * 2)
    print("done.")


def cmd_execute(exp_dir, passthrough):
    exp, runs, cache_dir = exp_paths(exp_dir)
    if not cache_dir.exists():
        print("no cache at %s, run generate first" % cache_dir, file=sys.stderr)
        sys.exit(2)

    # default timeout=0 unless user provided one
    if not any(
        arg == "--timeout" or arg.startswith("--timeout=")
        for arg in passthrough
    ):
        passthrough = ["--timeout", "0"] + passthrough

    print("execute: jcache project execute", " ".join(passthrough))
    rc = jcache_cli(
        "project", "execute", *passthrough, cwd=exp, cache_path=cache_dir
    )
    if rc != 0:
        sys.exit(rc)


def has_cache_match(cache_dir, nb_path):
    cache = get_cache(cache_dir)
    nb_node = nbformat.read(str(nb_path), as_version=4)
    try:
        cache.match_cache_notebook(nb_node)
        return True
    except KeyError:
        return False


def merge_one(cache_dir, src):
    tmp = src.with_suffix(".merged.tmp.ipynb")
    rc = jcache_cli(
        "notebook",
        "merge",
        str(src),
        str(tmp),
        cache_path=cache_dir,
        suppress_stdout=True,
    )
    if rc != 0:
        return False
    tmp.replace(src)
    return True


def cmd_merge(exp_dir):
    exp, runs, cache_dir = exp_paths(exp_dir)
    if not cache_dir.exists():
        print(
            "no cache at %s, run generate/execute first" % cache_dir,
            file=sys.stderr,
        )
        sys.exit(2)

    notebooks = list_runs(runs)
    if not notebooks:
        print("no notebooks to merge in %s" % runs)
        return

    merged = 0
    for nb in tqdm(notebooks, desc="merge"):
        if has_cache_match(cache_dir, nb):
            if merge_one(cache_dir, nb):
                merged += 1
    print("merged %d / %d notebooks" % (merged, len(notebooks)))


# Helpers for long-form collection via cartesian product
def _is_listlike(x):
    return isinstance(x, Sequence) and not isinstance(
        x, (str, bytes, bytearray)
    )


def _series_df(name, seq):
    seq = list(seq)
    df = pd.DataFrame({name: seq})
    df[f"{name}.index"] = range(len(df))
    return df[[f"{name}.index", name]]


def _lodf_df(name, seq):
    vals = pd.json_normalize(list(seq), max_level=1)
    vals.columns = [f"{name}.{c}" for c in vals.columns]
    vals.insert(0, f"{name}.index", range(len(vals)))
    return vals


def _flatten_top_level_dict(name, mapping):
    out = {}
    for k, v in mapping.items():
        if isinstance(v, (str, bool, Number, type(None), np.generic)):
            out[f"{name}.{k}"] = v.item() if isinstance(v, np.generic) else v
    return out


def _cross_join(a, b):
    if a is None:
        return b.copy()
    if b is None:
        return a.copy()
    a = a.copy()
    b = b.copy()
    a["_key"] = 1
    b["_key"] = 1
    res = a.merge(b, on="_key").drop(columns="_key")
    return res


def cmd_collect(exp_dir, out_csv):
    exp, runs, _ = exp_paths(exp_dir)
    cmd_merge(exp_dir)

    outs = []
    for nb in tqdm(list_runs(runs), desc="collect-explode"):
        try:
            scraps = sb.read_notebook(str(nb)).scraps
        except Exception:
            continue

        ids = {"run": nb.stem}
        series_parts = []

        for name, s in scraps.items():
            v = getattr(s, "data", s)

            if isinstance(v, Mapping):
                # treat top-level dicts of scalars as ID columns
                ids.update(_flatten_top_level_dict(name, v))

            elif _is_listlike(v):
                seq = list(v)
                if len(seq) == 0:
                    # empty sequence means no rows for this run once crossed
                    series_parts.append(
                        pd.DataFrame({f"{name}.index": [], name: []})
                    )
                else:
                    first = seq[0]
                    if isinstance(first, Mapping):
                        series_parts.append(
                            _lodf_df(name, seq)
                        )  # list of dicts
                    else:
                        series_parts.append(
                            _series_df(name, seq)
                        )  # list of scalars

            elif isinstance(v, (str, bool, Number, type(None), np.generic)):
                ids[name] = v.item() if isinstance(v, np.generic) else v
            else:
                # should not happen after glue normalization; ignore
                pass

        out = pd.DataFrame([ids])
        for part in series_parts:
            out = _cross_join(out, part)

        outs.append(out)

    if not outs:
        print("no scraps found; nothing to write")
        return

    df = pd.concat(outs, ignore_index=True)
    cols = ["run"] + sorted(c for c in df.columns if c != "run")
    df = df.reindex(columns=cols)

    out = out_csv if Path(out_csv).is_absolute() else (exp / out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(
        "collected %d rows, %d columns -> %s" % (len(df), len(df.columns), out)
    )


def cmd_list(exp_dir, passthrough):
    exp, runs, cache_dir = exp_paths(exp_dir)
    if not cache_dir.exists():
        print("no cache at %s, run generate first" % cache_dir, file=sys.stderr)
        sys.exit(2)
    print("list: jcache notebook list", " ".join(passthrough or []))
    rc = jcache_cli(
        "notebook", "list", *(passthrough or []), cwd=exp, cache_path=cache_dir
    )
    if rc != 0:
        sys.exit(rc)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="experiments.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser(
        "generate", help="prepare notebooks and reset/register cache"
    )
    p_gen.add_argument("experiment_folder", type=Path)

    p_exec = sub.add_parser(
        "execute", help="wrapper around jcache project execute"
    )
    p_exec.add_argument("experiment_folder", type=Path)
    p_exec.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="args passed to jcache",
    )

    p_merge = sub.add_parser(
        "merge", help="merge executed outputs back into runs/*.ipynb"
    )
    p_merge.add_argument("experiment_folder", type=Path)

    p_sum = sub.add_parser(
        "collect", help="merge then extract scrapbook scraps into CSV"
    )
    p_sum.add_argument("experiment_folder", type=Path)
    p_sum.add_argument(
        "--output", default="runs.csv", help="output CSV filename"
    )

    p_list = sub.add_parser("list", help="wrapper around jcache notebook list")
    p_list.add_argument("experiment_folder", type=Path)
    p_list.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="args passed to jcache",
    )

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        cmd_generate(args.experiment_folder)
    elif args.cmd == "execute":
        passthrough = args.passthrough or []
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]
        cmd_execute(args.experiment_folder, passthrough)
    elif args.cmd == "merge":
        cmd_merge(args.experiment_folder)
    elif args.cmd == "collect":
        cmd_collect(args.experiment_folder, args.output)
    elif args.cmd == "list":
        passthrough = args.passthrough or []
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]
        cmd_list(args.experiment_folder, passthrough)
    else:
        ap.error("unknown command")


if __name__ == "__main__":
    main()
