#!/usr/bin/env python3

import argparse
from pathlib import Path

from jupyter_cache import get_cache

CACHE_DIRNAME = ".jupyter_cache"
RUNS_DIRNAME = "runs"


def list_cache_only(cache):
    recs = cache.list_cache_records()

    if not recs:
        print("cache: empty")
        return 0

    print("cache records:\n")
    for r in sorted(recs, key=lambda x: x.pk):
        secs = None
        if r.data:
            secs = r.data.get("execution_seconds")
        uri = str(Path(r.uri))
        if secs is None:
            print(f"[{r.pk:4d}] {uri}")
        else:
            print(f"[{r.pk:4d}] {secs:10.3f} s  {uri}")
    print(f"\ncache total: {len(recs)}")
    return 0


def project_status(cache, exp):
    runs = exp / RUNS_DIRNAME
    nbs = sorted(runs.glob("*.ipynb"))

    if not nbs:
        print(f"no notebooks in {runs}")
        return 0

    done = 0
    todo = 0
    failed = 0

    print("project notebook status (hash match):\n")

    for nb in nbs:
        nb = nb.resolve()
        pr = None
        try:
            pr = cache.get_project_record(str(nb))
        except Exception:
            pr = None

        tb = getattr(pr, "traceback", None) if pr is not None else None
        if tb:
            failed += 1

        cr = cache.get_cached_project_nb(str(nb))
        if cr is None:
            print(f"{nb.name:40s}   --")
            todo += 1
            continue

        secs = None
        if cr.data:
            secs = cr.data.get("execution_seconds")
        if secs is None:
            print(f"{nb.name:40s}   ??  (cached pk={cr.pk})")
        else:
            print(f"{nb.name:40s} {secs:8.2f} s  (cached pk={cr.pk})")

        done += 1

    print("\nsummary:")
    print(f"  executed (hash match): {done}")
    print(f"  todo                : {todo}")
    print(f"  failed (traceback)  : {failed}")
    print(f"  total              : {len(nbs)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_folder", type=Path)
    ap.add_argument(
        "--cache",
        action="store_true",
        help="list cache DB records only (fast, no notebook hashing)",
    )
    args = ap.parse_args()

    exp = args.experiment_folder.resolve()
    cache_dir = exp / CACHE_DIRNAME
    if not cache_dir.exists():
        raise SystemExit("no .jupyter_cache found (run generate first?)")

    cache = get_cache(cache_dir)

    if args.cache:
        raise SystemExit(list_cache_only(cache))
    raise SystemExit(project_status(cache, exp))


if __name__ == "__main__":
    main()
