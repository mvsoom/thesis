import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import DictKey, GetAttrKey, SequenceKey


def _path_to_str(path):
    parts = []
    for k in path:
        if isinstance(k, GetAttrKey):
            parts.append(str(k.name))
        elif isinstance(k, SequenceKey):
            parts.append(f"[{k.idx}]")
        elif isinstance(k, DictKey):
            parts.append(f"['{k.key}']")
        else:
            parts.append(str(k))
    s = ""
    for p in parts:
        if p.startswith("[") or p.startswith("['"):
            s += p
        else:
            s = p if s == "" else s + "." + p
    return s if s else "<root>"


def _leaf_name_from_path(path):
    if not path:
        return "<root>"
    k = path[-1]
    if isinstance(k, GetAttrKey):
        return str(k.name)
    if isinstance(k, SequenceKey):
        return f"[{k.idx}]"
    if isinstance(k, DictKey):
        return str(k.key)
    return str(k)


def check_precisions(tree, return_issues=False):
    issues = []
    paths = []
    leaves = []

    # Collect (path, leaf) pairs in a version-robust way
    def _collect(path, x):
        paths.append(path)
        leaves.append(x)
        return None

    jax.tree_util.tree_map_with_path(_collect, tree)

    for i, (path, x) in enumerate(zip(paths, leaves)):
        leaf_path = _path_to_str(path)
        leaf_name = _leaf_name_from_path(path)

        def report(kind, dtype, shape, value):
            issues.append(
                {
                    "index": i,
                    "path": leaf_path,
                    "name": leaf_name,
                    "kind": kind,
                    "dtype": str(dtype),
                    "shape": tuple(shape) if hasattr(shape, "__iter__") else (),
                    "value": value,
                }
            )

        if isinstance(x, (jnp.ndarray, np.ndarray)):
            dt = x.dtype
            if jnp.issubdtype(dt, jnp.floating):
                if jnp.finfo(dt).bits > 32:
                    report(
                        "float>32",
                        dt,
                        x.shape,
                        x if x.shape == () else "<array>",
                    )
            elif jnp.issubdtype(dt, jnp.integer):
                if np.iinfo(dt).bits > 32:
                    report(
                        "int>32", dt, x.shape, x if x.shape == () else "<array>"
                    )
        elif isinstance(x, float):
            report("python-float", "float", (), x)
        elif isinstance(x, int):
            report("python-int", "int", (), x)

    if not issues:
        print("All numeric leaves are <=32-bit.")
    else:
        print("Found potential width issues:")
        for it in issues:
            print(
                f"  Leaf #{it['index']}: path={it['path']} name={it['name']} "
                f"dtype={it['dtype']} shape={it['shape']} kind={it['kind']} val={it['value']}"
            )

    if return_issues:
        return issues


if __name__ == "__main__":
    from iklp.hyperparams import Hyperparams

    h = Hyperparams()

    check_precisions(h, return_issues=True)
