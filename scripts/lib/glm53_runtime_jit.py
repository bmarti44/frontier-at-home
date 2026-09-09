"""Explicitly selected sealed Triton cache; never silently compile a miss.

This helper is not yet installed or enabled in a serving runtime. The complete
serving gate also needs guards for native-load fallback, DeepGEMM, CuTe and
other compilation paths. Cache roots must be immutable to the model process.
"""
from pathlib import Path
import re
import stat

from glm53_contract import strict_json, verify_inventory


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def sealed_cache_class(root, manifest):
    """Verify once before weights load; return a class for TRITON_CACHE_MANAGER.

    The returned class has no environment reads and no cache mutation path.
    Calls occur when loading a specialization, not on a resident kernel hit.
    """
    root = Path(root).absolute()
    identities = verify_inventory(root, manifest)
    directories = {str(p.relative_to(root)): _identity(p.lstat())
                   for p in (root, *root.rglob("*")) if p.is_dir()}
    keys = {name.split("/")[0] for name in identities if "/" in name}

    class SealedCache:
        def __init__(self, key, override=False, dump=False):
            if override or dump:
                raise ValueError("sealed cache rejects override and dump modes")
            if not isinstance(key, str) or not re.fullmatch(r"[A-Z2-7]{1,128}", key):
                raise ValueError("invalid sealed cache key")
            if key not in keys:
                raise ValueError("unknown unsealed kernel specialization")
            self.key = key

        def get_file(self, filename):
            if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename) or filename in (".", ".."):
                raise ValueError("invalid sealed cache path")
            name = self.key + "/" + filename
            if name not in identities:
                raise ValueError("unsealed or missing cache artifact: " + name)
            for directory in (root, root / self.key):
                observed = directory.lstat()
                relative = str(directory.relative_to(root))
                if not stat.S_ISDIR(observed.st_mode) or _identity(observed) != directories.get(relative):
                    raise ValueError("sealed cache directory changed")
            path = root / name
            try:
                observed = path.lstat()
            except FileNotFoundError as exc:
                raise ValueError("missing sealed cache artifact: " + name) from exc
            if not stat.S_ISREG(observed.st_mode) or _identity(observed) != identities[name]:
                raise ValueError("sealed cache artifact changed: " + name)
            return str(path)

        def get_group(self, filename):
            group_name = "__grp__" + filename
            group_path = self.get_file(group_name)
            group = strict_json(Path(group_path))
            if set(group) != {"child_paths"} or not isinstance(group["child_paths"], dict):
                raise ValueError("invalid sealed cache group metadata")
            if filename not in group["child_paths"]:
                raise ValueError("sealed group lacks requested metadata")
            children = {}
            for name, path in group["child_paths"].items():
                actual = self.get_file(name)
                if path != actual:
                    raise ValueError("unsealed group child path")
                children[name] = actual
            self.get_file(group_name)
            return children

        def put(self, data, filename, binary=True):
            raise ValueError("sealed cache cannot compile or write")

        def put_group(self, filename, group):
            raise ValueError("sealed cache cannot compile or write groups")

    return SealedCache
