"""Create a new relocatable CPython prefix; never authorize a serving launch.

Materialize internal symlinks, replace only the copied base site-packages with
the finalized build environment's packages, and freeze every resulting file.
Permission, import, loader, JIT and model checks remain separate launch gates.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat

from glm53_contract import sha256_file, verify_inventory


def _identity(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _absolute(path):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("symlink input or output ancestor: " + str(parent))
    return path


def _scan(root, excluded=None):
    """Resolve internal file/directory links, rejecting escapes and cycles."""
    result = {}

    def visit(path, name, ancestors):
        if excluded is not None and (name == excluded or name.is_relative_to(excluded)):
            return
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("symlink escapes source root: " + str(path))
        value = resolved.stat()
        if stat.S_ISDIR(value.st_mode):
            if resolved in ancestors:
                raise ValueError("directory symlink cycle: " + str(path))
            result[name.as_posix()] = (path, resolved, _identity(path), _identity(resolved), True)
            for child in sorted(resolved.iterdir()):
                visit(child, name / child.name, ancestors | {resolved})
        elif stat.S_ISREG(value.st_mode):
            result[name.as_posix()] = (path, resolved, _identity(path), _identity(resolved), False)
        else:
            raise ValueError("source contains special, non-regular file: " + str(path))

    visit(root, Path("."), set())
    return result


def _copy(entries, destination):
    for name, (_, source, _, identity, directory) in entries.items():
        target = destination / name
        if directory:
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(0o755)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755 if identity[2] & 0o111 else 0o644)
        if _identity(source) != identity or sha256_file(source) != sha256_file(target):
            raise ValueError("source changed during copy: " + str(source))


def package_runtime(python_prefix, site_packages, output):
    base, site, output = map(_absolute, (python_prefix, site_packages, output))
    if output.exists():
        raise FileExistsError(output)
    for first, second in ((base, site), (base, output), (site, output)):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError("runtime inputs/output overlap")
    if (base / "pyvenv.cfg").exists():
        raise ValueError("base must be a standalone interpreter, not a venv")
    if not (base / "bin/python3.12").is_file() or site.name != "site-packages":
        raise ValueError("expected CPython 3.12 prefix and site-packages")
    relative_site = Path("lib/python3.12/site-packages")
    base_files = _scan(base, excluded=relative_site)
    site_files = _scan(site)
    output.mkdir(parents=True, exist_ok=False)
    prefix = output / "runtime"
    # Partial/failed candidates are retained; subsequent calls must use a new
    # output directory. No source file, permission or symlink is changed.
    _copy(base_files, prefix)
    _copy(site_files, prefix / relative_site)
    if _scan(base, excluded=relative_site) != base_files or _scan(site) != site_files:
        raise ValueError("source inventory changed during packaging")
    inventory = {"schema_version": 1, "files": [
        {"path": p.relative_to(prefix).as_posix(), "size_bytes": p.stat().st_size,
         "sha256": sha256_file(p)}
        for p in sorted(prefix.rglob("*")) if p.is_file()]}
    verify_inventory(prefix, inventory)
    manifest = output / "inventory.json"
    manifest.write_text(json.dumps(inventory, indent=2) + "\n")
    summary = {"schema_version": 1, "qualification": "packaging_only",
               "status": "prepared_unqualified", "runtime": str(prefix),
               "python_prefix": str(base), "site_packages": str(site),
               "inventory_sha256": sha256_file(manifest),
               "scorer_sha256": sha256_file(Path(__file__)),
               "binary_sha256": sha256_file(prefix / "bin/python3.12"),
               "files": len(inventory["files"]),
               "bytes": sum(row["size_bytes"] for row in inventory["files"]),
               "startup_hooks": [p.relative_to(prefix).as_posix() for p in sorted(prefix.rglob("*.pth"))],
               "pending": ["service-credential permissions", "isolated imports and worker flags",
                           "native loader dependencies and RPATH", "JIT closure", "model qualification"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-prefix", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_runtime(args.python_prefix, args.site_packages, args.output)))
