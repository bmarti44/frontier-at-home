#!/usr/bin/env python3
"""Install the pinned standalone Rust toolchain into a new user-owned prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new download and installation directory")
    args = parser.parse_args()
    if os.geteuid() == 0:
        raise ValueError("user-owned installation only")
    lock_path = ROOT / "configs/build-manifests/glm53-rust-toolchain.json"
    lock = json.loads(lock_path.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / lock["url"].rsplit("/", 1)[1]
    with urllib.request.urlopen(lock["url"], timeout=60) as response, archive.open("xb") as stream:
        shutil.copyfileobj(response, stream, 1024 * 1024)
    if sha(archive) != lock["sha256"]:
        raise ValueError("Rust archive digest mismatch")
    with tarfile.open(archive) as stream:
        stream.extractall(output / "source", filter="data")
    source = output / "source" / archive.name.removesuffix(".tar.xz")
    prefix = output / "toolchain"
    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    with (output / "raw.log").open("wb") as log:
        subprocess.run(["bash", str(source / "install.sh"), f"--prefix={prefix}",
                        "--disable-ldconfig", "--components=" + ",".join(lock["components"])],
                       env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    files = [{"path": str(p.relative_to(prefix)), "sha256": sha(p), "size_bytes": p.stat().st_size}
             for p in sorted(prefix.rglob("*")) if p.is_file()]
    versions = {name: subprocess.check_output([str(prefix / "bin" / name), "--version"], env=env, text=True).strip()
                for name in ("cargo", "rustc")}
    if not all(value.startswith(name + " " + lock["version"] + " ") for name, value in versions.items()):
        raise ValueError("installed toolchain version mismatch")
    manifest = {"schema_version": 1, "status": "toolchain_install_only", "source_lock_sha256": sha(lock_path),
                "scorer_sha256": sha(Path(__file__)), "files": files, "versions": versions}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(prefix)


if __name__ == "__main__":
    main()
