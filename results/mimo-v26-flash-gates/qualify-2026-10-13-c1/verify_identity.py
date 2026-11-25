#!/usr/bin/env python3
"""Check the runtime and weights the c1 start will load against their pins.

Usage: verify_identity.py <repo>  -> one line per check, exit 0 iff all match.
- runtime-src: src/tabbyAPI (run from source) and src/exllamav3 are at the
  build manifest's revision and tree, with no tracked edits.
- runtime-wheel: the exllamav3 wheel has the manifest's sha256, and every file
  it installs (its RECORD) is byte-identical in the runtime's site-packages.
- weights: each served directory (the profile's artifact_roles) holds exactly
  its manifest artifact's files, at their sizes and sha256. Reads drop their
  pages afterwards (POSIX_FADV_DONTNEED), so the 94 GiB pass leaves no page cache.
"""
import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROFILE = "configs/profiles/mimo-v2.6-flash/cuda-spark-128g-262k.json"
WEIGHTS = "weights/mimo-v2.6-flash/manifest.json"
BUILD = "configs/build-manifests/mimo-v26-flash-exl3-runtime-001.json"
RUNTIME = Path("/home/bmarti44/.cache/mimo-v26-flash/native-runtime-001")
SITE = RUNTIME / "runtime/lib/python3.12/site-packages"
DIRS = {"model": "/home/bmarti44/models/mimo-v2.6-flash-serve/mimo-v2.6-flash",
        "draft_model": "/home/bmarti44/models/mimo-v2.6-flash-draft/mimo-v2.6-flash-dflash"}
CHUNK = 16 * 2**20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while chunk := os.read(fd, CHUNK):
            digest.update(chunk)
            os.posix_fadvise(fd, offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def git(source: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True, text=True).stdout


def runtime(build: dict) -> bool:
    ok = True
    for name in ("tabbyAPI", "exllamav3"):
        source = RUNTIME / "src" / name
        rev, tree = git(source, "rev-parse", "HEAD", "HEAD^{tree}").split()
        dirty = len(git(source, "status", "--porcelain", "--untracked-files=no").splitlines())
        want = build["sources"][name]
        match = (rev, tree, dirty) == (want["revision"], want["tree"], 0)
        print(f"runtime-src {name} {rev} {tree} dirty={dirty} match={match}", flush=True)
        ok &= match
    wheel = build["wheels"][0]
    path = RUNTIME / wheel["file"]
    wheel_ok = sha256(path) == wheel["sha256"]
    mismatches, files = [], 0
    with zipfile.ZipFile(path) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        for row in csv.reader(io.StringIO(archive.read(record).decode())):
            if len(row) < 2 or not row[1].startswith("sha256="):
                continue
            files += 1
            want = row[1][len("sha256="):]
            target = SITE / row[0]
            got = (base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest()).rstrip(b"=").decode()
                   if target.is_file() else None)
            if got != want:
                mismatches.append(row[0])
    print(f"runtime-wheel exllamav3 sha256_match={wheel_ok} files={files} mismatches={len(mismatches)}"
          + (f" {mismatches[:5]}" if mismatches else ""), flush=True)
    return ok and wheel_ok and files > 0 and not mismatches


def weights(repo: Path) -> bool:
    roles = json.loads((repo / PROFILE).read_text())["artifact_roles"]
    artifacts = json.loads((repo / WEIGHTS).read_text())["artifacts"]
    ok = True
    for role, directory in DIRS.items():
        name = roles[role]
        files = {f["name"]: f for f in artifacts[name]["files"]}
        root = Path(directory)
        present = sorted(os.listdir(root))
        problems = [] if present == sorted(files) else [f"listing {present} != manifest {sorted(files)}"]
        wanted = [root / n for n in sorted(files) if (root / n).is_file()]
        with ThreadPoolExecutor(max_workers=4) as pool:
            digests = dict(zip(wanted, pool.map(sha256, wanted)))
        problems += [p.name for p, d in digests.items()
                     if p.stat().st_size != files[p.name]["bytes"] or d != files[p.name]["sha256"]]
        total = sum(files[p.name]["bytes"] for p in wanted)
        print(f"weights {role} {name} files={len(digests)} bytes={total} mismatches={len(problems)}"
              + (f" {problems[:5]}" if problems else ""), flush=True)
        ok &= not problems and len(digests) == len(files)
    return ok


def main(repo: Path) -> int:
    build = json.loads((repo / BUILD).read_text())
    ok = runtime(build)
    ok &= weights(repo)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
