#!/usr/bin/env python3
"""Check the runtime and weights the Inkling 2 Mini M3 start will load against their pins.

Usage: verify_identity.py <repo>  -> one line per check, exit 0 iff all match.
- interpreter: the runtime's python3.12 has the build manifest's sha256.
- vllm-tree: site-packages/vllm hashes to the manifest's vllm_tree (the method
  recorded there: sorted relpath NUL sha256 LF lines, no __pycache__/*.pyc), i.e.
  the wheel plus exactly patches 0001-0004.
- patches: every patch file in the manifest has its pinned sha256.
- packages freeze: `uv pip freeze` of the runtime hashes to the manifest's freeze_sha256, so
  torch, triton, flashinfer and humming_kernels are the installed versions too (review A-M3-L2).
- weights: ~/models/inkling-2-mini holds exactly the weights manifest's files (outside the
  downloader's .cache/ metadata),
  at their sizes and sha256. Reads drop their pages afterwards
  (POSIX_FADV_DONTNEED), so the 20 GB pass leaves no page cache.
Adapted from results/mimo-v26-flash-gates/qualify-2026-10-13-c1/verify_identity.py.
"""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUILD = "configs/build-manifests/inkling-2-mini-runtime.json"
WEIGHTS = "weights/inkling-2-mini/manifest.json"
RUNTIME = Path("/home/bmarti44/.cache/inkling-2-mini/native-runtime-001/runtime")
SITE = RUNTIME / "lib/python3.12/site-packages"
MODEL = Path("/home/bmarti44/models/inkling-2-mini")
CHUNK = 16 * 2**20
UV = "/home/bmarti44/.local/bin/uv"


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


def vllm_tree() -> tuple[str, int]:
    root = SITE / "vllm"
    rows = []
    for d, _, files in os.walk(root):
        for n in files:
            p = Path(d) / n
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in rel.split("/") or rel.endswith(".pyc"):
                continue
            rows.append(f"{rel}\0{sha256(p)}\n")
    rows.sort()
    return hashlib.sha256("".join(rows).encode()).hexdigest(), len(rows)


def runtime(repo: Path, build: dict) -> bool:
    interp = sha256(RUNTIME / "bin/python3.12") == build["interpreter"]["sha256"]
    print(f"interpreter match={interp}", flush=True)
    tree, files = vllm_tree()
    want = build["vllm_tree"]
    tree_ok = (tree, files) == (want["sha256"], want["files"])
    print(f"vllm-tree {tree[:16]} files={files} after={want['after']!r} match={tree_ok}", flush=True)
    bad = [p["path"] for p in build["patches"] if sha256(repo / p["path"]) != p["sha256"]]
    print(f"patches {len(build['patches'])} mismatches={bad}", flush=True)
    frozen = subprocess.run([UV, "pip", "freeze", "--python", str(RUNTIME / "bin/python3")],
                            capture_output=True, timeout=120).stdout
    freeze = hashlib.sha256(frozen).hexdigest()
    freeze_ok = freeze == build["freeze_sha256"]
    print(f"packages freeze sha256_16 {freeze[:16]} match={freeze_ok}", flush=True)
    return interp and tree_ok and not bad and freeze_ok


def weights(repo: Path) -> bool:
    files = {f["name"]: f for f in json.loads((repo / WEIGHTS).read_text())["files"]}
    # .cache/huggingface is the downloader's metadata; vLLM never reads it.
    present = sorted(rel for rel in (p.relative_to(MODEL).as_posix() for p in MODEL.rglob("*") if p.is_file())
                     if not rel.startswith(".cache/"))
    problems = [] if present == sorted(files) else [f"listing differs: {sorted(set(present) ^ set(files))[:5]}"]
    wanted = [MODEL / n for n in sorted(files) if (MODEL / n).is_file()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = dict(zip(wanted, pool.map(sha256, wanted)))
    name = lambda p: p.relative_to(MODEL).as_posix()  # noqa: E731
    problems += [name(p) for p, d in digests.items()
                 if p.stat().st_size != files[name(p)]["bytes"] or d != files[name(p)]["sha256"]]
    total = sum(files[name(p)]["bytes"] for p in wanted)
    print(f"weights files={len(digests)} bytes={total} mismatches={len(problems)}"
          + (f" {problems[:5]}" if problems else ""), flush=True)
    return not problems and len(digests) == len(files)


def main(repo: Path) -> int:
    build = json.loads((repo / BUILD).read_text())
    ok = runtime(repo, build)
    ok &= weights(repo)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
