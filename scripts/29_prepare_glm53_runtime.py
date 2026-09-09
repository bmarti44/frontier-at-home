#!/usr/bin/env python3
"""Prepare clean, committed GLM source worktrees; never build or load a model.

The source lock is not runtime qualification. Run bounded import/kernel tests
after a clean build under the existing GLM cgroup and inference-lock wrapper.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from glm53_contract import sha256_file, strict_json


def command(argv, **kwargs):
    subprocess.run([str(arg) for arg in argv], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="pristine source worktrees from the source lock")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory, never an existing worktree")
    parser.add_argument("--python", type=Path, required=True,
                        help="absolute Python from the isolated build environment")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    if not args.python.is_absolute() or not args.python.is_file():
        parser.error("--python must name an existing absolute interpreter")
    lock_path = ROOT / "configs/build-manifests/glm53-flash-sources.json"
    lock = strict_json(lock_path)
    spec = importlib.util.spec_from_file_location("glm53_source_audit", ROOT / "scripts/27_audit_glm53_sources.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.verify_sources(args.source_root, lock)
    args.output.mkdir(parents=True, exist_ok=False)
    diffs = args.output / "diffs"
    diffs.mkdir()
    for name in ("vllm", "exllamav3", "vllm-exl3"):
        command(["git", "-C", args.source_root / name, "worktree", "add", "--detach",
                 args.output / name, lock["sources"][name]["revision"]])
    recipe = args.source_root / "recipe/scripts"
    vllm = args.output / "vllm"
    exllama = args.output / "exllamav3"
    # Do not apply optional speculative or diagnostic patches to this baseline.
    for name in ("patch_glm53_sm121_nope.py", "patch_kpool_tail_positions.py"):
        command([args.python, recipe / name, "--source", vllm])
    # The recipe's NoPE script also adds optional DFlash allocation and MTP
    # changes. Preserve the exact original files in this no-spec baseline.
    for relative in ("vllm/v1/core/kv_cache_utils.py", "vllm/models/glm5next/nvidia/mtp.py"):
        (vllm / relative).write_bytes((args.source_root / "vllm" / relative).read_bytes())
    command([args.python, recipe / "patch_glm53_dense_exl3_quant_config.py", vllm / "vllm"])
    command([args.python, recipe / "patch_exllamav3_aarch64.py", exllama / "exllamav3/exllamav3_ext"])
    command([args.python, ROOT / "scripts/28_patch_glm53_frames.py", "--source", vllm])
    requirements = vllm / "requirements/cuda.txt"
    text = requirements.read_text()
    old, new = "flashinfer-python==0.6.17", "flashinfer-python==0.6.18rc10"
    if text.count(old) != 1:
        raise ValueError("FlashInfer metadata anchor count is not one")
    requirements.write_text(text.replace(old, new))
    rust = vllm / "tools/build_rust.py"
    text = rust.read_text()
    for old, new in (( 'args=["--bin", "vllm-rs"],', 'args=["--bin", "vllm-rs", "--locked"],'),
                     ('features=["pyo3/abi3-py38"],', 'features=["pyo3/abi3-py38"],\n            args=["--locked"],')):
        if text.count(old) != 1:
            raise ValueError("Rust locked-build anchor count is not one")
        text = text.replace(old, new)
    rust.write_text(text)
    for relative, key, tag in (("CMakeLists.txt", "cutlass", "v4.4.2"),
                               ("cmake/external_projects/triton_kernels.cmake", "triton_kernels", "v3.5.1")):
        path = vllm / relative
        text = path.read_text()
        old = f'"{tag}"'
        if text.count(old) != 1:
            raise ValueError(f"{key}: transitive source anchor count is not one")
        text = text.replace(old, f'"{lock["transitive_sources"][key]["revision"]}"')
        if key == "cutlass":
            if text.count("GIT_SHALLOW TRUE") != 1:
                raise ValueError("CUTLASS shallow-clone anchor count is not one")
            text = text.replace("GIT_SHALLOW TRUE", "GIT_SHALLOW FALSE")
        path.write_text(text)
    command([args.python, ROOT / "scripts/tests/glm53_build_contract.py", "--source", vllm,
             "--pristine", args.source_root / "vllm"])
    command([args.python, ROOT / "scripts/tests/glm53_frame_contract.py", "--source", vllm])
    # Upstream patch backup copies are redundant with the preserved pristine
    # tree and diff. Only these named, newly generated files are removed.
    for name in ("kda.py.orig-p2", "model.py.orig-p2"):
        (vllm / "vllm/models/glm5next/nvidia" / name).unlink()
    manifest = {"schema_version": 1, "status": "source_prepared_only",
                "source_lock_sha256": sha256_file(lock_path),
                "scorer_sha256": sha256_file(Path(__file__)),
                "binary_sha256": sha256_file(args.python.resolve()), "sources": {}}
    for name in ("vllm", "exllamav3", "vllm-exl3"):
        path = args.output / name
        command(["git", "-C", path, "add", "-A"])
        diff = subprocess.check_output(["git", "-C", path, "diff", "--cached", "--binary"])
        patch_path = diffs / f"{name}.patch"
        patch_path.write_bytes(diff)
        if diff:
            command(["git", "-C", path, "commit", "-m", "Prepare optional GLM 5.3 Spark source candidate"])
        if subprocess.check_output(["git", "-C", path, "status", "--porcelain"]):
            raise ValueError(f"{name}: prepared source is dirty")
        manifest["sources"][name] = {
            "base_revision": lock["sources"][name]["revision"],
            "revision": subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip(),
            "tree": subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD^{tree}"], text=True).strip(),
            "patch_sha256": sha256_file(patch_path),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared source only; runtime remains unqualified:", args.output)


if __name__ == "__main__":
    main()
