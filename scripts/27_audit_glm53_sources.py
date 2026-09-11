#!/usr/bin/env python3
"""Reproduce bounded GLM runtime falsifiers without importing a model or CUDA.

This is a deterministic source audit, not a quality/context/performance gate.
Use isolated pristine worktrees at the revisions in the source lock.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from glm53_contract import sha256_file, strict_json


def isolated_functions(path, functions, methods=()):
    """Execute exact AST function bodies; exclude all upstream module imports."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in functions]
    for class_name, method_name in methods:
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected.append(next(node for node in cls.body
                             if isinstance(node, ast.FunctionDef) and node.name == method_name))
    if len(selected) != len(functions) + len(methods):
        raise ValueError("source function coverage changed")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    unit = ast.Module(body=[future, *selected], type_ignores=[])
    return compile(ast.fix_missing_locations(unit), str(path), "exec")


def verify_sources(source_root, lock):
    observations = {}
    for name, entry in lock["sources"].items():
        path = source_root / name
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if revision != entry["revision"]:
            raise ValueError(f"{name}: source revision mismatch")
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        if dirty:
            raise ValueError(f"{name}: source worktree is not clean")
        if not entry["files"]:
            raise ValueError(f"{name}: empty source inventory")
        for item in entry["files"]:
            if sha256_file(path / item["path"]) != item["sha256"]:
                raise ValueError(f"{name}: audited source digest mismatch: {item['path']}")
        observations[name] = revision
    return observations


def probe(source_root):
    # NumPy is a CPU audit dependency. It is not the serving runtime.
    import numpy as np
    root = source_root / "vllm/vllm"
    attention = root / "model_executor/layers/attention/mla_attention.py"
    namespace = {"is_quantized_kv_cache": lambda value: value.startswith("fp8")}
    exec(isolated_functions(attention, ["_canonicalize_sparse_mla_kv_cache_dtype"]), namespace)
    backend = type("SparkSparse", (), {"get_name": staticmethod(lambda: "FLASHINFER_MLA_SPARSE_SM120")})
    for requested in ("auto", "fp8"):
        resolved = namespace["_canonicalize_sparse_mla_kv_cache_dtype"](backend, requested)
        yield {"probe": "kv_canonicalization", "requested": requested, "resolved": resolved,
               "faithful_kv": False if resolved.startswith("fp8") else None}

    processor = root / "transformers_utils/processors/glm5next.py"
    namespace = {"np": np, "math": math, "GLM_VIDEO_DEFAULT_MAX_FRAMES": 2048,
                 "GLM_VIDEO_DEFAULT_FPS": 2.0}
    exec(isolated_functions(processor, ["glm_sample_frame_indices"],
                            [("Glm5NextVideoProcessor", "sample_frames")]), namespace)
    metadata = SimpleNamespace(total_num_frames=18000, fps=30, duration=600)
    for dynamic, overrides in ((2048, {"num_frames": 16}),
                               (16, {"max_frames": 16}),
                               (16, {"max_frames": 2048})):
        receiver = SimpleNamespace(fps_interval=2.0, max_frame_count_dynamic=dynamic,
                                   temporal_patch_size=2)
        indices = namespace["sample_frames"](receiver, metadata, **overrides)
        yield {"probe": "actual_video_sampler", "configured_dynamic_frames": dynamic,
               "request_overrides": overrides, "sampled_frames": len(indices),
               "within_sixteen_frames": len(indices) <= 16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive output creation preserves every attempt, including failures.
    args.output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / "configs/build-manifests/glm53-flash-sources.json"
    lock = strict_json(lock_path)
    manifest = {
        "kind": "deterministic_source_audit", "model_loaded": False,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_lock_sha256": sha256_file(lock_path),
        "scorer_sha256": sha256_file(Path(__file__)),
        "binary_sha256": sha256_file(Path(sys.executable).resolve()),
        "configuration_sha256": sha256_file(ROOT / "configs/profiles/glm-5.3-flash/cuda-spark-128g-1m.json"),
        "public_randomness": "not applicable: deterministic source falsifier; no authoritative model gate",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    rows, failures = [], []
    try:
        manifest["sources"] = verify_sources(args.source_root, lock)
        for row in probe(args.source_root):
            row["observed_utc"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
        # Identity is checked again after execution of the bounded bodies.
        verify_sources(args.source_root, lock)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    raw = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    (args.output / "raw.jsonl").write_text(raw)
    summary = {
        "kind": "deterministic_source_audit", "model_loaded": False,
        "verdict": "FAIL" if failures else "NO_RESULT",
        "runtime_qualification": "not attempted",
        "failures": failures,
        "checks": {"source_identity": not failures, "probe_count": len(rows)},
        "conclusion": "auto KV is not a faithful baseline on this pinned Spark sparse backend; num_frames alone is not a hard video cap",
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
