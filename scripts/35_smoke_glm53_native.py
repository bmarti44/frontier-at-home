#!/usr/bin/env python3
"""Run selected pinned upstream native correctness assertions without pytest.

Synthetic kernels only: this is neither model fidelity nor context/performance
qualification. Missing extensions and skipped checks fail. Run only through
the GLM containment wrapper, after all CUDA builds have finished.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from glm53_contract import sha256_file, strict_json


class RequiredChecks:
    @staticmethod
    def skip(message):
        raise RuntimeError("required native check cannot skip: " + message)

    @staticmethod
    def raises(exception, match):
        return unittest.TestCase().assertRaisesRegex(exception, match)


def load_functions(path, names, namespace):
    """Execute exact function bodies, dropping only pytest collection decorators.

    No upstream module initialization, optional-import fallback, test discovery,
    or bytecode write is involved. The complete input file is hashed separately.
    """
    source = ast.parse(path.read_bytes(), filename=str(path))
    selected = []
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
    found = {node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found.update(target.id for node in selected if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name))
    if found != set(names):
        raise ValueError("upstream smoke fixture names changed")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="post-freeze public seed used to permute check order")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plugin = args.prepared.resolve() / "vllm-exl3"
    lock = strict_json(ROOT / "configs/build-manifests/glm53-flash-sources.json")
    revision = lock["sources"]["vllm-exl3"]["revision"]
    files = {name: plugin / "tests" / name for name in ("test_exl3_linear.py", "test_native_moe_contract.py")}
    for name, path in files.items():
        original = subprocess.check_output(["git", "-C", str(plugin), "show", revision + ":tests/" + name])
        if path.read_bytes() != original:
            raise ValueError("upstream test source differs from pinned revision")
    manifest = {"schema_version": 1, "qualification": "synthetic_native_smoke_only",
                "source_revision": revision, "scorer_sha256": sha256_file(Path(__file__)),
                "binary_sha256": sha256_file(Path(sys.executable).resolve()),
                "test_hashes": {name: sha256_file(path) for name, path in files.items()},
                "seed": args.seed, "seed_use": "check order only; upstream fixture seeds retained",
                "start_unix": time.time(), "expected_checks": 14}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    completed = []
    with (output / "raw.jsonl").open("w") as raw, (output / "assertion-output.log").open("w") as log:
        def record(value):
            raw.write(json.dumps({"time_unix": time.time(), **value}, allow_nan=False) + "\n")
            raw.flush()
        try:
            for name in ("exllamav3_ext", "vllm_exl3_c"):
                spec = importlib.util.find_spec(name)
                if spec is None or not spec.origin.endswith(".so"):
                    raise ValueError("missing required AOT native extension: " + name)
                record({"native_extension": name, "path": spec.origin, "sha256": sha256_file(Path(spec.origin))})
            import torch
            import exllamav3_ext
            import vllm_exl3_c as native
            import vllm_exl3.exl3 as exl3
            if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
                raise ValueError("required GB10 SM121 device is unavailable")
            if not callable(exllamav3_ext.exl3_moe) or getattr(native, "P2B_MOE_ABI_VERSION", 0) < 2:
                raise ValueError("required native MoE ABI is missing")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            namespace = {"torch": torch, "math": math, "pytest": RequiredChecks, "exl3": exl3}
            dense = {"MCG_MARKER", "MUL1_MARKER", "K", "IN_FEATURES", "SHARDS", "_signs", "_rand_trellis",
                     "_reference_weight", "_patch_tp", "_run_layer", "test_exl3_linear_basic",
                     "test_exl3_linear_mixed_mul1", "test_exl3_linear_tp_slicing"}
            load_functions(files["test_exl3_linear.py"], dense, namespace)
            load_functions(files["test_native_moe_contract.py"],
                           {"test_native_moe_clipping_width_and_graph_parity", "test_native_moe_rejects_unsafe_calls_before_launch"}, namespace)
            checks = [(name, namespace[name], ()) for name in
                      ("test_exl3_linear_basic", "test_exl3_linear_mixed_mul1", "test_exl3_linear_tp_slicing")]
            for bits in (2, 3, 4):
                for width in (1024, 2048):
                    checks.append((f"native_moe_k{bits}_width{width}", namespace["test_native_moe_clipping_width_and_graph_parity"],
                                   (native, bits, width, SimpleNamespace(setattr=setattr))))
            for bad in ("rows", "width", "negative_limit", "nan_limit", "strided_pointers"):
                checks.append(("reject_" + bad, namespace["test_native_moe_rejects_unsafe_calls_before_launch"], (native, bad)))
            random.Random(args.seed).shuffle(checks)
            record({"check_order": [name for name, _, _ in checks]})
            for name, function, arguments in checks:
                record({"check_id": name, "event": "start"})
                with contextlib.redirect_stdout(log):
                    function(*arguments)
                torch.cuda.synchronize()
                log.flush()
                completed.append(name)
                record({"check_id": name, "event": "pass"})
            if len(completed) != manifest["expected_checks"]:
                raise ValueError("required native checks are missing")
            summary = {"verdict": "PASS", "checks_completed": len(completed)}
        except Exception as error:
            traceback.print_exc(file=log)
            record({"event": "failure", "failure": repr(error)})
            summary = {"verdict": "FAIL", "checks_completed": len(completed), "failure": repr(error)}
    summary.update(qualification="synthetic_native_smoke_only", model_loaded=False,
                   model_fidelity="not measured", context_capacity="not measured", performance="not measured",
                   raw_sha256=sha256_file(output / "raw.jsonl"),
                   test_output_sha256=sha256_file(output / "assertion-output.log"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
