#!/usr/bin/env python3
"""Generate the estimated off-host profile files (docs/PROFILE-SCHEMA.md).

Reads the weights sizes from configs/profiles/*/model.json and the KV rates
from configs/hardware-matrix.json, renders one static JSON per feasible
(model, backend, tier) cell, and writes them under configs/profiles/.
The generated files are committed as static JSON so review sees the
numbers; re-running regenerates deterministically.

Usage: scripts/dev/gen_profiles.py [--check]
  --check  exit non-zero if any committed file differs from regeneration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "configs/profiles"
MATRIX = json.loads((ROOT / "configs/hardware-matrix.json").read_text())

BASIS = (
    "computed 2026-08-27 from weights manifests + measured KV rates "
    "(configs/hardware-matrix.json); estimated until qualified on hardware "
    "per docs/QUALIFY-OFFHOST.md"
)

KV = {model: entry["f16"] for model, entry in MATRIX["kv_bytes_per_token"].items()}


def qwen_args(ctx, ngl, parallel, mmproj, uma, moe_cpu=False):
    args = ["--model", "{model}", "-ngl", str(ngl), "-fa", "on"]
    if uma:
        args.append("--no-mmap")
    args += ["-c", str(ctx)]
    if mmproj:
        args += ["--mmproj", "{mmproj}"]
    if moe_cpu:
        args += ["-ot", "exps=CPU"]
    args += [
        "--parallel", str(parallel), "--host", "127.0.0.1", "--port", "{port}",
        "--alias", "qwen3.8-27b",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "8",
        "--spec-draft-p-min", "0.6",
        "--chat-template-kwargs", "{\"reasoning_effort\":\"low\"}",
        "--cache-reuse", "256",
    ]
    return args


def laguna_args(ctx, parallel, draft, uma, moe_cpu=False):
    args = ["--model", "{model}", "-ngl", "99", "-fa", "on"]
    if uma:
        args.append("--no-mmap")
    args += ["-c", str(ctx)]
    if draft:
        args += ["-md", "{draft_model}"]
    if moe_cpu:
        args += ["-ot", "exps=CPU"]
    args += [
        "--parallel", str(parallel), "--host", "127.0.0.1", "--port", "{port}",
        "--alias", "laguna-s-2.1",
    ]
    if draft:
        args += ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4"]
    args += [
        "--jinja",
        "--chat-template-kwargs", "{\"enable_thinking\":true}",
        "--cache-reuse", "256",
    ]
    return args


def dsv4_args(ctx, ngl, parallel, uma, moe_cpu=False):
    # llamacpp-base pin: the fused hyper-connection ops are CUDA-only, so
    # every off-Spark profile runs the pre-fusion decode baseline.
    args = ["--model", "{model}", "-ngl", str(ngl), "-fa", "on"]
    if uma:
        args.append("--no-mmap")
    args += ["-c", str(ctx)]
    if moe_cpu:
        args += ["-ot", "exps=CPU"]
    args += [
        "--parallel", str(parallel), "--host", "127.0.0.1", "--port", "{port}",
        "--alias", "deepseek-v4-flash", "--cache-reuse", "256",
    ]
    return args


# (backend, class, tier, filename-stem, artifact_roles, args, ctx, extras)
SPECS = []


def spec(model, engine, backend, hardware_class, tier, stem, roles, args, ctx,
         floor, verify=(), tight_note=None, min_system_ram=None,
         decode_baseline=None):
    SPECS.append({
        "model": model, "engine": engine, "backend": backend,
        "class": hardware_class, "tier": tier, "stem": stem, "roles": roles,
        "args": args, "ctx": ctx, "floor": floor, "verify": list(verify),
        "tight_note": tight_note, "min_system_ram": min_system_ram,
        "decode_baseline": decode_baseline,
    })


Q = "qwen3.8-27b"
QE = "llamacpp-qwen38-portable"
METAL_MTP = "metal-mtp-draft-behavior"

spec(Q, QE, "apple-silicon", "mac", 32, "apple-silicon-32g",
     {"model": "q4km", "mmproj": "mmproj-f16"},
     qwen_args(32768, 99, 1, True, True), 32768, 10, [METAL_MTP])
spec(Q, QE, "apple-silicon", "mac", 48, "apple-silicon-48g",
     {"model": "q6k", "mmproj": "mmproj-f16"},
     qwen_args(65536, 99, 1, True, True), 65536, 10, [METAL_MTP])
spec(Q, QE, "apple-silicon", "mac", 64, "apple-silicon-64g",
     {"model": "q8", "mmproj": "mmproj-f16"},
     qwen_args(131072, 99, 1, True, True), 131072, 10, [METAL_MTP])
spec(Q, QE, "apple-silicon", "mac", 96, "apple-silicon-96g",
     {"model": "q8", "mmproj": "mmproj-f16"},
     qwen_args(262144, 99, 1, True, True), 262144, 10, [METAL_MTP])
spec(Q, QE, "apple-silicon", "mac", 128, "apple-silicon-128g",
     {"model": "q8", "mmproj": "mmproj-f16"},
     qwen_args(262144, 99, 2, True, True), 262144, 10, [METAL_MTP])
spec(Q, QE, "apple-silicon", "mac", 192, "apple-silicon-192g",
     {"model": "q8", "mmproj": "mmproj-f16"},
     qwen_args(262144, 99, 4, True, True), 262144, 10, [METAL_MTP])
spec(Q, QE, "cuda", "dgpu", 8, "cuda-dgpu-8g", {"model": "q4km"},
     qwen_args(16384, 24, 1, False, False), 16384, 8,
     ["dgpu-ngl-calibration"], min_system_ram=32)
spec(Q, QE, "cuda", "dgpu", 12, "cuda-dgpu-12g", {"model": "q4km"},
     qwen_args(32768, 36, 1, False, False), 32768, 8,
     ["dgpu-ngl-calibration"], min_system_ram=32)
spec(Q, QE, "cuda", "dgpu", 16, "cuda-dgpu-16g", {"model": "q4km"},
     qwen_args(32768, 46, 1, False, False), 32768, 8,
     ["dgpu-ngl-calibration"], min_system_ram=32)
spec(Q, QE, "cuda", "dgpu", 24, "cuda-dgpu-24g",
     {"model": "q4km", "mmproj": "mmproj-f16"},
     qwen_args(32768, 99, 1, False, False), 32768, 8, min_system_ram=32)
spec(Q, QE, "cuda", "dgpu", 32, "cuda-dgpu-32g",
     {"model": "q6k", "mmproj": "mmproj-f16"},
     qwen_args(65536, 99, 1, False, False), 65536, 8, min_system_ram=32)
spec(Q, QE, "rocm", "strix", 64, "rocm-strix-64g", {"model": "q6k"},
     qwen_args(131072, 99, 1, True, True), 131072, 8, ["rocm-hip-build"])
spec(Q, QE, "rocm", "strix", 96, "rocm-strix-96g", {"model": "q8"},
     qwen_args(262144, 99, 1, True, True), 262144, 8, ["rocm-hip-build"])
spec(Q, QE, "rocm", "strix", 128, "rocm-strix-128g", {"model": "q8"},
     qwen_args(262144, 99, 2, True, True), 262144, 8, ["rocm-hip-build"])
spec(Q, QE, "cpu", "any", 32, "cpu-32g", {"model": "q4km"},
     qwen_args(16384, 0, 1, False, False), 16384, 8,
     ["cpu-throughput (dense 27B: expect low single-digit tok/s)"])
spec(Q, QE, "cpu", "any", 64, "cpu-64g", {"model": "q8"},
     qwen_args(32768, 0, 1, False, False), 32768, 8,
     ["cpu-throughput (dense 27B: expect low single-digit tok/s)"])

L = "laguna-s-2.1"
LE = "llamacpp-laguna-portable"
FORK = "poolside-fork-build (no known Metal/HIP CI on the laguna branch)"

spec(L, LE, "apple-silicon", "mac", 128, "apple-silicon-128g",
     {"model": "udq4kxl", "draft_model": "dflash-bf16"},
     laguna_args(393216, 4, True, True), 393216, 10,
     [FORK, "metal-dflash-draft-behavior"],
     tight_note="mirrors the Spark-qualified 4x98304 shape; ~93 GiB fit "
                "against ~96 GiB usable")
spec(L, LE, "apple-silicon", "mac", 192, "apple-silicon-192g",
     {"model": "udq4kxl", "draft_model": "dflash-bf16"},
     laguna_args(524288, 1, True, True), 524288, 10,
     [FORK, "metal-dflash-draft-behavior"],
     tight_note="524288 single-slot; the Spark declined this for "
                "watchdog-floor reasons that do not bind at 192 GiB")
spec(L, LE, "cuda", "dgpu", 16, "cuda-dgpu-16g", {"model": "udq4kxl"},
     laguna_args(32768, 1, False, False, moe_cpu=True), 32768, 8,
     ["moe-on-cpu-throughput"], min_system_ram=128)
spec(L, LE, "cuda", "dgpu", 24, "cuda-dgpu-24g", {"model": "udq4kxl"},
     laguna_args(65536, 1, False, False, moe_cpu=True), 65536, 8,
     ["moe-on-cpu-throughput"], min_system_ram=128)
spec(L, LE, "rocm", "strix", 128, "rocm-strix-128g", {"model": "udq4kxl"},
     laguna_args(131072, 1, True, True), 131072, 8, [FORK])
spec(L, LE, "cpu", "any", 96, "cpu-96g", {"model": "udq4kxl"},
     laguna_args(32768, 1, False, False), 32768, 8,
     ["cpu-throughput (8B-active MoE: usable but slow)"])
spec(L, LE, "cpu", "any", 128, "cpu-128g", {"model": "udq4kxl"},
     laguna_args(131072, 1, False, False), 131072, 8,
     ["cpu-throughput (8B-active MoE: usable but slow)"])

D = "deepseek-v4-flash"
DE = "llamacpp-base"
PREFUSION = "pre-fusion"

spec(D, DE, "apple-silicon", "mac", 128, "apple-silicon-128g",
     {"model": "udq2kxl"}, dsv4_args(262144, 99, 1, True), 262144, 10,
     ["raised-wired-limit (fit ~95 GiB against ~96 GiB usable)"],
     tight_note="90.2 GiB weights leave almost no headroom at the default "
                "Metal wired limit", decode_baseline=PREFUSION)
spec(D, DE, "apple-silicon", "mac", 192, "apple-silicon-192g",
     {"model": "udq2kxl"}, dsv4_args(1048576, 99, 2, True), 1048576, 10,
     [], decode_baseline=PREFUSION)
spec(D, DE, "cuda", "dgpu", 12, "cuda-dgpu-12g",
     {"model": "udq2kxl"}, dsv4_args(32768, 99, 1, False, moe_cpu=True),
     32768, 8, ["moe-on-cpu-throughput (13B active streamed from host RAM)"],
     min_system_ram=128, decode_baseline=PREFUSION,
     tight_note="one profile covers 12-32 GiB VRAM: dense layers on GPU, "
                "experts on CPU")
spec(D, DE, "rocm", "strix", 128, "rocm-strix-128g",
     {"model": "udq2kxl"}, dsv4_args(65536, 99, 1, True), 65536, 8,
     ["rocm-hip-build"], decode_baseline=PREFUSION)
spec(D, DE, "cpu", "any", 128, "cpu-128g",
     {"model": "udq2kxl"}, dsv4_args(16384, 0, 1, False), 16384, 8,
     ["cpu-throughput (13B-active MoE)"], decode_baseline=PREFUSION)


def render(entry) -> dict:
    model_doc = json.loads(
        (PROFILES / entry["model"] / "model.json").read_text()
    )
    weights = sum(
        float(model_doc["artifacts"][name].get("weights_gib", 0.0))
        for name in entry["roles"].values()
    )
    profile = {
        "schema_version": 4,
        "profile_id": f"{entry['model']}/{entry['stem']}",
        "model": entry["model"],
        "backend": entry["backend"],
        "hardware_class": entry["class"],
        "ram_tier_gib": entry["tier"],
        "status": {"state": "estimated", "basis": BASIS},
        "engine": entry["engine"],
        "artifact_roles": entry["roles"],
        "launch": {
            "mechanism": "setsid-watchdog-portable",
            "log_name": entry["stem"],
            "env": {},
            "diagnostics_unset": [],
            "args": entry["args"],
        },
        "safety": {
            "kill_floor_gib": entry["floor"],
            "minimum_start_gib": int(weights) + 10,
            "sample_hz": 1,
            "startup_timeout_seconds": 1800,
        },
        "memory_model": {
            "kv_bytes_per_token": KV[entry["model"]],
            "overhead_gib": 3.0,
            "extra_gib": 0,
            "floor_gib": entry["floor"],
            "basis": BASIS,
        },
        "context_cap": entry["ctx"],
        "port_role": "production",
        "bench": {
            "stack_label": f"{entry['model']}-{entry['backend']}-"
                           f"{entry['class']}-{entry['tier']}g",
        },
    }
    if entry["min_system_ram"] is not None:
        profile["min_system_ram_gib"] = entry["min_system_ram"]
    if entry["verify"]:
        profile["verify_on_hardware"] = entry["verify"]
    purpose = []
    if entry["tight_note"]:
        purpose.append(entry["tight_note"])
    if entry["decode_baseline"]:
        purpose.append(f"decode_baseline: {entry['decode_baseline']}")
    if purpose:
        profile["purpose"] = "; ".join(purpose)
    return profile


def main() -> int:
    check = "--check" in sys.argv[1:]
    dirty = []
    for entry in SPECS:
        path = PROFILES / entry["model"] / f"{entry['stem']}.json"
        rendered = json.dumps(render(entry), indent=1) + "\n"
        if check:
            if not path.exists() or path.read_text() != rendered:
                dirty.append(str(path))
        else:
            path.write_text(rendered)
            print(f"wrote {path.relative_to(ROOT)}")
    if dirty:
        print("stale generated profiles:", *dirty, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
