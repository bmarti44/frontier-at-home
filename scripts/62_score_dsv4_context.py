#!/usr/bin/env python3
"""Build and score the strict DeepSeek W11 evidence triplet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
LIVE_ROOT = Path("/home/bmarti44/spark-deepseek-v4-flash")
BINARY = Path(
    "/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server"
)
MODEL_ROOT = LIVE_ROOT / "weights" / "unsloth-ud-q2_k_xl"
TOKENIZER_SHA256 = (
    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8"
    "392edb4e894fd10cc0d70b4957c8633cf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hash_as_dsv4(path: Path) -> str:
    """Hash a protected engine artifact through the narrow service delegation."""
    completed = subprocess.run(
        ["sudo", "-n", "-u", "dsv4", "sha256sum", "--", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    digest = completed.stdout.split(maxsplit=1)[0]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"invalid protected artifact digest: {path}")
    return digest


def validate_stage_lineage(
    stage: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Reject stage-supplied fixtures that differ from seed regeneration."""
    expected_fields = {
        "seed_sha256": expected["payload"]["seed"].to_bytes(4, "big").hex(),
        "target_tokens": None,
        "fixture_sha256": expected["fixture"]["fixture_sha256"],
        "records": expected["fixture"]["records"],
        "absent_value_sha256": expected["absent_value_sha256"],
        "request_sha256": expected["request_sha256"],
    }
    # Seed is compared by the caller's full seed because the request carries
    # only its deterministic 32-bit sampling projection.
    expected_fields["seed_sha256"] = stage.get("_expected_seed_sha256")
    expected_fields["target_tokens"] = stage.get("_expected_target_tokens")
    actual = {
        "seed_sha256": stage.get("seed_sha256"),
        "target_tokens": stage.get("target_tokens"),
        "fixture_sha256": stage.get("fixture_sha256"),
        "records": stage.get("records"),
        "absent_value_sha256": stage.get("absent_value_sha256"),
        "request_sha256": stage.get(
            "request_sha256",
            stage.get("response", {}).get("request_sha256")
            if isinstance(stage.get("response"), dict)
            else None,
        ),
    }
    # Unit callers may omit private expected fields; then their stage values
    # define only the already supplied seed/target test parameters.
    for field in ("seed_sha256", "target_tokens"):
        if expected_fields[field] is None:
            expected_fields[field] = actual[field]
    if actual != expected_fields:
        raise RuntimeError("stage lineage differs from regenerated fixture")


def write_failure_triplet(
    *,
    out: Path,
    candidate: str,
    seed: str,
    mode: str,
    error: Exception,
) -> None:
    """Preserve mandatory artifacts even when strict scoring cannot start."""
    failure = {
        "record_type": "context_failure",
        "candidate_hash": candidate,
        "seed_sha256": seed,
        "mode": mode,
        "failure_events": [
            {"event": f"{type(error).__name__}: {error}"}
        ],
    }
    manifest = {
        "schema_version": 2,
        "gate": "dsv4_reference_context",
        "qualification_authority": False,
        "candidate_hash": candidate,
        "seed_sha256": seed,
        "mode": mode,
        "scorer_id": "w11.context.v1",
        "scorer_sha256": sha256(Path(__file__).resolve()),
    }
    summary = {
        "scorer_id": "w11.context.v1",
        "verdict": "FAIL",
        "error": f"{type(error).__name__}: {error}",
    }
    (out / "raw.jsonl").write_text(
        json.dumps(failure, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    for name, value in (("manifest", manifest), ("summary", summary)):
        (out / f"{name}.json").write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_freeze(root: Path, candidate: str) -> dict[str, str]:
    manifest_path = root / "freeze-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_hash") != candidate:
        raise RuntimeError("frozen candidate hash does not match")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("frozen artifact manifest is empty")
    resolved = subprocess.run(
        [
            "git",
            "-C",
            str(LIVE_ROOT),
            "rev-parse",
            "--verify",
            f"{candidate}^{{commit}}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if resolved != candidate:
        raise RuntimeError("Git candidate does not resolve exactly")
    tree_output = subprocess.run(
        ["git", "-C", str(LIVE_ROOT), "ls-tree", "-rz", candidate],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    expected_blobs: dict[str, str] = {}
    for entry in tree_output.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise RuntimeError("Git candidate contains unsupported artifact type")
        expected_blobs[raw_path.decode()] = object_id
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_paths = set(expected_blobs) | {
        "vendor/official-encoding/tokenizer.json",
        "freeze-manifest.json",
    }
    added = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if added:
        raise RuntimeError(f"frozen candidate contains unlisted files: {added}")
    if missing:
        raise RuntimeError(f"frozen candidate is missing files: {missing}")
    for relative, object_id in expected_blobs.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Git candidate artifact type changed: {relative}")
        raw = path.read_bytes()
        blob_hash = hashlib.sha1(
            f"blob {len(raw)}\0".encode() + raw,
            usedforsecurity=False,
        ).hexdigest()
        if blob_hash != object_id:
            raise RuntimeError(f"artifact differs from Git candidate: {relative}")
    tokenizer_path = root / "vendor/official-encoding/tokenizer.json"
    if tokenizer_path.is_symlink() or sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise RuntimeError("frozen tokenizer differs from registered artifact")
    artifact_paths = expected_paths - {"freeze-manifest.json"}
    if set(artifacts) != artifact_paths:
        raise RuntimeError("freeze manifest artifact set differs from Git candidate")
    for relative, expected in artifacts.items():
        path = root / relative
        if sha256(path) != expected:
            raise RuntimeError(f"frozen artifact manifest changed: {relative}")
    return artifacts


def build_observation(
    out: Path,
    candidate: str,
    seed: str,
    mode: str,
    lifecycle_exit_status: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode != "graduated":
        raise RuntimeError("only graduated mode can produce W11 evidence")
    artifacts = verify_freeze(ROOT, candidate)
    goal = load_module("glm52_goal_frozen", ROOT / "scripts" / "glm52_goal.py")
    probe = load_module(
        "dsv4_context_probe_frozen", ROOT / "scripts" / "57_dsv4_context_probe.py"
    )
    tokenizer = probe.load_tokenizer()

    caps = (131_072, 262_144, 524_288, 1_048_576)
    targets = (130_000, 260_000, 520_000, 1_000_000)
    stages = []
    retrieval_results = []
    fixture_hashes = []
    for index, (cap, target) in enumerate(zip(caps, targets)):
        stage_path = out / f"stage-{cap}.json"
        engine_path = out / f"engine-{cap}.log"
        if not stage_path.is_file() or not engine_path.is_file():
            raise RuntimeError(f"missing stage evidence for {cap}")
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("pass") is not True or stage.get("context_cap") != cap:
            raise RuntimeError(f"stage {cap} did not pass strict probe")
        expected = probe.build_request_artifacts(
            tokenizer, target=target, seed_sha256=seed
        )
        expected["_expected_seed_sha256"] = seed
        expected["_expected_target_tokens"] = target
        lineage_stage = dict(stage)
        lineage_stage["_expected_seed_sha256"] = seed
        lineage_stage["_expected_target_tokens"] = target
        validate_stage_lineage(lineage_stage, expected)
        response = stage["response"]
        progress = probe.parse_engine_progress(engine_path.read_text(encoding="utf-8"))
        usage_tokens = stage["processed_tokens"]
        probe.require_token_count_agreement(
            requested_tokens=target,
            usage_tokens=usage_tokens,
            engine_tokens=progress["evaluated_tokens"],
        )
        if response["finish_reason"] != "stop" or stage.get("truncated") is not False:
            raise RuntimeError(f"stage {cap} was truncated")
        timestamps = response["event_timestamps"]
        completed = stage["completed_output_tokens"]
        if len(timestamps) != completed:
            raise RuntimeError(f"stage {cap} lacks per-token timestamps")
        content = response["content"]
        stages.append(
            {
                "context_cap": cap,
                "processed_tokens": progress["evaluated_tokens"],
                "started_at_seconds": response["started_at_seconds"],
                "finished_at_seconds": response["finished_at_seconds"],
                "completed_output_tokens": completed,
                "token_timestamps": timestamps,
                "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "finish_reason": "stop",
                "truncated": False,
            }
        )
        fixture_hashes.append(stage["fixture_sha256"])
        if index == len(caps) - 1:
            absent_value = expected["fixture"]["absent_value"]
            strict_retrieval = probe.validate_retrieval(
                content,
                expected["fixture"]["records"],
                absent_value=absent_value,
            )
            if strict_retrieval["pass"] is not True:
                raise RuntimeError("final retrieval differs from regenerated fixture")
            retrieval_results = [
                {
                    "case_id": record["case_id"],
                    "position": record["position"],
                    "expected_sha256": record["expected_sha256"],
                    "observed_sha256": record["expected_sha256"],
                }
                for record in expected["fixture"]["records"]
            ]
            negative_digest = hashlib.sha256(b"").hexdigest()
            negative_results = [
                {
                    "case_id": "seeded-absent-0",
                    "expected_sha256": negative_digest,
                    "observed_sha256": hashlib.sha256(
                        (absent_value if absent_value in content else "").encode()
                    ).hexdigest(),
                }
            ]

    memory = []
    for line_number, line in enumerate(
        (out / "memory.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        sample = json.loads(line)
        if (
            not isinstance(sample.get("timestamp_seconds"), (int, float))
            or not math.isfinite(float(sample["timestamp_seconds"]))
        ):
            raise RuntimeError(f"invalid memory sample {line_number}")
        memory.append(sample)
    kernel = (out / "kernel.log").read_text(encoding="utf-8", errors="replace")
    fault_terms = (
        "nv_err_no_memory",
        "nvrm: xid",
        "oom-kill",
        "out of memory: killed process",
    )
    failure_events = [
        {"event": "stage or lifecycle failure"}
        for path in out.glob("stage-*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("pass") is not True
    ]
    if lifecycle_exit_status != 0:
        failure_events.append(
            {
                "event": "qualification lifecycle failed",
                "exit_status": lifecycle_exit_status,
            }
        )
    oom_events = (
        [{"event": "kernel OOM"}]
        if any(
            term in kernel.lower()
            for term in (
                "oom-kill",
                "out of memory: killed process",
                "killed process ",
            )
        )
        else []
    )
    xid_events = [
        {"event": "kernel GPU Xid"}
    ] if any(term in kernel.lower() for term in fault_terms[:2]) else []

    build_manifest_path = ROOT / "configs/build-manifests/llamacpp-fusion.json"
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    binary_hash = hash_as_dsv4(BINARY)
    if binary_hash != build_manifest["binaries"]["llama-server"]["sha256"]:
        raise RuntimeError("live binary changed")
    weights_manifest = json.loads(
        (ROOT / "weights/unsloth-ud-q2_k_xl/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for entry in weights_manifest["files"]:
        path = MODEL_ROOT / entry["name"]
        if hash_as_dsv4(path) != entry["sha256"]:
            raise RuntimeError(f"model shard changed: {entry['name']}")
    model_hash = aggregate(
        [(entry["name"], entry["sha256"]) for entry in weights_manifest["files"]]
    )
    tokenizer_hash = sha256(ROOT / "vendor/official-encoding/tokenizer.json")
    configuration = {
        "candidate_hash": candidate,
        "seed_sha256": seed,
        "mode": mode,
        "caps": list(caps),
        "targets": list(targets),
        "batch": 512,
        "ubatch": 256,
        "parallel": 1,
        "no_mmap": True,
        "memory_floor_gib": 14,
        "swap_max_bytes": 0,
        "runtime_max_seconds": 43_200,
        "build_manifest_sha256": sha256(build_manifest_path),
    }
    observation = {
        "record_type": "context_observation",
        "binary_sha256": binary_hash,
        "configuration_sha256": aggregate(configuration),
        "model_sha256": model_hash,
        "tokenizer_sha256": tokenizer_hash,
        "fixture_sha256": aggregate(fixture_hashes),
        "stages": stages,
        "retrieval_results": retrieval_results,
        "negative_control_results": negative_results,
        "memory_samples": memory,
        "failure_events": failure_events,
        "oom_events": oom_events,
        "xid_events": xid_events,
    }
    summary = goal.score_registered_gate("W11", "w11.context.v1", [observation])
    manifest = {
        "schema_version": 2,
        "gate": "dsv4_reference_context",
        "qualification_authority": True,
        "candidate_hash": candidate,
        "seed_sha256": seed,
        "scorer_id": "w11.context.v1",
        "binary_sha256": binary_hash,
        "configuration_sha256": observation["configuration_sha256"],
        "model_sha256": model_hash,
        "tokenizer_sha256": tokenizer_hash,
        "fixture_sha256": observation["fixture_sha256"],
        "scorer_sha256": artifacts["scripts/glm52_goal.py"],
        "probe_sha256": artifacts["scripts/57_dsv4_context_probe.py"],
        "worker_sha256": artifacts["scripts/58_dsv4_context_worker.sh"],
        "launcher_sha256": artifacts["scripts/21_serve_llamacpp.sh"],
        "scheduler_sha256": artifacts["scripts/60_schedule_dsv4_context_user.sh"],
        "frozen_artifacts": artifacts,
        "configuration": configuration,
    }
    return observation, {"manifest": manifest, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--candidate-hash", required=True)
    parser.add_argument("--seed-sha256")
    parser.add_argument("--mode")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--lifecycle-exit-status", type=int, default=0)
    args = parser.parse_args()
    if args.verify_only:
        verify_freeze(ROOT, args.candidate_hash)
        return 0
    if args.out is None or args.seed_sha256 is None or args.mode is None:
        parser.error("--out, --seed-sha256 and --mode are required for scoring")
    try:
        observation, result = build_observation(
            args.out,
            args.candidate_hash,
            args.seed_sha256,
            args.mode,
            args.lifecycle_exit_status,
        )
        (args.out / "raw.jsonl").write_text(
            json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        for name in ("manifest", "summary"):
            (args.out / f"{name}.json").write_text(
                json.dumps(
                    result[name], sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
        return 0 if result["summary"]["verdict"] == "PASS" else 1
    except Exception as error:
        write_failure_triplet(
            out=args.out,
            candidate=args.candidate_hash,
            seed=args.seed_sha256,
            mode=args.mode,
            error=error,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
