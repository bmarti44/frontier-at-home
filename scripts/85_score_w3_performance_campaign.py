#!/usr/bin/env python3
"""Fixed scorer for the five-block W3 completed-time campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any


EXPECTED_ORDERS = (
    "off-on", "on-off",  # ABBA
    "on-off", "off-on",  # BAAB
    "off-on", "on-off",  # ABBA
    "on-off", "off-on",  # BAAB
    "off-on", "on-off",  # ABBA
)
TOKEN_RE = re.compile(
    r"DS4_TOKEN_TIMING request=(\S+) index=(\d+) "
    r"monotonic_ns=(\d+) token=(-?\d+)"
)
T95_DF4 = 2.1318
_CLI_AUTHORITY_TOKEN = object()
ALLOWED_ENVIRONMENT = {
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
SUMMARY_KEYS = {
    "schema_version", "gate", "status", "scope", "acceptance_formula",
    "checks", "engine_commit", "binary_sha256", "model_sha256",
    "tokenizer_sha256", "repository_head", "freeze_sha256",
    "freeze_bindings", "environment_sha256", "request_sha256", "arm_order",
    "required_completion_tokens", "public_randomness", "arms",
}
MANIFEST_KEYS = {
    "schema_version", "engine_commit", "binary_sha256", "model_sha256",
    "tokenizer_sha256", "repository_head", "freeze_sha256", "freeze_bindings",
    "environment_sha256", "request_sha256", "arm_order",
    "required_completion_tokens", "public_randomness_round", "public_randomness",
    "public_randomness_signature", "harness_sha256", "artifact_sha256",
}
CHECK_KEYS = {
    "same_frozen_binary", "same_model", "same_request", "safe_returncodes_zero",
    "http_200", "independent_exact_output_tokens",
    "thinking_disabled_no_reasoning_channel", "all_generated_outputs_nonempty",
    "generated_output_byte_identical", "warm_generated_output_byte_identical",
    "off_path_not_mapped", "off_path_has_no_direct_dispatches", "on_path_mapped",
    "on_path_dispatched_for_compared_warm_response",
    "on_path_dispatched_for_compared_measured_response", "clean_exit_attested",
    "no_fault_markers",
}
ARM_KEYS = {
    "schema_version", "arm", "direct_requested", "safe_returncode",
    "warm_http_code", "warm_wall_seconds", "measured_http_code",
    "measured_wall_seconds", "completion_tokens", "warm_completion_tokens",
    "independent_completion_tokens", "independent_warm_completion_tokens",
    "measured_reasoning_bytes", "warm_reasoning_bytes", "measured_token_scorer",
    "warm_token_scorer", "generated_sha256", "generated_bytes", "mapping_markers",
    "admission_markers", "dispatch_markers", "warm_dispatch_delta",
    "measured_dispatch_delta", "clean_exit_attestation", "fault_markers",
    "environment_sha256", "request_sha256", "binary_sha256", "model_sha256",
    "tokenizer_sha256", "engine_commit", "crash_evidence",
    "crash_evidence_identity", "crash_artifact_sha256", "timing_receipt",
}
TOKEN_RECORD_KEYS = {
    "schema_version", "label", "reference_token_count", "content_bytes",
    "reasoning_bytes", "response_sha256", "response_identity", "tokenizer_sha256",
    "tokenizer_identity", "runtime_init_sha256", "runtime_init_identity",
    "runtime_native_sha256", "runtime_native_identity", "runtime_init_path",
    "runtime_native_path", "runtime_native_loaded_path",
}
PAIR_ARTIFACTS = {
    "output-directory.txt", "request.json", "summary.json",
    *(f"{arm}/{name}" for arm in ("off", "on") for name in (
        "arm.json", "containment.stderr", "containment.stdout",
        "measured.dispatch-counts", "measured.http", "measured.json",
        "measured.tokens.json", "memory-preflight.json", "warm.dispatch-counts",
        "warm.http", "warm.json", "warm.tokens.json",
    )),
}
FREEZE_KEYS = {
    "schema_version", "repository_parent_commit", "engine_commit", "binary_sha256",
    "model_sha256", "drand_floor_round", "campaign_contract", "artifacts",
    "engine_source_sha256",
}


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=strict_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return raw, value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stat_identity(path: Path) -> str:
    value = path.stat()
    return f"{value.st_dev}:{value.st_ino}"


def file_identity(path: Path) -> str:
    value = path.stat()
    return (f"{value.st_dev}:{value.st_ino}:{value.st_size}:"
            f"{value.st_mtime_ns}:{value.st_ctime_ns}")


def exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an exact integer")
    return value


def require_digest(value: Any, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(fr"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{label} is not a lowercase hexadecimal digest")
    return value


def committed_authority(repo: Path, freeze_path: Path) -> dict[str, Any]:
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout:
        raise ValueError("repository is not clean at campaign scoring")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD^"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    relative = freeze_path.relative_to(repo)
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repo,
        capture_output=True, check=True,
    ).stdout
    freeze_raw, freeze = strict_json(freeze_path)
    if committed != freeze_raw or set(freeze) != FREEZE_KEYS or freeze.get("schema_version") != 1:
        raise ValueError("campaign freeze is not the exact committed authority")
    if parent != freeze.get("repository_parent_commit"):
        raise ValueError("campaign freeze is not a freeze-only commit")
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD^", "HEAD"], cwd=repo,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    if changed != [str(relative)]:
        raise ValueError("campaign freeze commit changed additional files")
    artifact_paths = {
        "harness": repo / "results/glm52-gates/harness/w3_direct_slot_probe_v3.sh",
        "campaign_scorer": Path(__file__).resolve(),
        "token_scorer_sha256": repo / "scripts/84_count_glm_output_tokens.py",
        "cgroup": repo / "results/glm52-gates/harness/glm_cgroup_run.sh",
        "safe": repo / "results/glm52-gates/harness/glm_safe_run.sh",
        "memory_guard": repo / "scripts/03_memory_guard.py",
    }
    artifacts = freeze.get("artifacts")
    if (not isinstance(artifacts, dict) or set(artifacts) != {
            "harness", "campaign_scorer", "token_scorer_sha256", "cgroup", "safe",
            "memory_guard", "binary", "tokenizer_sha256", "tokenizers_init_sha256",
            "tokenizers_so_sha256"}):
        raise ValueError("freeze artifact inventory is malformed")
    if (artifacts["binary"] != freeze.get("binary_sha256") or
            not all(re.fullmatch(r"[0-9a-f]{64}", value or "")
                    for value in artifacts.values())):
        raise ValueError("freeze artifact digests are malformed")
    engine_source = freeze.get("engine_source_sha256")
    if (not isinstance(engine_source, dict) or set(engine_source) != {
            "ds4.c", "ds4_cuda.cu", "Makefile",
            "tests/test_gpu_expert_slot_dispatch_marker.sh",
            "tests/test_gpu_expert_slot_lifetime.c",
            "tests/test_gpu_expert_slot_gemv.c", "tests/test_instance_lock_safety.sh"} or
            not all(re.fullmatch(r"[0-9a-f]{64}", value or "")
                    for value in engine_source.values())):
        raise ValueError("freeze engine-source inventory is malformed")
    for name, path in artifact_paths.items():
        if artifacts.get(name) != sha256(path):
            raise ValueError(f"frozen authority artifact changed: {name}")
    contract = freeze.get("campaign_contract")
    if contract != {
        "completion_tokens": 129,
        "decode_formula": "128/(t129-t1)",
        "block_schedule": ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA"],
        "block_value": "arithmetic mean of the two same-arm completed seconds",
        "acceptance": "one-sided 95% upper bound of the geometric paired candidate/baseline completed-time ratio <= 0.95",
    }:
        raise ValueError("campaign contract differs from the fixed scorer")
    return {
        "freeze": freeze,
        "freeze_sha256": sha256_bytes(freeze_raw),
        "freeze_commit": head,
        "harness_sha256": artifacts["harness"],
    }


def authenticate_drand(record: dict[str, Any], floor: int) -> None:
    if set(record) != {"round", "randomness", "signature", "freeze_floor_round"}:
        raise ValueError("public randomness schema is invalid")
    round_number = exact_int(record["round"], "drand round")
    recorded_floor = exact_int(record["freeze_floor_round"], "drand freeze floor")
    randomness = require_digest(record["randomness"], "drand randomness")
    signature = require_digest(record["signature"], "drand signature", 192)
    if recorded_floor != floor or round_number <= floor:
        raise ValueError("drand record predates or changes the frozen floor")
    if hashlib.sha256(bytes.fromhex(signature)).hexdigest() != randomness:
        raise ValueError("drand signature does not derive the recorded randomness")
    expected = {"round": round_number, "randomness": randomness, "signature": signature}
    for host in ("api.drand.sh", "api2.drand.sh", "api3.drand.sh"):
        response = subprocess.run(
            ["/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail",
             "--max-time", "15", "--proto", "=https",
             f"https://{host}/public/{round_number}"],
            env=ALLOWED_ENVIRONMENT, capture_output=True, check=True,
        )
        published = json.loads(response.stdout, object_pairs_hook=strict_object)
        if any(published.get(key) != value for key, value in expected.items()):
            raise ValueError(f"drand relay disagreement: {host}")


def verified_manifest(pair: Path, summary_raw: bytes) -> dict[str, Any]:
    _, manifest = strict_json(pair / "manifest.json")
    if set(manifest) != MANIFEST_KEYS or manifest.get("schema_version") != 1:
        raise ValueError(f"{pair}: manifest schema is not exact")
    artifacts = manifest.get("artifact_sha256")
    if (not isinstance(artifacts, dict) or set(artifacts) != PAIR_ARTIFACTS or
            artifacts.get("summary.json") != sha256_bytes(summary_raw)):
        raise ValueError(f"{pair}: manifest does not bind summary.json")
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"{pair}: unsafe manifest artifact path")
        require_digest(expected, f"{pair}:{relative} digest")
        artifact = pair / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"{pair}: artifact digest mismatch: {relative}")
    return manifest


def measured_timing(cmd_path: Path, expected_sha: str) -> dict[str, Any]:
    require_digest(expected_sha, f"{cmd_path} expected digest")
    raw = cmd_path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"{cmd_path}: cmd.log digest mismatch")
    groups: list[tuple[str, list[tuple[int, int, int]]]] = []
    for raw_line in raw.decode("utf-8", errors="strict").splitlines():
        match = TOKEN_RE.fullmatch(raw_line)
        if match is None:
            if "DS4_TOKEN_TIMING" in raw_line:
                raise ValueError(f"{cmd_path}: malformed token timing marker")
            continue
        request = match.group(1)
        row = tuple(int(match.group(i)) for i in (2, 3, 4))
        if not groups or groups[-1][0] != request:
            if any(previous == request for previous, _ in groups):
                raise ValueError(f"{cmd_path}: interleaved token timing request")
            groups.append((request, []))
        groups[-1][1].append(row)
    if len(groups) != 2:
        raise ValueError(f"{cmd_path}: expected exactly warm and measured timing groups")
    for phase, (_, rows) in zip(("warm", "measured"), groups, strict=True):
        if len(rows) != 129 or [row[0] for row in rows] != list(range(1, 130)):
            raise ValueError(f"{cmd_path}: {phase} group is not exactly 129 ordered tokens")
        stamps = [row[1] for row in rows]
        if any(right <= left for left, right in zip(stamps, stamps[1:])):
            raise ValueError(f"{cmd_path}: {phase} timestamps are not strictly increasing")
    warm_request = groups[0][0]
    request, rows = groups[1]
    timestamps = [row[1] for row in rows]
    seconds = (timestamps[-1] - timestamps[0]) / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{cmd_path}: invalid completed decode time")
    return {
        "request_id": request,
        "warm_request_id": warm_request,
        "token_timestamps_ns": timestamps,
        "token_ids": [row[2] for row in rows],
        "warm_token_timestamps_ns": [row[1] for row in groups[0][1]],
        "warm_token_ids": [row[2] for row in groups[0][1]],
        "completed_seconds": seconds,
        "decode_tokens_per_second": 128.0 / seconds,
        "cmd_log_sha256": expected_sha,
        "cmd_log_identity": file_identity(cmd_path),
    }


def generated(payload: dict[str, Any]) -> dict[str, str]:
    message = payload["choices"][0]["message"]
    return {"content": message.get("content", ""),
            "reasoning_content": message.get("reasoning_content", "")}


def score_pair(pair: Path, expected_order: str, authority: dict[str, Any]) -> dict[str, Any]:
    summary_raw, summary = strict_json(pair / "summary.json")
    manifest = verified_manifest(pair, summary_raw)
    if (set(summary) != SUMMARY_KEYS or summary.get("schema_version") != 1 or
            summary.get("status") != "PASS" or summary.get("arm_order") != expected_order):
        raise ValueError(f"{pair}: pair status or randomized arm order is invalid")
    if exact_int(summary.get("required_completion_tokens"), f"{pair}: token count") != 129:
        raise ValueError(f"{pair}: pair does not require 129 completion tokens")
    checks = summary.get("checks")
    if (not isinstance(checks, dict) or set(checks) != CHECK_KEYS or
            any(value is not True for value in checks.values())):
        raise ValueError(f"{pair}: runtime eligibility checks did not all pass")
    frozen = authority["freeze"]
    if (summary.get("freeze_bindings") != frozen or
            manifest.get("freeze_bindings") != frozen or
            summary.get("freeze_sha256") != authority["freeze_sha256"] or
            summary.get("repository_head") != authority["freeze_commit"] or
            manifest.get("harness_sha256") != authority["harness_sha256"]):
        raise ValueError(f"{pair}: pair is not bound to the exact campaign freeze")
    bindings = ("binary_sha256", "model_sha256", "tokenizer_sha256", "freeze_sha256")
    for key in bindings:
        require_digest(summary.get(key), f"{pair}:{key}")
        if manifest.get(key) != summary[key]:
            raise ValueError(f"{pair}: manifest {key} mismatch")
    if (summary["binary_sha256"] != frozen["binary_sha256"] or
            summary["model_sha256"] != frozen["model_sha256"] or
            summary["tokenizer_sha256"] != frozen["artifacts"]["tokenizer_sha256"] or
            summary.get("engine_commit") != frozen["engine_commit"] or
            manifest.get("engine_commit") != frozen["engine_commit"]):
        raise ValueError(f"{pair}: candidate bindings differ from the campaign freeze")
    repository_head = require_digest(summary.get("repository_head"), f"{pair}:repository_head", 40)
    if manifest.get("repository_head") != repository_head:
        raise ValueError(f"{pair}: manifest repository head mismatch")
    request_sha = require_digest(summary.get("request_sha256"), f"{pair}:request_sha256")
    if manifest.get("request_sha256") != request_sha:
        raise ValueError(f"{pair}: manifest request digest mismatch")
    randomness = summary.get("public_randomness")
    if not isinstance(randomness, dict):
        raise ValueError(f"{pair}: public randomness is missing")
    authenticate_drand(randomness, frozen["drand_floor_round"])
    round_number = exact_int(randomness.get("round"), f"{pair}: randomness round")
    floor = exact_int(randomness.get("freeze_floor_round"), f"{pair}: freeze floor")
    require_digest(randomness.get("randomness"), f"{pair}: randomness")
    require_digest(randomness.get("signature"), f"{pair}: randomness signature", 192)
    if (round_number <= floor or manifest.get("public_randomness_round") != round_number or
            manifest.get("public_randomness") != randomness["randomness"] or
            manifest.get("public_randomness_signature") != randomness["signature"] or
            manifest.get("arm_order") != expected_order or
            manifest.get("required_completion_tokens") != 129):
        raise ValueError(f"{pair}: public randomness or campaign fields differ from manifest")
    request_raw, request = strict_json(pair / "request.json")
    if sha256_bytes(request_raw) != request_sha:
        raise ValueError(f"{pair}: request digest mismatch")
    expected_request = {
        "model": "glm-5.2",
        "messages": [{
            "role": "user",
            "content": (
                "Generate a deterministic sequence of exactly 200 lowercase letters "
                "by repeating the alphabet in order. Do not stop early. "
                f"Confirmation nonce: {randomness['randomness'][:24]}."
            ),
        }],
        "max_tokens": 129,
        "temperature": 0,
        "seed": int(randomness["randomness"][:16], 16) % 2147483647,
        "thinking_enabled": False,
    }
    if request != expected_request:
        raise ValueError(f"{pair}: request is not derived from authenticated randomness")
    arms = summary.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"off", "on"}:
        raise ValueError(f"{pair}: expected exact off/on arms")
    measured: dict[str, Any] = {}
    for name in ("off", "on"):
        arm = arms[name]
        _, local_arm = strict_json(pair / name / "arm.json")
        if (not isinstance(arm, dict) or set(arm) != ARM_KEYS or arm != local_arm or
                arm.get("schema_version") != 1 or arm.get("arm") != name or
                arm.get("direct_requested") != (name == "on")):
            raise ValueError(f"{pair}: malformed {name} arm")
        if (arm.get("safe_returncode") != 0 or
                arm.get("independent_completion_tokens") != 129 or
                arm.get("independent_warm_completion_tokens") != 129):
            raise ValueError(f"{pair}: {name} arm is short or unsafe")
        crash = Path(arm.get("crash_evidence", ""))
        crash_hashes = arm.get("crash_artifact_sha256")
        if (not isinstance(crash_hashes, dict) or
                set(crash_hashes) != {"main.log", "cmd.log", "samples.log", "kernel.log"} or
                not crash.is_dir()):
            raise ValueError(f"{pair}: {name} crash evidence is missing")
        measured[name] = measured_timing(crash / "cmd.log", crash_hashes.get("cmd.log", ""))
        crash_identity = stat_identity(crash)
        if crash_identity != arm.get("crash_evidence_identity"):
            raise ValueError(f"{pair}: {name} crash identity changed")
        measured[name]["crash_identity"] = crash_identity
        for log_name in ("main.log", "samples.log", "kernel.log"):
            if sha256(crash / log_name) != crash_hashes.get(log_name):
                raise ValueError(f"{pair}: {name} crash artifact changed: {log_name}")
        receipt = (pair / name / "containment.stdout").read_text(encoding="utf-8")
        if receipt.strip() != f"SAFE_RUN_DONE rc=0 killed=no dir={crash}":
            raise ValueError(f"{pair}: {name} safe-run receipt is invalid")
        main_log = (crash / "main.log").read_text(encoding="utf-8", errors="strict")
        cmd_log = (crash / "cmd.log").read_text(encoding="utf-8", errors="strict")
        kernel_log = (crash / "kernel.log").read_text(encoding="utf-8", errors="strict")
        identity_match = re.search(
            r"executed_candidate_verified pid=(\d+) start_ticks=(\d+) path=\S+ "
            r"executed_binary_sha256=([0-9a-f]{64}) device_inode=([0-9:]+)", main_log,
        )
        if (identity_match is None or identity_match.group(3) != frozen["binary_sha256"] or
                "SAFE_RUN end rc=0 killed=no" not in main_log or
                "no identity contradiction observed" not in main_log):
            raise ValueError(f"{pair}: {name} lifecycle attestation is invalid")
        measured[name]["executed_identity"] = identity_match.groups()
        environment_match = re.search(
            r"executed_environment_allowlist=[A-Z0-9_,]+ "
            r"executed_environment_sha256=([0-9a-f]{64})", main_log,
        )
        if environment_match is None or environment_match.group(1) != arm.get("environment_sha256"):
            raise ValueError(f"{pair}: {name} executed environment is unbound")
        warm_counts = [int(value) for value in
                       (pair / name / "warm.dispatch-counts").read_text().split()]
        measured_counts = [int(value) for value in
                           (pair / name / "measured.dispatch-counts").read_text().split()]
        if len(warm_counts) != 2 or len(measured_counts) != 2:
            raise ValueError(f"{pair}: {name} dispatch receipts are malformed")
        mapping_markers = cmd_log.count("direct expert-slot arena mapping enabled")
        admission_markers = cmd_log.count("direct expert-slot hit layer=")
        dispatch_markers = cmd_log.count("direct expert-slot dispatch layer=")
        if (arm.get("mapping_markers") != mapping_markers or
                arm.get("admission_markers") != admission_markers or
                arm.get("dispatch_markers") != dispatch_markers or
                arm.get("warm_dispatch_delta") != warm_counts[1] - warm_counts[0] or
                arm.get("measured_dispatch_delta") != measured_counts[1] - measured_counts[0]):
            raise ValueError(f"{pair}: {name} direct-path marker claims are inconsistent")
        fault_re = re.compile(
            r"FATAL|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|"
            r"Out of memory|NVRM.*Xid", re.I,
        )
        if arm.get("fault_markers") != len(fault_re.findall(cmd_log + "\n" + main_log + "\n" + kernel_log)):
            raise ValueError(f"{pair}: {name} fault-marker claim is inconsistent")
        if ("memory_swap_max=0" not in main_log or
                re.search(r"cgroup_final .*swap_current_bytes=0 .*oom_kill 0", main_log) is None):
            raise ValueError(f"{pair}: {name} cgroup safety receipt is invalid")
        sample_values = [
            int(match.group(1)) / 1_048_576
            for match in re.finditer(r"mem_avail_kb=(\d+)",
                                     (crash / "samples.log").read_text(encoding="utf-8"))
        ]
        if not sample_values or min(sample_values) < 10.0:
            raise ValueError(f"{pair}: {name} memory evidence crossed the safety floor")
        preflight_raw, preflight = strict_json(pair / name / "memory-preflight.json")
        if (preflight.get("pass") is not True or preflight.get("required_gib") != 110.0 or
                not isinstance(preflight.get("mem_available_gib"), (int, float)) or
                preflight["mem_available_gib"] < 110.0):
            raise ValueError(f"{pair}: {name} memory preflight did not pass")
        phase_records: dict[str, tuple[dict[str, Any], dict[str, Any], bytes]] = {}
        for phase in ("warm", "measured"):
            response_raw, response = strict_json(pair / name / f"{phase}.json")
            _, token_record = strict_json(pair / name / f"{phase}.tokens.json")
            if (set(token_record) != TOKEN_RECORD_KEYS or token_record.get("schema_version") != 1 or
                    token_record.get("label") != f"{name}-{phase}" or
                    token_record.get("response_sha256") != sha256_bytes(response_raw) or
                    token_record.get("reference_token_count") != 129 or
                    token_record.get("tokenizer_sha256") != frozen["artifacts"]["tokenizer_sha256"]):
                raise ValueError(f"{pair}: {name} {phase} response/token record is invalid")
            measured[name][f"{phase}_generated"] = generated(response)
            expected_timing_id = (measured[name]["warm_request_id"] if phase == "warm"
                                  else measured[name]["request_id"])
            if response.get("id") != expected_timing_id:
                raise ValueError(f"{pair}: {name} {phase} timing request is unbound")
            http_fields = (pair / name / f"{phase}.http").read_text().split()
            if (len(http_fields) != 2 or int(http_fields[0]) != 200 or
                    arm.get(f"{phase}_http_code") != 200 or
                    not math.isclose(float(http_fields[1]), arm.get(f"{phase}_wall_seconds"),
                                     rel_tol=0.0, abs_tol=5e-7)):
                raise ValueError(f"{pair}: {name} {phase} HTTP receipt is inconsistent")
            phase_records[phase] = (response, token_record, response_raw)
        canonical = json.dumps(
            measured[name]["measured_generated"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        if arm.get("generated_sha256") != sha256_bytes(canonical):
            raise ValueError(f"{pair}: {name} generated digest is self-reported")
        if arm.get("generated_bytes") != len(canonical):
            raise ValueError(f"{pair}: {name} generated byte count is inconsistent")
        expected_timing_receipt = {
            "warm": {
                "request_id": measured[name]["warm_request_id"],
                "token_indices": list(range(1, 130)),
                "token_timestamps_ns": measured[name]["warm_token_timestamps_ns"],
                "token_ids": measured[name]["warm_token_ids"],
                "response_sha256": sha256_bytes(phase_records["warm"][2]),
                "response_identity": phase_records["warm"][1]["response_identity"],
            },
            "measured": {
                "request_id": measured[name]["request_id"],
                "token_indices": list(range(1, 130)),
                "token_timestamps_ns": measured[name]["token_timestamps_ns"],
                "token_ids": measured[name]["token_ids"],
                "response_sha256": sha256_bytes(phase_records["measured"][2]),
                "response_identity": phase_records["measured"][1]["response_identity"],
            },
            "cmd_log_identity": measured[name]["cmd_log_identity"],
            "cmd_log_sha256": measured[name]["cmd_log_sha256"],
        }
        if arm.get("timing_receipt") != expected_timing_receipt:
            raise ValueError(f"{pair}: {name} response-bound timing receipt is invalid")
        if (arm.get("completion_tokens") != phase_records["measured"][0]["usage"]["completion_tokens"] or
                arm.get("warm_completion_tokens") != phase_records["warm"][0]["usage"]["completion_tokens"] or
                arm.get("measured_token_scorer") != phase_records["measured"][1] or
                arm.get("warm_token_scorer") != phase_records["warm"][1] or
                arm.get("measured_reasoning_bytes") != phase_records["measured"][1]["reasoning_bytes"] or
                arm.get("warm_reasoning_bytes") != phase_records["warm"][1]["reasoning_bytes"]):
            raise ValueError(f"{pair}: {name} response metrics are inconsistent")
    if (arms["off"].get("generated_sha256") != arms["on"].get("generated_sha256") or
            measured["off"]["measured_generated"] != measured["on"]["measured_generated"] or
            measured["off"]["warm_generated"] != measured["on"]["warm_generated"]):
        raise ValueError(f"{pair}: measured generated bytes differ across arms")
    recomputed_checks = {
        "same_frozen_binary": all(arm["binary_sha256"] == frozen["binary_sha256"]
                                  for arm in arms.values()),
        "same_model": all(arm["model_sha256"] == frozen["model_sha256"]
                          for arm in arms.values()),
        "same_request": all(arm["request_sha256"] == request_sha for arm in arms.values()),
        "safe_returncodes_zero": all(arm["safe_returncode"] == 0 for arm in arms.values()),
        "http_200": all(arm["warm_http_code"] == 200 and arm["measured_http_code"] == 200
                        for arm in arms.values()),
        "independent_exact_output_tokens": all(
            arm["independent_completion_tokens"] == 129 and
            arm["independent_warm_completion_tokens"] == 129 for arm in arms.values()
        ),
        "thinking_disabled_no_reasoning_channel": all(
            arm["measured_reasoning_bytes"] == 0 and arm["warm_reasoning_bytes"] == 0
            for arm in arms.values()
        ),
        "all_generated_outputs_nonempty": all(
            measured[name]["measured_generated"]["content"] and
            measured[name]["warm_generated"]["content"] for name in ("off", "on")
        ),
        "generated_output_byte_identical": (
            measured["off"]["measured_generated"] == measured["on"]["measured_generated"]
        ),
        "warm_generated_output_byte_identical": (
            measured["off"]["warm_generated"] == measured["on"]["warm_generated"]
        ),
        "off_path_not_mapped": arms["off"]["mapping_markers"] == 0,
        "off_path_has_no_direct_dispatches": arms["off"]["dispatch_markers"] == 0,
        "on_path_mapped": arms["on"]["mapping_markers"] >= 1,
        "on_path_dispatched_for_compared_warm_response": arms["on"]["warm_dispatch_delta"] >= 1,
        "on_path_dispatched_for_compared_measured_response": arms["on"]["measured_dispatch_delta"] >= 1,
        "clean_exit_attested": all(arm["clean_exit_attestation"] is True for arm in arms.values()),
        "no_fault_markers": all(arm["fault_markers"] == 0 for arm in arms.values()),
    }
    if recomputed_checks != checks or any(value is not True for value in recomputed_checks.values()):
        raise ValueError(f"{pair}: stored checks differ from recomputed runtime evidence")
    if (summary.get("environment_sha256") != {
            name: arms[name]["environment_sha256"] for name in ("off", "on")} or
            manifest.get("environment_sha256") != summary["environment_sha256"] or
            arms["off"]["environment_sha256"] == arms["on"]["environment_sha256"]):
        raise ValueError(f"{pair}: arm environments are not independently bound")
    return {
        "path": str(pair),
        "arm_order": expected_order,
        "request_sha256": request_sha,
        "randomness_round": round_number,
        "randomness": randomness["randomness"],
        "bindings": {key: summary[key] for key in (*bindings, "repository_head")},
        "arms": measured,
        "summary_sha256": sha256_bytes(summary_raw),
        "manifest_sha256": sha256(pair / "manifest.json"),
    }


def upper_ratio(candidate: list[float], baseline: list[float]) -> float:
    logs = [math.log(c / b) for c, b in zip(candidate, baseline, strict=True)]
    mean = statistics.fmean(logs)
    sem = statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean + T95_DF4 * sem)


def write_result(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    authority: dict[str, Any] | None = None,
    cli_token: object | None = None,
) -> dict[str, Any]:
    if __name__ != "__main__" or cli_token is not _CLI_AUTHORITY_TOKEN:
        raise ValueError("campaign writer requires validated CLI authority")
    assert authority is not None
    if set(authority) != {"freeze", "freeze_sha256", "freeze_commit", "harness_sha256"}:
        raise ValueError("campaign writer authority is incomplete")
    if len(rows) != 10:
        raise ValueError("W3 requires exactly ten fresh-server pairs")
    request_hashes = [row["request_sha256"] for row in rows]
    rounds = [row["randomness_round"] for row in rows]
    random_values = [row["randomness"] for row in rows]
    if len(set(request_hashes)) != 10 or len(set(rounds)) != 10 or len(set(random_values)) != 10:
        raise ValueError("fixtures and public-randomness records must be fresh and unique")
    pair_paths = [row["path"] for row in rows]
    crash_identities = [row["arms"][arm]["crash_identity"]
                        for row in rows for arm in ("off", "on")]
    cmd_hashes = [row["arms"][arm]["cmd_log_sha256"]
                  for row in rows for arm in ("off", "on")]
    executed = [row["arms"][arm]["executed_identity"]
                for row in rows for arm in ("off", "on")]
    for values, expected, label in (
        (pair_paths, 10, "pair paths"),
        (crash_identities, 20, "crash identities"),
        (cmd_hashes, 20, "timing logs"),
        (executed, 20, "executed process identities"),
    ):
        if len(set(map(str, values))) != expected:
            raise ValueError(f"campaign reuses {label}")
    frozen = rows[0]["bindings"]
    if any(row["bindings"] != frozen for row in rows[1:]):
        raise ValueError("campaign pairs do not share one frozen candidate")
    baseline: list[float] = []
    candidate: list[float] = []
    blocks: list[dict[str, Any]] = []
    for block_index in range(5):
        left, right = rows[2 * block_index:2 * block_index + 2]
        off = [left["arms"]["off"]["completed_seconds"],
               right["arms"]["off"]["completed_seconds"]]
        on = [left["arms"]["on"]["completed_seconds"],
              right["arms"]["on"]["completed_seconds"]]
        baseline.append(statistics.fmean(off))
        candidate.append(statistics.fmean(on))
        blocks.append({
            "block": block_index + 1,
            "schedule": "ABBA" if block_index % 2 == 0 else "BAAB",
            "baseline_seconds": baseline[-1],
            "candidate_seconds": candidate[-1],
            "pair_request_sha256": [left["request_sha256"], right["request_sha256"]],
        })
    bound = upper_ratio(candidate, baseline)
    log_ratios = [math.log(c / b) for c, b in zip(candidate, baseline, strict=True)]
    log_mean = statistics.fmean(log_ratios)
    log_stdev = statistics.stdev(log_ratios)
    log_sem = log_stdev / math.sqrt(5)
    log_upper = log_mean + T95_DF4 * log_sem
    passed = bound <= 0.95
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    raw_path = output / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "gate": "W3-completed-time-campaign",
        "status": "PASS" if passed else "FAIL",
        "acceptance_formula": (
            "five ABBA/BAAB block means; one-sided 95% upper bound of the "
            "geometric paired candidate/baseline completed-time ratio <= 0.95"
        ),
        "timing_formula": "seconds=(t129-t1)/1e9; decode_tps=128/seconds",
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "completed_time_ratio_upper_95": bound,
        "confidence_derivation": {
            "paired_log_ratios": log_ratios,
            "mean_log_ratio": log_mean,
            "sample_standard_deviation": log_stdev,
            "standard_error": log_sem,
            "degrees_of_freedom": 4,
            "one_sided_t_critical_95": T95_DF4,
            "upper_log_bound": log_upper,
            "upper_ratio_bound": math.exp(log_upper),
        },
        "blocks": blocks,
        "frozen_bindings": frozen,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scorer = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "scorer_sha256": sha256(scorer),
        "freeze_sha256": authority["freeze_sha256"],
        "freeze_commit": authority["freeze_commit"],
        "harness_sha256": authority["harness_sha256"],
        "binary_sha256": authority["freeze"]["binary_sha256"],
        "model_sha256": authority["freeze"]["model_sha256"],
        "raw_sha256": sha256(raw_path),
        "summary_sha256": sha256(summary_path),
        "input_manifest_sha256": [row["manifest_sha256"] for row in rows],
        "input_summary_sha256": [row["summary_sha256"] for row in rows],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ValueError("scorer requires isolated Python with bytecode disabled")
    if os.environ != ALLOWED_ENVIRONMENT:
        raise ValueError("scorer environment differs from the exact allowlist")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("pairs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.pairs) != 10:
        raise ValueError("expected exactly ten chronological pair directories")
    repo = args.repo.resolve()
    authority = committed_authority(repo, args.freeze.resolve())
    rows = [score_pair(path.resolve(), order, authority)
            for path, order in zip(args.pairs, EXPECTED_ORDERS, strict=True)]
    summary = write_result(
        args.output.resolve(), rows, authority=authority,
        cli_token=_CLI_AUTHORITY_TOKEN,
    )
    print(json.dumps({"output": str(args.output.resolve()), "status": summary["status"],
                      "completed_time_ratio_upper_95": summary["completed_time_ratio_upper_95"]},
                     sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"W3 campaign scoring failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
