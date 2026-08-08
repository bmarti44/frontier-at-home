#!/usr/bin/env python3
"""Fail-closed scorer for W4's matched serving-prefill confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import subprocess
from typing import Any


T_ONE_SIDED_95_DF4 = 2.131846786326649
MIN_PROMPT_TOKENS = 16_000
MIN_MEMORY_KIB = 10 * 1024 * 1024
TOKENIZER = Path("/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json")
TOKENIZER_RUNTIME = Path("/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-0.22.2")
RENDER_ORACLE = Path("/home/bmarti44/.cache/glm52-w7-render-oracle-c8/oracle")
DRAND_NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
DRAND_RUNTIME = Path("/home/bmarti44/.cache/glm52-drand-client-1.4.2")
SYSTEM_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
STAGED_BINARY = Path("/home/bmarti44/.cache/glm52-w4-topk-c5-serving/ds4-server")
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_INIT_SHA256 = "eff4eff4386074cbbd5e34e009bdfccf5879a7e5c5f0da6f4b6babc0597c09e4"
TOKENIZER_NATIVE_SHA256 = "fa049ce975669d8a90fb48960f412e626fa54cf596c2f75d6820949f4888e910"
RENDER_ORACLE_SHA256 = "6bd6896581db71bdb76a9afdb59a9254b151ade22017e17f111fd3345fb5ad66"
BINARY_SHA256 = "620fd8fa2b6cd0885f11c70cebfecf0ca128580a5dd2e27f05822d4ff4b4651f"
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
FIXTURE_SHA256 = "2d31aeb3156ae01ab7213cdf50eb7660df8e869de12be7646a6b19aaf3405031"
MICROGATE_SHA256 = "9aaf51b0722ec2573876d6a35ce733e6e574bb1349daf6f72f61100995c39bde"
BASE_SHA256 = "e2f6235cd5f94b67773e75cff0f4fbceaa264f5b88e3d12b45ae3bb1e31e6924"
CGROUP_SHA256 = "d604c4e64f102ce03a7d6660b887e5b6c78091eeea72eab82874f34f9f4efb14"
SAFE_SHA256 = "2ddffb19f79b790c419db8ac53574d23ccf9f2c7699136fbaa55fc2a890b19e6"
MEMORY_GUARD_SHA256 = "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58"
DRAND_VERIFIER_SHA256 = "c191d301e1ff8460fffaea9dfeaab7d0fce0d63f92d3fdfcfa20442ccfdc2131"
DRAND_NODE_SHA256 = "3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90"
DRAND_RUNTIME_TREE_SHA256 = "38161b0df115fbd3c2e1dda87759a1eb1e87500109661f0f2fa18874d6e1a0e4"
SYSTEM_CA_BUNDLE_SHA256 = "6602a85a36afc2e51c66a0df5ae3d383c5b7c2fed93339ccef7d37e01faf09e8"
ENGINE_SOURCE_COMMIT = "0424a6b406e4f6e125be3269104f3d16ad39c951"
REQUEST_SHA256 = "e5aa55b32992e3033a90b0c5be77c7346d88202ca3f209d2036d05f5f64cfd46"
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
EXPECTED_CONFIGURATION = {
    "context": 32768, "fixture_fraction": 0.46, "cache_gib": 40,
    "fetch_threads": 6, "stable_model_remap": 1, "max_tokens": 0,
    "temperature": 0, "boundary_align": 4, "boundary_trim": 8,
    "on_flag": "DS4_CUDA_TOPK2048_CUB=1", "off_flag": "unset",
    "containment": {"kill_floor_gib": 24, "minimum_start_gib": 110,
                    "timeout_s": 3600, "memory_swap_max": 0},
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FORMULA = (
    "five fresh-server ABBA/BAAB blocks; same novel >=16000-token request; "
    "external prefill time=response_complete_ns-request_start_ns with max_tokens=0 "
    "and cache_write_tokens=prompt_tokens; block speedup=mean(OFF seconds)/mean(ON "
    "seconds); one-sided 95% lower log-ratio t bound with df=4 >=1.05; exact "
    "semantic response and final logits; independently replayed W4 CUDA top-k "
    "lower-95 speedup >=2.0 with exact selected IDs"
)


class InvalidCampaign(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidCampaign(message)


def _integer(value: object) -> bool:
    return type(value) is int


def _sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _read_stable(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"artifact is not regular: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 f"artifact changed while read: {path}")
        payload = b"".join(chunks)
        _require(len(payload) == before.st_size, f"short artifact read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _read_stable_identity(path: Path) -> tuple[bytes, tuple[int, int, int, int]]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"artifact is not regular: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        _require(identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 f"artifact changed while snapshotted: {path}")
        payload = b"".join(chunks)
        _require(len(payload) == before.st_size, f"short artifact read: {path}")
        return payload, identity
    finally:
        os.close(descriptor)


def _snapshot_files(root: Path, names: set[str]) -> dict[str, tuple[bytes, tuple[int, int, int, int]]]:
    snapshot = {}
    for relative in sorted(names):
        path = root / relative
        _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                 "invalid artifact path")
        snapshot[relative] = _read_stable_identity(path)
    return snapshot


def _verify_snapshot_unchanged(
    root: Path,
    snapshot: dict[str, tuple[bytes, tuple[int, int, int, int]]],
    names: set[str],
) -> None:
    discovered = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.name != "manifest.json" and (path.is_file() or path.is_symlink())
    }
    _require(discovered == names, "artifact membership changed during scoring")
    for relative, (payload, identity) in snapshot.items():
        current, current_identity = _read_stable_identity(root / relative)
        _require(current_identity == identity and current == payload,
                 f"artifact changed after snapshot: {relative}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_stable(path)).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    _require(root.is_dir() and not root.is_symlink(), "invalid drand runtime tree")
    paths = sorted(root.rglob("*"))
    files = []
    for path in paths:
        _require(not path.is_symlink(), "invalid drand runtime tree")
        if path.is_file():
            files.append(path)
        else:
            _require(path.is_dir(), "invalid drand runtime tree")
    _require(bool(files), "invalid drand runtime tree")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        data = _read_stable(path)
        digest.update(len(relative).to_bytes(4, "big")); digest.update(relative)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def independent_tokenization(request_path: Path,
                             request_payload: bytes | None = None) -> dict[str, object]:
    dependencies = (
        (TOKENIZER, TOKENIZER_SHA256),
        (TOKENIZER_RUNTIME / "tokenizers/__init__.py", TOKENIZER_INIT_SHA256),
        (TOKENIZER_RUNTIME / "tokenizers/tokenizers.abi3.so", TOKENIZER_NATIVE_SHA256),
        (RENDER_ORACLE, RENDER_ORACLE_SHA256),
    )
    for path, digest in dependencies:
        _require(path.is_file() and not path.is_symlink() and _sha256_file(path) == digest,
                 f"tokenization dependency mismatch: {path}")
    request = _read_stable(request_path) if request_payload is None else request_payload
    first = subprocess.run([str(RENDER_ORACLE)], input=request, capture_output=True,
                           check=False, timeout=30)
    second = subprocess.run([str(RENDER_ORACLE)], input=request, capture_output=True,
                            check=False, timeout=30)
    _require(first.returncode == second.returncode == 0 and not first.stderr
             and not second.stderr and first.stdout == second.stdout,
             "render oracle failed or was nondeterministic")
    code = (
        "import hashlib,json,struct,sys;from tokenizers import Tokenizer;"
        "p=sys.stdin.buffer.read().decode('utf-8');"
        f"t=Tokenizer.from_file({str(TOKENIZER)!r}).encode(p,add_special_tokens=False).ids;"
        "print(json.dumps({'prompt_tokens':len(t),'token_ids_sha256':"
        "hashlib.sha256(struct.pack(f'<{len(t)}i',*t)).hexdigest()},sort_keys=True))"
    )
    env = {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
           "LC_ALL": "C.UTF-8", "PYTHONPATH": str(TOKENIZER_RUNTIME)}
    tokenized = subprocess.run(["/usr/bin/python3", "-c", code], input=first.stdout,
                               capture_output=True, check=False, timeout=60, env=env)
    _require(tokenized.returncode == 0 and not tokenized.stderr,
             "independent tokenizer failed")
    try:
        result = json.loads(tokenized.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidCampaign("independent tokenizer emitted invalid JSON") from error
    _require(isinstance(result, dict) and set(result) == {"prompt_tokens", "token_ids_sha256"}
             and _integer(result["prompt_tokens"]) and result["prompt_tokens"] >= MIN_PROMPT_TOKENS
             and _sha256(result["token_ids_sha256"]), "invalid independent tokenization")
    return {**result, "rendered_prompt_sha256": hashlib.sha256(first.stdout).hexdigest(),
            "rendered_prompt_bytes": len(first.stdout)}


def _ratio_lower95(ratios: list[float]) -> float:
    _require(len(ratios) == 5, "confidence bound requires five blocks")
    _require(all(math.isfinite(value) and value > 0 for value in ratios),
             "invalid ratio")
    logs = [math.log(value) for value in ratios]
    return math.exp(
        statistics.fmean(logs)
        - T_ONE_SIDED_95_DF4 * statistics.stdev(logs) / math.sqrt(5)
    )


def _validated_row(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "row is not an object")
    row = value
    required = {
        "block", "position", "arm", "run_id", "binary_sha256",
        "model_sha256", "common_config_sha256", "request_sha256",
        "topk_cub", "request_start_ns", "response_complete_ns",
        "prompt_tokens", "cached_tokens", "cache_write_tokens",
        "response_semantic_sha256", "final_logits_sha256",
        "logit_sequence_sha256", "executed_environment_sha256",
        "topk_marker_count", "server_fresh", "safety",
    }
    _require(set(row) == required, "row keys differ")
    _require(_integer(row["block"]) and 0 <= row["block"] < 5, "invalid block")
    _require(_integer(row["position"]) and 0 <= row["position"] < 4,
             "invalid position")
    _require(row["arm"] in {"off", "on"}, "invalid arm")
    _require(isinstance(row["run_id"], str) and bool(row["run_id"]),
             "invalid run id")
    for name in (
        "binary_sha256", "model_sha256", "common_config_sha256",
        "request_sha256", "response_semantic_sha256", "final_logits_sha256",
        "logit_sequence_sha256",
        "executed_environment_sha256",
    ):
        _require(_sha256(row[name]), f"invalid {name}")
    expected_flag = 1 if row["arm"] == "on" else 0
    _require(_integer(row["topk_cub"]) and row["topk_cub"] == expected_flag,
             "flag/arm mismatch")
    _require(_integer(row["topk_marker_count"]) and
             row["topk_marker_count"] == expected_flag,
             "effective marker mismatch")
    for name in (
        "request_start_ns", "response_complete_ns", "prompt_tokens",
        "cached_tokens", "cache_write_tokens",
    ):
        _require(_integer(row[name]) and row[name] >= 0, f"invalid {name}")
    _require(row["response_complete_ns"] > row["request_start_ns"],
             "nonpositive external prefill time")
    _require(row["prompt_tokens"] >= MIN_PROMPT_TOKENS, "prompt too short")
    _require(row["cached_tokens"] == 0, "request was not novel")
    _require(row["cache_write_tokens"] == row["prompt_tokens"],
             "evaluated-token accounting differs")
    _require(row["server_fresh"] is True, "server was not fresh")

    safety = row["safety"]
    safety_keys = {
        "containment_rc", "minimum_mem_available_kib", "swap_growth_bytes",
        "cgroup_max_delta", "cgroup_oom_delta", "cgroup_oom_kill_delta",
        "xid_count", "surviving_descendants",
    }
    _require(isinstance(safety, dict) and set(safety) == safety_keys,
             "invalid safety schema")
    _require(all(_integer(safety[name]) for name in safety_keys),
             "non-integer safety value")
    for name in safety_keys - {"minimum_mem_available_kib"}:
        _require(safety[name] == 0, f"unsafe {name}")
    _require(safety["minimum_mem_available_kib"] >= MIN_MEMORY_KIB,
             "memory floor violated")
    return row


def _validate_microgate(value: object) -> dict[str, Any]:
    _require(isinstance(value, dict), "microgate is not an object")
    required = {
        "block_a_ms", "block_b_ms", "selected_ids_sha256",
        "speedup_lower_95", "required_speedup_lower_95", "verdict",
    }
    _require(required.issubset(value), "microgate fields missing")
    a = value["block_a_ms"]
    b = value["block_b_ms"]
    _require(isinstance(a, list) and isinstance(b, list) and
             len(a) == len(b) == 5, "microgate samples differ")
    _require(all(type(item) in {int, float} and math.isfinite(float(item)) and
                 float(item) > 0 for item in a + b), "invalid microgate timing")
    recomputed = _ratio_lower95([float(left) / float(right)
                                 for left, right in zip(a, b)])
    _require(type(value["speedup_lower_95"]) in {int, float} and
             math.isclose(float(value["speedup_lower_95"]), recomputed,
                          rel_tol=1e-12, abs_tol=1e-12),
             "microgate speedup does not replay")
    _require(_sha256(value["selected_ids_sha256"]), "invalid selected IDs digest")
    _require(value["required_speedup_lower_95"] == 2.0 and
             value["verdict"] == "PASS", "microgate did not pass")
    return {"topk_speedup_lower_95": recomputed,
            "selected_ids_sha256": value["selected_ids_sha256"]}


def _fail(message: str) -> dict[str, object]:
    return {"schema": "glm52-w4-serving-summary-v1", "formula": FORMULA,
            "checks": {"input_and_acceptance_checks": False},
            "observed": {}, "failure": message, "verdict": "FAIL"}


def score_campaign_rows(rows: object, schedules: object,
                        microgate: object) -> dict[str, object]:
    try:
        _require(isinstance(schedules, list) and len(schedules) == 5,
                 "need five schedules")
        _require(all(item in {"ABBA", "BAAB"} for item in schedules),
                 "invalid schedule")
        _require(isinstance(rows, list) and len(rows) == 20,
                 "need exactly twenty rows")
        validated = [_validated_row(row) for row in rows]
        micro = _validate_microgate(microgate)

        slots: dict[tuple[int, int], dict[str, Any]] = {}
        for row in validated:
            slot = (row["block"], row["position"])
            _require(slot not in slots, "duplicate block position")
            slots[slot] = row
        for block, schedule in enumerate(schedules):
            for position, letter in enumerate(schedule):
                expected = "off" if letter == "A" else "on"
                _require(slots.get((block, position), {}).get("arm") == expected,
                         "arm schedule mismatch")
        _require(len({row["run_id"] for row in validated}) == 20,
                 "duplicate run id")
        for name in (
            "binary_sha256", "model_sha256", "common_config_sha256",
            "request_sha256", "prompt_tokens", "response_semantic_sha256",
            "final_logits_sha256", "logit_sequence_sha256",
        ):
            _require(len({row[name] for row in validated}) == 1,
                     f"unequal {name}")

        seconds = {
            row["run_id"]:
            (row["response_complete_ns"] - row["request_start_ns"]) / 1e9
            for row in validated
        }
        block_ratios: list[float] = []
        off_seconds: list[float] = []
        on_seconds: list[float] = []
        for block in range(5):
            block_rows = [slots[(block, position)] for position in range(4)]
            off = [seconds[row["run_id"]] for row in block_rows
                   if row["arm"] == "off"]
            on = [seconds[row["run_id"]] for row in block_rows
                  if row["arm"] == "on"]
            _require(len(off) == len(on) == 2, "unbalanced block")
            off_mean = statistics.fmean(off)
            on_mean = statistics.fmean(on)
            off_seconds.append(off_mean)
            on_seconds.append(on_mean)
            block_ratios.append(off_mean / on_mean)
        prefill_lower95 = _ratio_lower95(block_ratios)
        prompt_tokens = validated[0]["prompt_tokens"]
        checks = {
            "ids_identical": True,
            "logits_identical": True,
            "semantic_response_identical": True,
            "topk_speedup": micro["topk_speedup_lower_95"] >= 2.0,
            "prefill_speedup": prefill_lower95 >= 1.05,
            "all_safety_checks": True,
        }
        return {
            "schema": "glm52-w4-serving-summary-v1",
            "formula": FORMULA,
            "checks": checks,
            "observed": {
                "prompt_tokens": prompt_tokens,
                "off_block_seconds": off_seconds,
                "on_block_seconds": on_seconds,
                "prefill_block_speedup_ratios": block_ratios,
                "prefill_speedup_lower_95": prefill_lower95,
                "off_prefill_tokens_per_second": [prompt_tokens / value
                                                     for value in off_seconds],
                "on_prefill_tokens_per_second": [prompt_tokens / value
                                                    for value in on_seconds],
                **micro,
            },
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
    except (InvalidCampaign, KeyError, TypeError, ValueError, OverflowError) as error:
        return _fail(str(error))


def _strict_json(path: Path) -> object:
    return json.loads(_read_stable(path), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")))


def _git_bytes(root: Path, candidate: str, relative: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(root), "show",
         f"{candidate}:{relative}"], check=True, capture_output=True,
    ).stdout


def score_run_dir(run_dir: Path) -> dict[str, object]:
    try:
        manifest_payload, manifest_identity = _read_stable_identity(run_dir / "manifest.json")
        manifest = json.loads(
            manifest_payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")))
        _require(isinstance(manifest, dict), "manifest is not an object")
        required_manifest = {
            "schema", "candidate_hash", "runner_sha256", "scorer_sha256",
            "base_runner_sha256", "cgroup_sha256", "safe_run_sha256",
            "memory_guard_sha256", "binary_sha256", "engine_source_commit",
            "model_sha256", "model_bytes", "fixture_sha256",
            "executed_request_sha256", "microgate_sha256", "configuration",
            "configuration_sha256", "tokenization", "public_randomness_sha256",
            "public_randomness_receipt_sha256", "publication_receipt", "schedules", "completed_rows",
            "drand_runtime_tree_sha256", "system_ca_bundle_sha256",
            "artifacts", "verdict",
        }
        _require(set(manifest) == required_manifest and
                 manifest["schema"] == "glm52-w4-serving-campaign-v1",
                 "manifest schema differs")
        _require(manifest["verdict"] == "PASS", "manifest is not a PASS candidate")
        candidate = manifest["candidate_hash"]
        _require(isinstance(candidate, str) and COMMIT_RE.fullmatch(candidate) is not None,
                 "invalid candidate hash")
        root = Path(__file__).resolve().parents[1]
        executing_head = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        executing_dirty = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True).stdout
        _require(executing_head == candidate and not executing_dirty,
                 "scorer is not executing from the clean reviewed candidate")
        runner_path = root / "scripts/102_run_w4_serving_campaign.py"
        scorer_path = root / "scripts/101_score_w4_serving_campaign.py"
        repo_dependencies = {
            "scripts/91_run_w7_cache_generation_campaign.py": BASE_SHA256,
            "results/glm52-gates/harness/glm_cgroup_run.sh": CGROUP_SHA256,
            "results/glm52-gates/harness/glm_safe_run.sh": SAFE_SHA256,
            "scripts/03_memory_guard.py": MEMORY_GUARD_SHA256,
            "scripts/89_verify_drand_receipt.mjs": DRAND_VERIFIER_SHA256,
        }
        for relative, digest in repo_dependencies.items():
            dependency = root / relative
            payload = _read_stable(dependency)
            _require(hashlib.sha256(payload).hexdigest() == digest and
                     _git_bytes(root, candidate, relative) == payload,
                     f"fixed dependency differs: {dependency}")
        _require(_sha256_file(DRAND_NODE) == DRAND_NODE_SHA256,
                 "drand runtime differs")
        _require(_tree_sha256(DRAND_RUNTIME) == DRAND_RUNTIME_TREE_SHA256,
                 "drand imported runtime tree differs")
        _require(_sha256_file(SYSTEM_CA_BUNDLE) == SYSTEM_CA_BUNDLE_SHA256,
                 "system CA bundle differs")
        _require(_sha256_file(STAGED_BINARY) == BINARY_SHA256,
                 "executed binary artifact differs")
        fixture_path = root / "fixtures/ctx-32k.txt"
        _require(_sha256_file(fixture_path) == FIXTURE_SHA256 and
                 _git_bytes(root, candidate, "fixtures/ctx-32k.txt") == _read_stable(fixture_path),
                 "fixed fixture differs")
        runner_bytes = _read_stable(runner_path)
        scorer_bytes = _read_stable(scorer_path)
        _require(_git_bytes(root, candidate, "scripts/102_run_w4_serving_campaign.py") == runner_bytes,
                 "runner is not the candidate's tracked version")
        _require(_git_bytes(root, candidate, "scripts/101_score_w4_serving_campaign.py") == scorer_bytes,
                 "scorer is not the candidate's tracked version")
        expected_bindings = {
            "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
            "scorer_sha256": hashlib.sha256(scorer_bytes).hexdigest(),
            "base_runner_sha256": BASE_SHA256, "cgroup_sha256": CGROUP_SHA256,
            "safe_run_sha256": SAFE_SHA256, "memory_guard_sha256": MEMORY_GUARD_SHA256,
            "binary_sha256": BINARY_SHA256, "engine_source_commit": ENGINE_SOURCE_COMMIT,
            "model_sha256": MODEL_SHA256, "fixture_sha256": FIXTURE_SHA256,
            "executed_request_sha256": REQUEST_SHA256, "microgate_sha256": MICROGATE_SHA256,
            "drand_runtime_tree_sha256": DRAND_RUNTIME_TREE_SHA256,
            "system_ca_bundle_sha256": SYSTEM_CA_BUNDLE_SHA256,
        }
        _require(all(manifest.get(name) == value for name, value in expected_bindings.items()),
                 "fixed manifest binding differs")
        expected_config_sha = hashlib.sha256(json.dumps(
            EXPECTED_CONFIGURATION, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        _require(manifest["configuration"] == EXPECTED_CONFIGURATION and
                 manifest["configuration_sha256"] == expected_config_sha,
                 "fixed configuration differs")
        _require(manifest["model_bytes"] == 211075856448,
                 "model size binding differs")
        artifacts = manifest["artifacts"]
        _require(isinstance(artifacts, dict) and artifacts, "artifact map missing")
        discovered = [path for path in run_dir.rglob("*") if path.name != "manifest.json"
                      and (path.is_file() or path.is_symlink())]
        _require(all(not path.is_symlink() for path in discovered),
                 "artifact closure contains a symlink")
        actual_files = {str(path.relative_to(run_dir)) for path in discovered if path.is_file()}
        _require(set(artifacts) == actual_files, "artifact closure differs")
        snapshot = _snapshot_files(run_dir, actual_files)
        for relative, digest in artifacts.items():
            _require(_sha256(digest) and
                     hashlib.sha256(snapshot[relative][0]).hexdigest() == digest,
                     f"artifact digest differs: {relative}")
        def snapshot_json(relative: str) -> object:
            return json.loads(
                snapshot[relative][0],
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")))
        schedules = snapshot_json("schedules.json")
        microgate = snapshot_json("microgate-summary.json")
        tokenization = snapshot_json("request-tokenization.json")
        _require(snapshot_json("configuration.json") == EXPECTED_CONFIGURATION,
                 "configuration artifact differs")
        model_identity = snapshot_json("model-identity.json")
        _require(isinstance(model_identity, dict) and
                 set(model_identity) == {"sha256", "bytes", "device", "inode",
                                         "mtime_ns", "descriptor_held_for_all_arms"} and
                 model_identity["sha256"] == MODEL_SHA256 and
                 model_identity["bytes"] == manifest["model_bytes"] and
                 model_identity["descriptor_held_for_all_arms"] is True and
                 all(_integer(model_identity[name]) and model_identity[name] >= 0
                     for name in ("device", "inode", "mtime_ns")),
                 "captured model descriptor identity differs")
        request_payload = snapshot["request.json"][0]
        _require(hashlib.sha256(request_payload).hexdigest() == REQUEST_SHA256,
                 "request artifact differs from the fixed request")
        recomputed_tokenization = independent_tokenization(
            run_dir / "request.json", request_payload)
        _require(tokenization == recomputed_tokenization == manifest["tokenization"],
                 "independent tokenization differs")
        _require(schedules == manifest["schedules"], "schedule binding differs")
        frozen_microgate_path = root / "results/glm52-gates/W4-topk-candidate5-pass/summary.json"
        frozen_microgate_relative = "results/glm52-gates/W4-topk-candidate5-pass/summary.json"
        _require(_sha256_file(frozen_microgate_path) == MICROGATE_SHA256 and
                 _git_bytes(root, candidate, frozen_microgate_relative) ==
                 _read_stable(frozen_microgate_path),
                 "frozen microgate artifact differs")
        frozen_microgate = _strict_json(frozen_microgate_path)
        selected_names = (
            "block_a_ms", "block_b_ms", "selected_ids_sha256", "speedup_lower_95",
            "required_speedup_lower_95", "verdict")
        _require(microgate == {name: frozen_microgate[name] for name in selected_names},
                 "microgate selection differs from frozen evidence")
        raw_rows = [_strict_json_line(line) for line in
                    snapshot["raw.jsonl"][0].decode("utf-8").splitlines() if line]
        _require(manifest["completed_rows"] == len(raw_rows) == 20,
                 "completed-row count differs")

        spec = importlib.util.spec_from_file_location("w4_frozen_runner", runner_path)
        _require(spec is not None and spec.loader is not None, "cannot load frozen runner")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        publication = snapshot_json("publication-receipt.json")
        _require(publication == manifest["publication_receipt"] ==
                 runner.fetch_publication_receipt(candidate),
                 "GitHub publication receipt differs")
        seed, receipt_sha, _ = runner.verify_randomness_bytes(
            snapshot["randomness-receipt.json"][0], candidate, publication)
        _require(manifest["public_randomness_sha256"] == seed and
                 manifest["public_randomness_receipt_sha256"] == receipt_sha and
                 schedules == runner.derive_schedules(seed),
                 "public randomness or derived schedule differs")
        replayed = []
        for raw in raw_rows:
            _require(isinstance(raw, dict), "raw row is not an object")
            run_id = raw.get("run_id")
            _require(isinstance(run_id, str) and re.fullmatch(
                r"b[0-4]-p[0-3]-(off|on)-[0-9a-f]{12}", run_id) is not None,
                "run id is malformed")
            out = run_dir / run_id
            prefix = run_id + "/"
            arm_snapshot = {
                relative[len(prefix):]: value for relative, value in snapshot.items()
                if relative.startswith(prefix)
            }
            containment_rc = int(arm_snapshot["containment.rc"][0].decode("utf-8").strip())
            containment_stdout = arm_snapshot["containment.stdout"][0].decode("utf-8")
            row = runner.parse_arm(
                raw["arm"], raw["block"], raw["position"], out, containment_rc,
                containment_stdout, REQUEST_SHA256, manifest["configuration_sha256"],
                recomputed_tokenization["prompt_tokens"], None, True, arm_snapshot)
            replayed.append(row)
        _require(replayed == raw_rows, "raw rows do not replay from bound arm artifacts")
        result = score_campaign_rows(replayed, schedules, microgate)
        recorded_summary = snapshot["summary.json"][0]
        expected_summary = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
        _require(recorded_summary == expected_summary, "recorded summary does not replay")
        _verify_snapshot_unchanged(run_dir, snapshot, actual_files)
        final_manifest, final_manifest_identity = _read_stable_identity(run_dir / "manifest.json")
        _require(final_manifest_identity == manifest_identity and
                 final_manifest == manifest_payload,
                 "manifest changed during scoring")
        return result
    except (InvalidCampaign, OSError, UnicodeError, json.JSONDecodeError,
            subprocess.SubprocessError, KeyError, TypeError, ValueError, OverflowError,
            RuntimeError) as error:
        return _fail(str(error))


def _strict_json_line(line: str) -> object:
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        result = score_run_dir(args.run_dir)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result = _fail(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
