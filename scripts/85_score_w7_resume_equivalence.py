#!/usr/bin/env python3
"""Fixed W7 restored-frontier equivalence scorer."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


N_VOCAB = 154880
LOGIT_BYTES = N_VOCAB * 4
PRIMARY_SHA256 = "a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3"
LIVE_SHA256 = "d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a"

EXPECTED_DUMPS = {
    "strict": {
        "logits.sync1.start0.prompt5055.suffix5055",
        "logits.sync2.start0.prompt5066.suffix5066",
    },
    "candidate": {
        "logits.sync1.start0.prompt5055.suffix5055",
        "logits.sync2.start5044.prompt5066.suffix22",
    },
    "cold": {"logits.sync1.start0.prompt5066.suffix5066"},
}
REQUIRED_BINDINGS = {
    "harness_sha256", "binary_sha256", "model_sha256", "scorer_sha256",
    "trace_scorer_sha256", "fixture_sha256", "tokenizer_sha256",
    "tokenizer_init_sha256", "tokenizer_native_sha256", "cgroup_sha256",
    "safe_sha256", "memory_guard_sha256", "engine_freeze_sha256",
    "configuration_sha256", "seed_sha256", "live_request_sha256",
    "primary_request_sha256",
}


def _pairs(items: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _json(path: Path) -> dict:
    value = json.loads(
        path.read_bytes(), object_pairs_hook=_pairs, parse_constant=_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _logits(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) != LOGIT_BYTES:
        raise ValueError(f"{path}: expected {LOGIT_BYTES} bytes, got {len(raw)}")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _inventory(path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(
            r"([0-9a-f]{64})  ([0-9a-f]{64})  ([0-9a-f]{40}\.kv)", line
        )
        if match is None or match.group(3) in result:
            raise ValueError(f"{path}: malformed or duplicate inventory row")
        result[match.group(3)] = (match.group(1), match.group(2))
    if not result:
        raise ValueError(f"{path}: empty inventory")
    return result


def _normalized_kv_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header = bytearray(handle.read(64))
        if len(header) != 64:
            raise ValueError("short KV header")
        # hits and access timestamps are mutable bookkeeping; all other header
        # bytes plus the complete payload define the checkpoint identity.
        header[12:16] = b"\0" * 4
        header[24:40] = b"\0" * 16
        digest.update(header)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_pass(path: Path) -> bool:
    try:
        value = _json(path)
        checks = value.get("checks")
        observed = value.get("observed")
        expected_checks = {
            "trace_exactly_two_requests",
            "trace_request_ids_exact",
            "trace_request_bytes_exact",
            "trace_rendered_bytes_exact",
            "trace_token_vectors_exact",
        }
        observation_keys = {
            "request_sha256", "rendered_sha256", "token_count", "token_ids_sha256"
        }
        return (
            set(value) == {"schema_version", "checks", "observed", "error", "verdict"}
            and value.get("schema_version") == 1
            and value.get("verdict") == "PASS"
            and value.get("error") is None
            and isinstance(checks, dict)
            and set(checks) == expected_checks
            and all(item is True for item in checks.values())
            and isinstance(observed, list)
            and len(observed) == 2
            and all(
                isinstance(item, dict)
                and set(item) == observation_keys
                and isinstance(item["token_count"], int)
                and not isinstance(item["token_count"], bool)
                and item["token_count"] > 0
                and all(
                    isinstance(item[key], str)
                    and re.fullmatch(r"[0-9a-f]{64}", item[key]) is not None
                    for key in ("request_sha256", "rendered_sha256", "token_ids_sha256")
                )
                for item in observed
            )
            and path.with_name("trace-scorer.rc").read_text().strip() == "0"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _selected_kv_identity(
    arm: Path, log: str
) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    try:
        matches = re.findall(
            r"kv cache hit text tokens=5044[^\n]* file=([^\s]+\.kv)", log
        )
        if len(matches) != 1:
            return None
        selected = Path(matches[0])
        if selected.parent != arm / "kv":
            return None
        before = _inventory(arm / "kv-before.sha256")
        after = _inventory(arm / "kv-after.sha256")
        if not (
            selected.name in before
            and selected.name in after
            and before[selected.name][1] == after[selected.name][1]
            and _sha256(selected) == after[selected.name][0]
            and _normalized_kv_sha256(selected) == after[selected.name][1]
        ):
            return None
        normalized_pre = tuple(
            (name, digests[1]) for name, digests in sorted(before.items())
        )
        return selected.name, before[selected.name][1], normalized_pre
    except (OSError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safety_pass(
    arm: Path, expected_binary_sha256: str, expected_binary_path: str,
    expected_binary_device_inode: str,
) -> bool:
    try:
        if (arm / "containment.rc").read_text().strip() != "0":
            return False
        stdout = (arm / "containment.stdout").read_text(encoding="utf-8")
        if len(re.findall(r"^SAFE_RUN_DONE rc=0 killed=no dir=\S+$", stdout, re.M)) != 1:
            return False
        main = (arm / "safety/main.log").read_text(encoding="utf-8")
        kernel = (arm / "safety/kernel.log").read_text(encoding="utf-8")
        samples = (arm / "safety/samples.log").read_text(encoding="utf-8")
        identities = re.findall(
            r"^\S+ executed_candidate_verified pid=([0-9]+) start_ticks=([0-9]+) "
            r"path=(/\S+) executed_binary_sha256=([0-9a-f]{64}) "
            r"device_inode=([0-9]+:[0-9]+)$",
            main,
            re.M,
        )
        identity_ok = (
            len(identities) == 1
            and int(identities[0][0]) > 0
            and int(identities[0][1]) > 0
            and identities[0][2] == expected_binary_path
            and identities[0][3] == expected_binary_sha256
            and identities[0][4] == expected_binary_device_inode
        )
        required = (
            identity_ok,
            re.search(
                r"cgroup_final .*swap_current_bytes=0 "
                r"events=low 0,high 0,max 0,oom 0,oom_kill 0,oom_group_kill 0,",
                main,
            ) is not None,
            "wrapper and descendant checks clean" in main,
            "SAFE_RUN end rc=0 killed=no" in main,
            kernel == "-- No entries --\n",
        )
        if not all(required):
            return False
        rows = [line for line in samples.splitlines() if line]
        if not rows:
            return False
        minimum = None
        for row in rows:
            match = re.fullmatch(
                r"\S+ mem_avail_kb=([0-9]+) eng_rss_kb=[0-9]+ read_bytes=[0-9]+ "
                r"cgroup_current_bytes=[0-9]+ cgroup_peak_bytes=[0-9]+ "
                r"cgroup_swap_current_bytes=0",
                row,
            )
            if match is None:
                return False
            value = int(match.group(1))
            minimum = value if minimum is None else min(minimum, value)
        return minimum is not None and minimum >= 10 * 1024 * 1024
    except (OSError, UnicodeError):
        return False


def _response_ok(arm: Path, role: str, expected_tokens: int) -> bool:
    try:
        response = _json(arm / f"{role}-response.json")
        return (
            (arm / f"{role}-http-status").read_text().strip() == "200"
            and response.get("usage", {}).get("prompt_tokens") == expected_tokens
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _recompute_trace(
    arm: Path,
    trace_scorer: Path,
    pool: Path,
    tokenizer: Path,
    tokenizer_runtime: Path,
) -> None:
    result = subprocess.run(
        [
            "/usr/bin/python3", "-I", "-B", str(trace_scorer),
            "--trace", str(arm / "request.trace"),
            "--pool", str(pool),
            "--live-request", str(arm / "live-request.json"),
            "--primary-request", str(arm / "primary-request.json"),
            "--tokenizer", str(tokenizer),
            "--tokenizer-runtime", str(tokenizer_runtime),
        ],
        capture_output=True,
        check=False,
        timeout=300,
    )
    (arm / "trace-scorer.recomputed.stderr").write_bytes(result.stderr)
    (arm / "trace-scorer.rc").write_text(f"{result.returncode}\n")
    (arm / "trace-result.json").write_bytes(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"authoritative trace scorer failed for {arm}")


def _artifact_rows(root: Path) -> list[dict]:
    rows = []
    for arm_name in ("strict", "candidate", "cold"):
        arm = root / arm_name
        artifacts = {}
        for path in sorted(arm.rglob("*")):
            if not path.is_file() or path.is_symlink() or "kv" in path.relative_to(arm).parts[:1]:
                continue
            artifacts[str(path.relative_to(arm))] = _sha256(path)
        rows.append({"schema_version": 1, "arm": arm_name, "artifacts": artifacts})
    return rows


def _expected_arm_order(seed_sha256: str) -> list[str]:
    seed = bytes.fromhex(seed_sha256)
    return sorted(
        ("strict", "candidate", "cold"),
        key=lambda arm: hashlib.sha256(seed + arm.encode()).digest(),
    )


def write_evidence_contract(
    root: Path, bindings: dict[str, str], arm_order: list[str],
    engine_source_commit: str = "0" * 40,
    binary_path: str = "/sealed/ds4",
    binary_device_inode: str = "1:2",
) -> None:
    if arm_order != _expected_arm_order(bindings.get("seed_sha256", "")):
        raise ValueError("invalid arm order")
    if set(bindings) != REQUIRED_BINDINGS or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in bindings.values()
    ):
        raise ValueError("invalid evidence binding")
    if re.fullmatch(r"[0-9a-f]{40}", engine_source_commit) is None:
        raise ValueError("invalid engine source commit")
    if not binary_path.startswith("/") or re.fullmatch(
        r"[0-9]+:[0-9]+", binary_device_inode
    ) is None:
        raise ValueError("invalid binary identity")
    rows = _artifact_rows(root)
    raw_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    (root / "raw.jsonl").write_bytes(raw_bytes)
    manifest = {
        "schema_version": 1,
        "gate": "W7-resume-bpe-lineage-v1",
        "bindings": bindings,
        "engine_source_commit": engine_source_commit,
        "binary_identity": {
            "path": binary_path, "device_inode": binary_device_inode,
        },
        "arm_order": arm_order,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _evidence_contract_pass(
    root: Path, expected_bindings: dict[str, str], expected_source_commit: str,
    expected_binary_path: str, expected_binary_device_inode: str,
) -> bool:
    try:
        manifest = _json(root / "manifest.json")
        raw_bytes = (root / "raw.jsonl").read_bytes()
        rows = [
            json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
            for line in raw_bytes.splitlines()
        ]
        bindings = manifest.get("bindings")
        return (
            set(manifest) == {"schema_version", "gate", "bindings", "engine_source_commit", "binary_identity", "arm_order", "raw_sha256"}
            and manifest["schema_version"] == 1
            and manifest["gate"] == "W7-resume-bpe-lineage-v1"
            and isinstance(bindings, dict)
            and set(bindings) == REQUIRED_BINDINGS
            and bindings == expected_bindings
            and all(
                isinstance(value, str)
                and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                for value in bindings.values()
            )
            and manifest["engine_source_commit"] == expected_source_commit
            and manifest["binary_identity"] == {
                "path": expected_binary_path,
                "device_inode": expected_binary_device_inode,
            }
            and manifest["arm_order"] == _expected_arm_order(bindings["seed_sha256"])
            and manifest["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
            and rows == _artifact_rows(root)
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def score(
    strict: Path, candidate: Path, cold: Path,
    expected_bindings: dict[str, str] | None = None,
    expected_source_commit: str | None = None,
    expected_binary_path: str | None = None,
    expected_binary_device_inode: str | None = None,
) -> dict:
    arms = {"strict": strict, "candidate": candidate, "cold": cold}
    checks: dict[str, bool] = {}
    logs: dict[str, str] = {}
    metadata: dict[str, dict] = {}
    for name, arm in arms.items():
        try:
            metadata[name] = _json(arm / "arm.json")
            logs[name] = (arm / "server.log").read_text(
                encoding="utf-8", errors="strict"
            )
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
            metadata[name] = {}
            logs[name] = ""

    checks["arm_identity_exact"] = all(
        metadata[name].get("schema_version") == 1
        and metadata[name].get("arm") == name
        and metadata[name].get("containment_rc") == 0
        for name in arms
    )
    expected_requests = {
        "strict": {"live": LIVE_SHA256, "primary": PRIMARY_SHA256},
        "candidate": {"live": LIVE_SHA256, "primary": PRIMARY_SHA256},
        "cold": {"primary": PRIMARY_SHA256},
    }
    actual_requests: dict[str, dict[str, str]] = {}
    try:
        for name, arm in arms.items():
            actual_requests[name] = {"primary": _sha256(arm / "primary-request.json")}
            if name != "cold":
                actual_requests[name]["live"] = _sha256(arm / "live-request.json")
        checks["request_hashes_exact"] = all(
            actual_requests[name] == expected_requests[name]
            and metadata[name].get("request_sha256") == actual_requests[name]
            for name in arms
        ) and (
            strict.joinpath("primary-request.json").read_bytes()
            == candidate.joinpath("primary-request.json").read_bytes()
            == cold.joinpath("primary-request.json").read_bytes()
        ) and (
            strict.joinpath("live-request.json").read_bytes()
            == candidate.joinpath("live-request.json").read_bytes()
        )
    except OSError:
        checks["request_hashes_exact"] = False
    checks["http_and_prompt_tokens_exact"] = (
        _response_ok(strict, "live", 5055)
        and _response_ok(strict, "primary", 5066)
        and _response_ok(candidate, "live", 5055)
        and _response_ok(candidate, "primary", 5066)
        and _response_ok(cold, "primary", 5066)
    )
    checks["traces_pass"] = (
        _trace_pass(strict / "trace-result.json")
        and _trace_pass(candidate / "trace-result.json")
    )
    if checks["traces_pass"]:
        strict_trace = _json(strict / "trace-result.json")
        candidate_trace = _json(candidate / "trace-result.json")
        expected_observed = [LIVE_SHA256, PRIMARY_SHA256]
        checks["traces_pass"] = all(
            [item["request_sha256"] for item in trace["observed"]]
            == expected_observed
            for trace in (strict_trace, candidate_trace)
        )
    checks["dump_sets_exact"] = all(
        {path.name for path in arm.glob("logits*")} == EXPECTED_DUMPS[name]
        for name, arm in arms.items()
    )
    checks["strict_guard_observed"] = (
        "GLM resume guard: prompt (5066) extends/diverges past evaluated frontier 5055 (checkpoint 5044)"
        in logs["strict"]
        and "GLM sync start=0 prompt=5066 suffix=5066" in logs["strict"]
        and "restored-frontier diagnostic: authoritative" not in logs["strict"]
    )
    checks["candidate_frontier_observed"] = (
        "GLM restored-frontier diagnostic: authoritative checkpoint=5044 compact_rows=5044"
        in logs["candidate"]
        and "GLM sync start=5044 prompt=5066 suffix=22" in logs["candidate"]
        and "GLM resume guard:" not in logs["candidate"]
    )
    checks["cold_control_observed"] = (
        "GLM sync start=0 prompt=5066 suffix=5066" in logs["cold"]
        and "GLM resume guard:" not in logs["cold"]
        and "restored-frontier diagnostic:" not in logs["cold"]
    )
    strict_kv = _selected_kv_identity(strict, logs["strict"])
    candidate_kv = _selected_kv_identity(candidate, logs["candidate"])
    checks["selected_kv_unchanged"] = strict_kv is not None and candidate_kv is not None
    checks["selected_kv_cross_arm_exact"] = (
        strict_kv is not None and strict_kv == candidate_kv
    )
    expected_bindings = expected_bindings or {}
    expected_binary = expected_bindings.get("binary_sha256", "")
    checks["safety_evidence_pass"] = (
        re.fullmatch(r"[0-9a-f]{64}", expected_binary) is not None
        and isinstance(expected_binary_path, str)
        and isinstance(expected_binary_device_inode, str)
        and all(_safety_pass(
            arm, expected_binary, expected_binary_path,
            expected_binary_device_inode,
        ) for arm in arms.values())
    )
    checks["evidence_contract_pass"] = (
        expected_source_commit is not None
        and expected_binary_path is not None
        and expected_binary_device_inode is not None
        and _evidence_contract_pass(
            strict.parent, expected_bindings, expected_source_commit,
            expected_binary_path, expected_binary_device_inode,
        )
    )

    observed = {
        "candidate_argmax": None,
        "cold_argmax": None,
        "strict_argmax": None,
        "max_abs_logit_delta": None,
        "strict_vs_cold_max_abs_logit_delta": None,
    }
    try:
        candidate_logits = _logits(
            candidate / "logits.sync2.start5044.prompt5066.suffix22"
        )
        cold_logits = _logits(cold / "logits.sync1.start0.prompt5066.suffix5066")
        strict_logits = _logits(strict / "logits.sync2.start0.prompt5066.suffix5066")
        checks["candidate_logits_finite"] = all(map(math.isfinite, candidate_logits))
        checks["cold_logits_finite"] = all(map(math.isfinite, cold_logits))
        checks["strict_logits_finite"] = all(map(math.isfinite, strict_logits))
        if all(
            checks[key]
            for key in (
                "candidate_logits_finite",
                "cold_logits_finite",
                "strict_logits_finite",
            )
        ):
            observed["candidate_argmax"] = max(
                range(N_VOCAB), key=candidate_logits.__getitem__
            )
            observed["cold_argmax"] = max(range(N_VOCAB), key=cold_logits.__getitem__)
            observed["strict_argmax"] = max(
                range(N_VOCAB), key=strict_logits.__getitem__
            )
            observed["max_abs_logit_delta"] = max(
                abs(left - right)
                for left, right in zip(candidate_logits, cold_logits, strict=True)
            )
            observed["strict_vs_cold_max_abs_logit_delta"] = max(
                abs(left - right)
                for left, right in zip(strict_logits, cold_logits, strict=True)
            )
        checks["candidate_argmax_matches_cold"] = (
            observed["candidate_argmax"] is not None
            and observed["candidate_argmax"] == observed["cold_argmax"]
        )
        checks["candidate_max_abs_delta_lt_1e_2"] = (
            observed["max_abs_logit_delta"] is not None
            and observed["max_abs_logit_delta"] < 0.01
        )
        checks["strict_control_matches_cold"] = (
            observed["strict_argmax"] is not None
            and observed["strict_argmax"] == observed["cold_argmax"]
            and observed["strict_vs_cold_max_abs_logit_delta"] == 0.0
        )
    except (OSError, ValueError):
        checks["candidate_logits_finite"] = False
        checks["cold_logits_finite"] = False
        checks["strict_logits_finite"] = False
        checks["candidate_argmax_matches_cold"] = False
        checks["candidate_max_abs_delta_lt_1e_2"] = False
        checks["strict_control_matches_cold"] = False

    verdict = "PASS" if checks and all(checks.values()) else "FAIL"
    return {
        "schema_version": 1,
        "gate": "W7-resume-bpe-lineage-v1",
        "formula": {
            "max_abs_logit_delta": "max_i(abs(candidate_i-cold_i))",
            "threshold": "strictly less than 0.01",
            "argmax": "first maximum index must be identical",
        },
        "checks": checks,
        "observed": observed,
        "artifact_sha256": {
            name: hashlib.sha256((arm / "server.log").read_bytes()).hexdigest()
            if (arm / "server.log").is_file()
            else None
            for name, arm in arms.items()
        },
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-scorer", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-runtime", type=Path, required=True)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--scorer-sha256", required=True)
    parser.add_argument("--seed-sha256", required=True)
    parser.add_argument("--trace-scorer-sha256", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--tokenizer-init-sha256", required=True)
    parser.add_argument("--tokenizer-native-sha256", required=True)
    parser.add_argument("--cgroup-sha256", required=True)
    parser.add_argument("--safe-sha256", required=True)
    parser.add_argument("--memory-guard-sha256", required=True)
    parser.add_argument("--engine-freeze-sha256", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--engine-source-commit", required=True)
    parser.add_argument("--binary-path", required=True)
    parser.add_argument("--binary-device-inode", required=True)
    parser.add_argument("--arm-order", type=Path, required=True)
    args = parser.parse_args()
    for arm in (args.strict, args.candidate):
        _recompute_trace(
            arm, args.trace_scorer, args.pool, args.tokenizer, args.tokenizer_runtime
        )
    bindings = {
            "harness_sha256": args.harness_sha256,
            "binary_sha256": args.binary_sha256,
            "model_sha256": args.model_sha256,
            "scorer_sha256": args.scorer_sha256,
            "seed_sha256": args.seed_sha256,
            "trace_scorer_sha256": args.trace_scorer_sha256,
            "fixture_sha256": args.fixture_sha256,
            "tokenizer_sha256": args.tokenizer_sha256,
            "tokenizer_init_sha256": args.tokenizer_init_sha256,
            "tokenizer_native_sha256": args.tokenizer_native_sha256,
            "cgroup_sha256": args.cgroup_sha256,
            "safe_sha256": args.safe_sha256,
            "memory_guard_sha256": args.memory_guard_sha256,
            "engine_freeze_sha256": args.engine_freeze_sha256,
            "configuration_sha256": args.configuration_sha256,
            "live_request_sha256": LIVE_SHA256,
            "primary_request_sha256": PRIMARY_SHA256,
        }
    write_evidence_contract(
        args.strict.parent, bindings,
        args.arm_order.read_text(encoding="utf-8").splitlines(),
        args.engine_source_commit,
        args.binary_path,
        args.binary_device_inode,
    )
    result = score(
        args.strict, args.candidate, args.cold, bindings,
        args.engine_source_commit, args.binary_path, args.binary_device_inode,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
