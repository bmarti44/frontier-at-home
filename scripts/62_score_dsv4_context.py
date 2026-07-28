#!/usr/bin/env python3
"""Build and score the strict DeepSeek W11 evidence triplet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
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
DRAND_HOSTS = ("api.drand.sh", "api2.drand.sh", "api3.drand.sh")
DRAND_GENESIS_UNIX = 1_595_431_050
DRAND_PERIOD_SECONDS = 30
CONTEXT_GATE = "dsv4_reference_context"
SECURE_ENV = {
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
}
JOURNAL_TAG = "dsv4-context-witness"
QUALIFICATION_UNIT = "dsv4-context-graduation.service"


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
        [
            "/usr/bin/sudo",
            "-n",
            "-u",
            "dsv4",
            "/usr/bin/sha256sum",
            "--",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=SECURE_ENV,
    )
    digest = completed.stdout.split(maxsplit=1)[0]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"invalid protected artifact digest: {path}")
    return digest


def _journal_rows(since_seconds: float) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "/usr/bin/journalctl",
            "--since",
            f"@{since_seconds:.6f}",
            "--identifier",
            JOURNAL_TAG,
            "--output",
            "json",
            "--no-pager",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=SECURE_ENV,
    )
    rows = []
    for line in completed.stdout.splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def require_qualification_invocation() -> str:
    invocation = os.environ.get("INVOCATION_ID", "")
    if not (
        len(invocation) == 32
        and all(character in "0123456789abcdef" for character in invocation)
    ):
        raise RuntimeError("qualification INVOCATION_ID is absent or invalid")
    if os.geteuid() != 0:
        raise RuntimeError("qualification authority requires the root attestor")
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    expected_suffix = f"/{QUALIFICATION_UNIT}"
    if not any(
        line.split(":", 2)[-1].endswith(expected_suffix)
        for line in cgroup.splitlines()
    ):
        raise RuntimeError("process is outside the registered worker cgroup")
    return invocation


def emit_journal_witness(
    payload: dict[str, Any], *, required_user_unit: str | None = None
) -> dict[str, str]:
    """Bind canonical evidence to system-owned journal metadata."""
    message = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    trusted_uid = str(os.geteuid())
    started = time.time() - 1.0
    journal_process = subprocess.Popen(
        [
            "/usr/bin/systemd-cat",
            f"--identifier={JOURNAL_TAG}",
            "--priority=notice",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=SECURE_ENV,
    )
    try:
        assert journal_process.stdin is not None
        journal_process.stdin.write(message + "\n")
        journal_process.stdin.flush()
        for _ in range(40):
            matches = [
                row
                for row in _journal_rows(started)
                if row.get("MESSAGE") == message
                and row.get("_UID") == trusted_uid
            ]
            if matches:
                row = matches[-1]
                receipt = {
                    "cursor": str(row.get("__CURSOR", "")),
                    "realtime_timestamp": str(
                        row.get("__REALTIME_TIMESTAMP", "")
                    ),
                    "boot_id": str(row.get("_BOOT_ID", "")),
                    "invocation_id": str(
                        row.get("_SYSTEMD_INVOCATION_ID", "")
                    ),
                    "stream_id": str(row.get("_STREAM_ID", "")),
                    "pid": str(row.get("_PID", "")),
                    "uid": str(row.get("_UID", "")),
                    "cgroup": str(row.get("_SYSTEMD_CGROUP", "")),
                    "user_unit": str(row.get("_SYSTEMD_USER_UNIT", "")),
                    "system_unit": str(row.get("_SYSTEMD_UNIT", "")),
                }
                receipt["scope_id"] = (
                    receipt["invocation_id"] or receipt["stream_id"]
                )
                required_invocation = os.environ.get("INVOCATION_ID", "")
                if required_invocation and (
                    receipt["invocation_id"] != required_invocation
                ):
                    time.sleep(0.05)
                    continue
                if required_user_unit and (
                    receipt["system_unit"] != required_user_unit
                ):
                    time.sleep(0.05)
                    continue
                if all(
                    receipt[field]
                    for field in (
                        "cursor",
                        "realtime_timestamp",
                        "boot_id",
                        "pid",
                        "uid",
                        "scope_id",
                    )
                ):
                    return receipt
            time.sleep(0.05)
        raise RuntimeError("journal witness was not persisted")
    finally:
        if journal_process.stdin is not None:
            journal_process.stdin.close()
        try:
            journal_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            journal_process.terminate()
            journal_process.wait(timeout=2)
        if journal_process.stderr is not None:
            journal_process.stderr.close()


def verify_journal_witness(
    payload: dict[str, Any],
    receipt: dict[str, Any],
    *,
    required_user_unit: str | None = None,
) -> None:
    message = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    try:
        since = int(receipt["realtime_timestamp"]) / 1_000_000 - 1.0
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("journal witness receipt is invalid") from exc
    matches = [
        row
        for row in _journal_rows(since)
        if row.get("__CURSOR") == receipt.get("cursor")
        and row.get("MESSAGE") == message
        and row.get("_UID") == receipt.get("uid")
        and receipt.get("uid") in {"0", "1000"}
        and row.get("_BOOT_ID") == receipt.get("boot_id")
        and row.get("_SYSTEMD_INVOCATION_ID", "")
        == receipt.get("invocation_id")
        and row.get("_STREAM_ID", "") == receipt.get("stream_id", "")
        and row.get("_PID") == receipt.get("pid")
        and row.get("_SYSTEMD_CGROUP", "") == receipt.get("cgroup", "")
        and row.get("_SYSTEMD_USER_UNIT", "")
        == receipt.get("user_unit", "")
        and row.get("_SYSTEMD_UNIT", "")
        == receipt.get("system_unit", "")
    ]
    if len(matches) != 1:
        raise RuntimeError("journal witness does not match trusted record")
    if required_user_unit and (
        receipt.get("uid") != "0"
        or receipt.get("system_unit") != required_user_unit
        or not str(receipt.get("cgroup", "")).endswith(
            f"/{required_user_unit}"
        )
    ):
        raise RuntimeError("journal witness is outside the registered worker unit")


def create_artifact_witness(
    *,
    root: Path,
    event: str,
    candidate: str,
    seed: str,
    artifacts: list[str],
    claims: dict[str, Any] | None = None,
    required_user_unit: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    digests: dict[str, str] = {}
    for relative in sorted(artifacts):
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RuntimeError(f"artifact witness path is invalid: {relative}")
        digests[relative] = sha256(path)
    payload = {
        "event": event,
        "candidate_hash": candidate,
        "seed_sha256": seed,
        "artifacts": digests,
    }
    if claims is not None:
        payload["claims"] = claims
    return {
        "payload": payload,
        "receipt": emit_journal_witness(
            payload, required_user_unit=required_user_unit
        ),
    }


def verify_artifact_witness(
    *,
    root: Path,
    witness: dict[str, Any],
    required_user_unit: str | None = None,
) -> None:
    try:
        payload = witness["payload"]
        receipt = witness["receipt"]
        artifacts = payload["artifacts"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("artifact witness is malformed") from exc
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("artifact witness is empty")
    root = root.resolve()
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.is_symlink()
            or sha256(path) != expected
        ):
            raise RuntimeError(f"artifact witness changed: {relative}")
    verify_journal_witness(
        payload, receipt, required_user_unit=required_user_unit
    )


def fetch_public_drand(host: str, round_number: int) -> dict[str, Any]:
    if host not in DRAND_HOSTS:
        raise RuntimeError("unregistered drand relay")
    completed = subprocess.run(
        [
            "/usr/bin/curl",
            "--disable",
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            "10",
            "--proto",
            "=https",
            f"https://{host}/public/{round_number}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
        },
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("drand relay response is not an object")
    return value


def git_commit_time(candidate: str) -> str:
    return subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            f"safe.directory={LIVE_ROOT}",
            "-C",
            str(LIVE_ROOT),
            "show",
            "-s",
            "--format=%cI",
            candidate,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        },
    ).stdout.strip()


def capture_public_lineage(
    *,
    candidate: str,
    now: datetime,
    relay_fetcher: Any = fetch_public_drand,
    commit_time_fetcher: Any = git_commit_time,
    frozen_at: datetime | None = None,
    round_number: int | None = None,
) -> dict[str, Any]:
    """Capture three-relay randomness strictly after the candidate commit."""
    if now.tzinfo is None or now.utcoffset() != timezone.utc.utcoffset(now):
        raise RuntimeError("lineage capture time must be UTC")
    if round_number is None:
        round_number = (
            int(now.timestamp() - DRAND_GENESIS_UNIX)
            // DRAND_PERIOD_SECONDS
            + 1
        )
    if round_number < 1:
        raise RuntimeError("drand round is invalid")
    responses = [
        relay_fetcher(host, round_number) for host in DRAND_HOSTS
    ]
    fields = ("round", "randomness", "signature")
    beacon = {field: responses[0].get(field) for field in fields}
    if any(
        any(response.get(field) != beacon[field] for field in fields)
        for response in responses
    ):
        raise RuntimeError("public drand relays disagree")
    signature = beacon["signature"]
    if (
        not isinstance(signature, str)
        or len(signature) != 192
        or any(char not in "0123456789abcdef" for char in signature)
    ):
        raise RuntimeError("drand signature is invalid")
    randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
    if beacon["round"] != round_number or beacon["randomness"] != randomness:
        raise RuntimeError("drand beacon is invalid")
    if frozen_at is None:
        frozen_at_text = commit_time_fetcher(candidate)
        frozen_time = datetime.fromisoformat(frozen_at_text)
    else:
        frozen_time = frozen_at
        frozen_at_text = frozen_time.isoformat()
    beacon_time = datetime.fromtimestamp(
        DRAND_GENESIS_UNIX + (round_number - 1) * DRAND_PERIOD_SECONDS,
        timezone.utc,
    )
    if frozen_time.tzinfo is None or beacon_time <= frozen_time or now < beacon_time:
        raise RuntimeError("drand beacon was not published after candidate freeze")
    seed = hashlib.sha256(
        f"{candidate}:{randomness}:{CONTEXT_GATE}".encode()
    ).hexdigest()
    return {
        "freeze": {
            "candidate_hash": candidate,
            "frozen_at": frozen_at_text,
        },
        "randomness": {
            "source": "drand-default",
            "round": round_number,
            "randomness": randomness,
            "signature": signature,
            "obtained_at": now.isoformat(),
            "seed_sha256": seed,
        },
    }


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
            "/usr/bin/git",
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
        env=SECURE_ENV,
    ).stdout.strip()
    if resolved != candidate:
        raise RuntimeError("Git candidate does not resolve exactly")
    tree_output = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            f"safe.directory={LIVE_ROOT}",
            "-C",
            str(LIVE_ROOT),
            "ls-tree",
            "-rz",
            candidate,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=SECURE_ENV,
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
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    freeze_witness = json.loads(
        (out / "freeze-witness.json").read_text(encoding="utf-8")
    )
    verify_journal_witness(
        freeze_witness["payload"],
        freeze_witness["receipt"],
        required_user_unit=QUALIFICATION_UNIT,
    )
    if (
        freeze_witness["payload"].get("event") != "candidate-freeze"
        or freeze_witness["payload"].get("candidate_hash") != candidate
    ):
        raise RuntimeError("candidate freeze witness is invalid")
    witnessed_at = datetime.fromtimestamp(
        int(freeze_witness["receipt"]["realtime_timestamp"]) / 1_000_000,
        timezone.utc,
    )
    if datetime.fromisoformat(lineage["freeze"]["frozen_at"]) != witnessed_at:
        raise RuntimeError("lineage freeze time differs from journal witness")
    goal.validate_manifest_lineage(
        lineage,
        CONTEXT_GATE,
        candidate,
        relay_fetcher=fetch_public_drand,
    )
    if lineage["randomness"]["seed_sha256"] != seed:
        raise RuntimeError("worker seed differs from public lineage")

    caps = (131_072, 262_144, 524_288, 1_048_576)
    targets = (130_000, 260_000, 520_000, 1_000_000)
    stages = []
    retrieval_results = []
    fixture_hashes = []
    worker_invocation = freeze_witness["receipt"].get("invocation_id")
    for index, (cap, target) in enumerate(zip(caps, targets)):
        stage_path = out / f"stage-{cap}.json"
        engine_path = out / f"engine-{cap}.log"
        if not stage_path.is_file() or not engine_path.is_file():
            raise RuntimeError(f"missing stage evidence for {cap}")
        stage_witness = json.loads(
            (out / f"witness-stage-{cap}.json").read_text(encoding="utf-8")
        )
        verify_artifact_witness(
            root=out,
            witness=stage_witness,
            required_user_unit=QUALIFICATION_UNIT,
        )
        if (
            stage_witness["payload"].get("event")
            != f"stage-{cap}-complete"
            or stage_witness["payload"].get("candidate_hash") != candidate
            or stage_witness["payload"].get("seed_sha256") != seed
            or stage_witness["receipt"].get("invocation_id")
            != worker_invocation
        ):
            raise RuntimeError(f"stage {cap} journal witness is invalid")
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
        "gate": CONTEXT_GATE,
        "qualification_authority": False,
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
        "lineage": lineage,
        "freeze_witness": freeze_witness,
    }
    return observation, {"manifest": manifest, "summary": summary}


def finalize_attempt(
    *,
    out: Path,
    candidate: str,
    seed: str,
    mode: str,
    lifecycle_exit_status: int,
) -> dict[str, Any]:
    """Recompute any PASS and place its authority inside the final seal."""
    invocation = require_qualification_invocation()
    persisted = {
        "manifest": json.loads(
            (out / "manifest.json").read_text(encoding="utf-8")
        ),
        "raw": json.loads((out / "raw.jsonl").read_text(encoding="utf-8")),
        "summary": json.loads(
            (out / "summary.json").read_text(encoding="utf-8")
        ),
    }
    verdict = persisted["summary"].get("verdict")
    if verdict not in {"PASS", "FAIL"}:
        raise RuntimeError("persisted summary verdict is invalid")
    if verdict == "PASS":
        observation, result = build_observation(
            out,
            candidate,
            seed,
            mode,
            lifecycle_exit_status,
        )
        freeze_invocation = (
            result["manifest"]
            .get("freeze_witness", {})
            .get("receipt", {})
            .get("invocation_id")
        )
        if freeze_invocation != invocation:
            raise RuntimeError(
                "final invocation differs from freeze/stage invocation"
            )
        expected = {
            "manifest": result["manifest"],
            "raw": observation,
            "summary": result["summary"],
        }
        for name in ("manifest", "raw", "summary"):
            if persisted[name] != expected[name]:
                raise RuntimeError(f"persisted {name} differs from recomputation")
        if (
            persisted["manifest"].get("candidate_hash") != candidate
            or persisted["manifest"].get("seed_sha256") != seed
            or persisted["manifest"].get("qualification_authority") is not False
        ):
            raise RuntimeError("persisted manifest identity is invalid")

    artifacts = ["manifest.json", "raw.jsonl", "summary.json"]
    scoring_inputs = (
        "lineage.json",
        "freeze-witness.json",
        "memory.jsonl",
        "kernel.log",
    )
    for name in scoring_inputs:
        if (out / name).is_file():
            artifacts.append(name)
    for cap in (131_072, 262_144, 524_288, 1_048_576):
        for name in (
            f"stage-{cap}.json",
            f"engine-{cap}.log",
            f"witness-stage-{cap}.json",
        ):
            if (out / name).is_file():
                artifacts.append(name)
    if verdict == "PASS" and len(artifacts) != 19:
        raise RuntimeError("PASS finalization is missing mandatory evidence")
    claims = {
        "gate": CONTEXT_GATE,
        "verdict": verdict,
        "qualification_authority": verdict == "PASS",
        "worker_invocation_id": invocation,
    }
    witness = create_artifact_witness(
        root=out,
        event="context-attempt-complete",
        candidate=candidate,
        seed=seed,
        artifacts=artifacts,
        claims=claims,
        required_user_unit=QUALIFICATION_UNIT,
    )
    verify_artifact_witness(
        root=out,
        witness=witness,
        required_user_unit=QUALIFICATION_UNIT,
    )
    if witness["receipt"].get("invocation_id") != invocation:
        raise RuntimeError("final witness invocation differs from worker")
    (out / "journal-seal.json").write_text(
        json.dumps(witness, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return witness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--candidate-hash", required=True)
    parser.add_argument("--seed-sha256")
    parser.add_argument("--mode")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--lifecycle-exit-status", type=int, default=0)
    parser.add_argument("--capture-lineage", type=Path)
    parser.add_argument("--witness-stage", type=int)
    parser.add_argument("--witness-final", action="store_true")
    parser.add_argument("--record-preflight-failure")
    args = parser.parse_args()
    if args.record_preflight_failure is not None:
        if args.out is None:
            parser.error("--record-preflight-failure requires --out")
        write_failure_triplet(
            out=args.out,
            candidate=args.candidate_hash,
            seed=args.seed_sha256 or "0" * 64,
            mode=args.mode or "graduated",
            error=RuntimeError(args.record_preflight_failure),
        )
        return 0
    if args.capture_lineage is not None:
        require_qualification_invocation()
        freeze_payload = {
            "event": "candidate-freeze",
            "candidate_hash": args.candidate_hash,
            "nonce": secrets.token_hex(16),
        }
        freeze_witness = {
            "payload": freeze_payload,
            "receipt": emit_journal_witness(
                freeze_payload, required_user_unit=QUALIFICATION_UNIT
            ),
        }
        frozen_at = datetime.fromtimestamp(
            int(freeze_witness["receipt"]["realtime_timestamp"]) / 1_000_000,
            timezone.utc,
        )
        round_number = (
            int(frozen_at.timestamp() - DRAND_GENESIS_UNIX)
            // DRAND_PERIOD_SECONDS
            + 2
        )
        beacon_time = datetime.fromtimestamp(
            DRAND_GENESIS_UNIX
            + (round_number - 1) * DRAND_PERIOD_SECONDS,
            timezone.utc,
        )
        delay = (beacon_time - datetime.now(timezone.utc)).total_seconds() + 0.5
        if delay > 35:
            raise RuntimeError("next drand beacon wait is unexpectedly long")
        if delay > 0:
            time.sleep(delay)
        lineage = capture_public_lineage(
            candidate=args.candidate_hash,
            now=datetime.now(timezone.utc),
            frozen_at=frozen_at,
            round_number=round_number,
        )
        args.capture_lineage.write_text(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        args.capture_lineage.with_name("freeze-witness.json").write_text(
            json.dumps(
                freeze_witness, sort_keys=True, separators=(",", ":")
            )
            + "\n",
            encoding="utf-8",
        )
        print(lineage["randomness"]["seed_sha256"])
        return 0
    if args.witness_stage is not None:
        if args.out is None or args.seed_sha256 is None:
            parser.error("--witness-stage requires --out and --seed-sha256")
        require_qualification_invocation()
        cap = args.witness_stage
        if cap not in {131_072, 262_144, 524_288, 1_048_576}:
            parser.error("--witness-stage cap is invalid")
        witness = create_artifact_witness(
            root=args.out,
            event=f"stage-{cap}-complete",
            candidate=args.candidate_hash,
            seed=args.seed_sha256,
            artifacts=[f"stage-{cap}.json", f"engine-{cap}.log"],
            required_user_unit=QUALIFICATION_UNIT,
        )
        (args.out / f"witness-stage-{cap}.json").write_text(
            json.dumps(witness, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.witness_final:
        if (
            args.out is None
            or args.seed_sha256 is None
            or args.mode is None
        ):
            parser.error(
                "--witness-final requires --out, --seed-sha256 and --mode"
            )
        finalize_attempt(
            out=args.out,
            candidate=args.candidate_hash,
            seed=args.seed_sha256,
            mode=args.mode,
            lifecycle_exit_status=args.lifecycle_exit_status,
        )
        return 0
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
