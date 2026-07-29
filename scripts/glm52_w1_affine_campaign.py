#!/usr/bin/env python3
"""Run and resume the strict, memory-contained W1 affine quality campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SCORER = ROOT / "scripts/glm52_goal.py"
MASTER_MANIFEST = Path(
    "gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv"
)
COMMON_ENGINE_ENVIRONMENT = {
    "DS4_LOCK_FILE": "/run/user/1000/ds4-engine.lock",
    "DS4_CUDA_EXPERT_CACHE_GB": "0",
    "DS4_CUDA_EXPERT_CACHE_PIN": "1",
    "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
    "DS4_CUDA_FETCH_THREADS": "6",
    "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
    "DS4_CUDA_MOE_NO_EXPERT_TILES": "1",
}
FIDELITY_ENVIRONMENT_NAMES = (
    "DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE",
    "DS4_GLM_COMPACT_CACHE_E4M3_FAKE",
    "DS4_GLM_COMPACT_CACHE_F16",
    "DS4_GLM_COMPACT_CACHE_INT8_FAKE",
)
FORWARDED_ENGINE_ENVIRONMENT_NAMES = (
    "GLM_EXPERT_CACHE_GB",
    "GLM_PORT",
    "GLM_REQUIRE_TOKEN_TIMING_LOG",
    "DS4_CUDA_IQ2_DOWN_REFERENCE",
    *COMMON_ENGINE_ENVIRONMENT,
    *FIDELITY_ENVIRONMENT_NAMES,
)
PROVENANCE_NAMES = tuple(
    sorted((*COMMON_ENGINE_ENVIRONMENT, *FIDELITY_ENVIRONMENT_NAMES))
)
SAFE_ENVIRONMENT = {
    "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
    "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
    "GLM_SAFE_KILL_FLOOR_GIB": "40",
    "GLM_SAFE_MIN_START_GIB": "110",
    "GLM_SAFE_TIMEOUT_S": "1800",
}
START_RE = re.compile(
    r"^ds4: GLM compact cache fidelity resolved_mode=(\d+)$", re.MULTILINE
)
EXIT_RE = re.compile(
    r"^ds4: GLM compact cache fidelity attestation resolved_mode=(\d+) "
    r"affine_store_rows=(\d+)$",
    re.MULTILINE,
)
FAULT_RE = re.compile(
    r"NVRM.*Xid|NV_ERR_NO_MEMORY|oom-kill|Out of memory|Memory cgroup out of memory",
    re.IGNORECASE,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def candidate_arm(seed_sha256: str) -> str:
    _validate_sha256(seed_sha256, "seed")
    return "A" if int(seed_sha256[:2], 16) % 2 == 0 else "B"


def schedules(seed_sha256: str) -> tuple[str, ...]:
    _validate_sha256(seed_sha256, "seed")
    first = "ABBA" if int(seed_sha256[2:4], 16) % 2 == 0 else "BAAB"
    other = "BAAB" if first == "ABBA" else "ABBA"
    return tuple(first if block % 2 == 0 else other for block in range(5))


def environment_sha256(names: Iterable[str], values: dict[str, str]) -> str:
    canonical = b"".join(
        name.encode("ascii")
        + b"="
        + values.get(name, "<UNSET>").encode("ascii")
        + b"\n"
        for name in sorted(names)
    )
    return sha256_bytes(canonical)


def confirmation_seed(
    drand_randomness: str,
    engine_commit: str,
    binary_sha256: str,
    harness_commit: str,
) -> str:
    for value, label, length in (
        (drand_randomness, "drand randomness", 64),
        (engine_commit, "engine commit", 40),
        (binary_sha256, "binary", 64),
        (harness_commit, "harness commit", 40),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            raise ValueError(f"{label} is invalid")
    digest = hashlib.sha256()
    for value in (
        drand_randomness,
        engine_commit,
        binary_sha256,
        harness_commit,
    ):
        digest.update(bytes.fromhex(value))
    digest.update(b"affine-int8-b16-quality-v3-strict-abba")
    return digest.hexdigest()


def _manifest_rows(manifest: Path) -> list[tuple[str, str, str, str]]:
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [
            tuple(line)
            for line in csv.reader(
                (raw for raw in stream if not raw.startswith("#")),
                delimiter="\t",
            )
            if line
        ]
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"malformed fixture manifest: {manifest}")
    return rows  # type: ignore[return-value]


def content_complete_fixture_sha256(
    source: Path, manifests: Iterable[Path]
) -> str:
    source = source.resolve()
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    for manifest in manifests:
        raw = manifest.read_bytes()
        name = manifest.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        for case_id, *relative_paths in _manifest_rows(manifest):
            if case_id in seen_ids:
                raise ValueError(f"duplicate fixture case: {case_id}")
            seen_ids.add(case_id)
            for relative in relative_paths:
                path = (source / relative).resolve()
                if not path.is_relative_to(source) or not path.is_file():
                    raise ValueError(f"fixture path escapes or is absent: {relative}")
                relative_bytes = relative.encode("utf-8")
                data = path.read_bytes()
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
    if not seen_ids:
        raise ValueError("fixture is empty")
    return digest.hexdigest()


def parse_attestation(log: str) -> tuple[int, int]:
    starts = START_RE.findall(log)
    exits = EXIT_RE.findall(log)
    if len(starts) != 1 or len(exits) != 1:
        raise ValueError("runtime mode attestation is missing or duplicated")
    start_mode = int(starts[0])
    exit_mode, rows = map(int, exits[0])
    if start_mode != exit_mode:
        raise ValueError("runtime mode changed within one process")
    return start_mode, rows


def parse_quality_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 20:
        raise ValueError(f"quality output needs 20 cases, got {len(rows)}")
    cases: list[dict[str, Any]] = []
    for row in rows:
        try:
            case = {
                "case_id": row["id"],
                "tokens": int(row["target_tokens"]),
                "nll_sum": float(row["nll"]),
                "top1_correct": int(row["target_top1_correct"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("quality output contains malformed values") from exc
        if (
            not case["case_id"]
            or case["tokens"] <= 0
            or not 0 <= case["top1_correct"] <= case["tokens"]
            or not float("-inf") < case["nll_sum"] < float("inf")
        ):
            raise ValueError("quality output contains invalid values")
        cases.append(case)
    if len({case["case_id"] for case in cases}) != 20:
        raise ValueError("quality output case IDs are duplicated")
    return cases


def _validate_sha256(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _source_commit(source: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    if status.stdout:
        raise ValueError("engine source has tracked modifications")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("engine candidate commit is invalid")
    return commit


def _drand_record(path: Path) -> dict[str, Any]:
    record = _strict_json(path)
    if not isinstance(record, dict):
        raise ValueError("drand record must be an object")
    round_number = record.get("round")
    randomness = record.get("randomness")
    signature = record.get("signature")
    if (
        not isinstance(round_number, int)
        or isinstance(round_number, bool)
        or round_number <= 0
        or not isinstance(randomness, str)
        or not re.fullmatch(r"[0-9a-f]{64}", randomness)
        or not isinstance(signature, str)
        or not re.fullmatch(r"[0-9a-f]+", signature)
    ):
        raise ValueError("drand record is malformed")
    return {
        "round": round_number,
        "randomness": randomness,
        "signature": signature,
    }


def _write_manifests(source: Path, output: Path, seed: str) -> list[Path]:
    master = source / MASTER_MANIFEST
    rows = _manifest_rows(master)
    if len(rows) != 100 or len({row[0] for row in rows}) != 100:
        raise ValueError("master fixture must contain 100 unique cases")
    random.Random(int(seed, 16)).shuffle(rows)
    manifest_dir = output / "fixtures"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    for block in range(5):
        manifest = manifest_dir / f"block-{block}.tsv"
        text = "# id\tprompt_file\tcontinuation_file\tresponse_file\n"
        text += "".join("\t".join(row) + "\n" for row in rows[block * 20 : (block + 1) * 20])
        if manifest.exists() and manifest.read_text(encoding="utf-8") != text:
            raise ValueError("existing fixture manifest differs")
        if not manifest.exists():
            manifest.write_text(text, encoding="utf-8")
        manifests.append(manifest)
    return manifests


def _journal_cursor() -> str:
    result = subprocess.run(
        ["journalctl", "-k", "-n", "0", "--show-cursor", "--no-pager"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    matches = re.findall(r"^-- cursor: (.+)$", result.stdout, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("cannot capture kernel journal cursor")
    return matches[0]


def _kernel_since(cursor: str) -> str:
    return subprocess.run(
        ["journalctl", "-k", f"--after-cursor={cursor}", "--no-pager"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout


def _server_instance(main_log: str) -> str:
    matches = re.findall(
        r"executed_candidate_verified pid=(\d+) start_ticks=(\d+)", main_log
    )
    if len(matches) != 1:
        raise ValueError("executed process identity is missing or duplicated")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return f"{boot_id}:{matches[0][0]}:{matches[0][1]}"


def _minimum_available_gib(samples: str) -> float:
    values = [int(value) for value in re.findall(r"mem_avail_kb=(\d+)", samples)]
    if not values:
        raise ValueError("memory sampler produced no measurements")
    return min(values) / 1048576


def _score(campaign: dict[str, Any]) -> dict[str, Any]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("glm52_goal_fixed", SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load fixed scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._score_w1_affine([campaign])


def _freeze_scorer(source: Path, engine_commit: str, binary_sha256: str) -> Path:
    frozen = Path(
        f"/home/bmarti44/.cache/glm52-w1-affine-score-{engine_commit[:12]}"
    )
    frozen.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = frozen / "ds4-server"
    scorer = source / "gguf-tools/quality-testing/score_official"
    if not target.exists():
        temporary = frozen / ".ds4-server.tmp"
        shutil.copyfile(scorer, temporary)
        os.chmod(temporary, 0o500)
        os.replace(temporary, target)
    if sha256_file(target) != binary_sha256:
        raise ValueError("frozen quality binary hash mismatch")
    return frozen


def _campaign_paths(seed: str, engine_commit: str, output: Path | None) -> Path:
    return output or Path(
        f"/home/bmarti44/.local/state/glm52-confirm-w1-affine-"
        f"{engine_commit[:12]}-{seed[:12]}"
    )


def run(args: argparse.Namespace) -> int:
    source = args.engine_source.resolve()
    model = args.model.resolve()
    if not model.is_file() or not source.is_dir():
        raise ValueError("engine source or model is absent")
    engine_commit = _source_commit(source)
    if args.engine_candidate_hash and args.engine_candidate_hash != engine_commit:
        raise ValueError("engine candidate hash changed")
    scorer_binary = source / "gguf-tools/quality-testing/score_official"
    binary_sha256 = sha256_file(scorer_binary)
    harness_commit = _source_commit(ROOT)
    drand = _drand_record(args.drand_json.resolve())
    seed = confirmation_seed(
        drand["randomness"], engine_commit, binary_sha256, harness_commit
    )
    frozen = _freeze_scorer(source, engine_commit, binary_sha256)
    output = _campaign_paths(seed, engine_commit, args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifests = _write_manifests(source, output, seed)
    fixture_sha256 = content_complete_fixture_sha256(source, manifests)

    common = dict(COMMON_ENGINE_ENVIRONMENT)
    baseline_environment = dict(common)
    candidate_environment = {
        **common,
        "DS4_GLM_COMPACT_CACHE_AFFINE_INT8_FAKE": "1",
    }
    baseline_environment_sha256 = environment_sha256(
        PROVENANCE_NAMES, baseline_environment
    )
    candidate_environment_sha256 = environment_sha256(
        PROVENANCE_NAMES, candidate_environment
    )
    configuration = {
        "schema_version": 1,
        "harness_candidate_hash": harness_commit,
        "engine_candidate_hash": engine_commit,
        "binary_sha256": binary_sha256,
        "model": str(model),
        "launch_arguments": [
            str(model),
            "{manifest}",
            "{output}",
            "8192",
            "--ssd-streaming",
            "--ssd-streaming-cache-experts",
            "40GB",
        ],
        "provenance_environment_names": list(PROVENANCE_NAMES),
        "baseline_environment": baseline_environment,
        "candidate_environment": candidate_environment,
        "schedules": list(schedules(seed)),
        "public_randomness": {
            **drand,
            "seed_sha256": seed,
            "seed_formula": (
                "sha256(drand_randomness_bytes || engine_commit_bytes || "
                "binary_sha256_bytes || harness_commit_bytes || "
                "affine-int8-b16-quality-v3-strict-abba)"
            ),
        },
        "safety": SAFE_ENVIRONMENT,
    }
    configuration_bytes = (
        json.dumps(configuration, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    configuration_path = output / "configuration.json"
    if (
        configuration_path.exists()
        and configuration_path.read_bytes() != configuration_bytes
    ):
        raise ValueError("existing campaign configuration differs")
    configuration_path.write_bytes(configuration_bytes)
    configuration_sha256 = sha256_bytes(configuration_bytes)

    model_digest_path = output / "model.sha256"
    if model_digest_path.exists():
        model_sha256 = model_digest_path.read_text().strip()
        _validate_sha256(model_sha256, "recorded model")
    else:
        model_sha256 = sha256_file(model)
        model_digest_path.write_text(model_sha256 + "\n", encoding="ascii")
    if args.model_sha256 and args.model_sha256 != model_sha256:
        raise ValueError("model hash differs from expected digest")

    campaign_path = output / "campaign.json"
    campaign = {
        "record_type": "w1_affine_campaign",
        "engine_candidate_hash": engine_commit,
        "seed_sha256": seed,
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "fixture_sha256": fixture_sha256,
        "baseline_environment_sha256": baseline_environment_sha256,
        "candidate_environment_sha256": candidate_environment_sha256,
        "candidate_arm": candidate_arm(seed),
        "attempts": [],
    }
    if campaign_path.exists():
        existing = _strict_json(campaign_path)
        expected_without_attempts = {**campaign, "attempts": existing.get("attempts")}
        if existing != expected_without_attempts:
            raise ValueError("existing campaign identity differs")
        campaign = existing
    _atomic_json(output / "run-metadata.json", {
        "schema_version": 1,
        "model_sha256": model_sha256,
        "engine_source": str(source),
        "frozen_binary": str(frozen / "ds4-server"),
        "fixture_manifests": [str(path) for path in manifests],
    })
    _atomic_json(output / "randomness.json", configuration["public_randomness"])
    _atomic_json(campaign_path, campaign)

    flattened = [
        (block, sequence, arm)
        for block, order in enumerate(schedules(seed))
        for sequence, arm in enumerate(order)
    ]
    for index in range(len(campaign["attempts"]), 20):
        block, sequence, arm = flattened[index]
        is_candidate = arm == campaign["candidate_arm"]
        engine_environment = (
            candidate_environment if is_candidate else baseline_environment
        )
        expected_environment_sha256 = (
            candidate_environment_sha256
            if is_candidate
            else baseline_environment_sha256
        )
        before = content_complete_fixture_sha256(source, manifests)
        if before != fixture_sha256:
            raise ValueError("fixture bytes changed before attempt")
        cursor = _journal_cursor()
        result_path = output / f"attempt-{index:02d}.tsv"
        log_path = output / f"attempt-{index:02d}.launcher.log"
        kernel_path = output / f"attempt-{index:02d}.kernel.log"
        if result_path.exists():
            raise ValueError(f"stale attempt output exists: {result_path}")
        environment = os.environ.copy()
        for name in FORWARDED_ENGINE_ENVIRONMENT_NAMES:
            environment.pop(name, None)
        environment.update(SAFE_ENVIRONMENT)
        environment.update(engine_environment)
        environment.update(
            {
                "GLM_CANDIDATE_SRC": str(frozen),
                "GLM_SAFE_EXPECTED_BINARY_SHA256": binary_sha256,
                "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(PROVENANCE_NAMES),
                "GLM_SAFE_EXPECTED_ENV_SHA256": expected_environment_sha256,
            }
        )
        command = [
            str(LAUNCHER),
            "--tag",
            f"w1-{seed[:8]}-{index:02d}-{arm}",
            "--",
            str(frozen / "ds4-server"),
            str(model),
            str(manifests[block]),
            str(result_path),
            "8192",
            "--ssd-streaming",
            "--ssd-streaming-cache-experts",
            "40GB",
        ]
        result = subprocess.run(
            command,
            cwd=source,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1900,
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        kernel = _kernel_since(cursor)
        kernel_path.write_text(kernel, encoding="utf-8")
        failures: list[str] = []
        if result.returncode != 0:
            failures.append(f"launcher_rc={result.returncode}")
        safe_dirs = re.findall(r"SAFE_RUN_DONE .* dir=(\S+)$", result.stdout, re.MULTILINE)
        main_log = ""
        samples = ""
        command_log = ""
        if len(safe_dirs) != 1:
            failures.append("safe_run_directory_missing_or_duplicated")
        else:
            safe_dir = Path(safe_dirs[0])
            try:
                main_log = (safe_dir / "main.log").read_text(encoding="utf-8")
                samples = (safe_dir / "samples.log").read_text(encoding="utf-8")
                command_log = (safe_dir / "cmd.log").read_text(encoding="utf-8")
            except OSError:
                failures.append("safe_run_logs_missing")
        try:
            resolved_mode, store_count = parse_attestation(command_log)
        except ValueError as exc:
            failures.append(str(exc))
            resolved_mode, store_count = 0, 0
        try:
            cases = parse_quality_tsv(result_path)
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
            cases = []
        try:
            server_instance = _server_instance(main_log)
        except ValueError as exc:
            failures.append(str(exc))
            server_instance = f"invalid-attempt-{index}"
        try:
            available_memory_gib = _minimum_available_gib(samples)
        except ValueError as exc:
            failures.append(str(exc))
            available_memory_gib = 0.0
        after = content_complete_fixture_sha256(source, manifests)
        if after != fixture_sha256:
            failures.append("fixture_bytes_changed_after_attempt")
        fault_text = "\n".join((kernel, main_log, command_log))
        oom = bool(re.search(r"oom-kill|Out of memory|memory event", fault_text, re.I))
        xid = bool(re.search(r"NVRM.*Xid|NV_ERR_NO_MEMORY", fault_text, re.I))
        if FAULT_RE.search(fault_text):
            failures.append("kernel_or_memory_fault")
        if "memory_swap_max=0" not in main_log:
            failures.append("zero_swap_cgroup_not_attested")
        attempt = {
            "block": block,
            "sequence": sequence,
            "arm": arm,
            "server_instance_id": server_instance,
            "binary_sha256": binary_sha256,
            "configuration_sha256": configuration_sha256,
            "fixture_sha256_before": before,
            "fixture_sha256_after": after,
            "environment_sha256": expected_environment_sha256,
            "resolved_mode": resolved_mode,
            "affine_store_count": store_count,
            "completed": not failures,
            "available_memory_gib": available_memory_gib,
            "swap_bytes": 0,
            "oom": oom,
            "xid": xid,
            "failures": failures,
            "cases": cases,
        }
        campaign["attempts"].append(attempt)
        _atomic_json(campaign_path, campaign)
        if failures:
            print(
                f"attempt {index:02d} failed: {', '.join(failures)}",
                file=sys.stderr,
            )
            return 1
        print(
            f"completed attempt {index + 1}/20 block={block} sequence={sequence} arm={arm}",
            flush=True,
        )

    summary = _score(campaign)
    _atomic_json(output / "summary.json", summary)
    (output / "raw.jsonl").write_text(
        json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0 if summary["verdict"] == "PASS" else 1


def status(args: argparse.Namespace) -> int:
    campaign = _strict_json(args.campaign / "campaign.json")
    summary_path = args.campaign / "summary.json"
    result = {
        "campaign": str(args.campaign),
        "completed_attempts": len(campaign.get("attempts", [])),
        "total_attempts": 20,
        "summary": _strict_json(summary_path) if summary_path.exists() else None,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--drand-json", required=True, type=Path)
    run_parser.add_argument("--engine-source", required=True, type=Path)
    run_parser.add_argument("--engine-candidate-hash")
    run_parser.add_argument("--model", required=True, type=Path)
    run_parser.add_argument("--model-sha256")
    run_parser.add_argument("--output", type=Path)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("campaign", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args) if args.command == "run" else status(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"glm52-w1-affine: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
