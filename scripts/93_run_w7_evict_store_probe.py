#!/usr/bin/env python3
"""Run the contained two-arm W7.2 preload-evict-store diagnostic probe."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import types
import uuid
from typing import Any


ROOT = Path("/home/bmarti44/spark-deepseek-v4-flash")
BASE_PATH = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
SCORER = ROOT / "scripts/92_score_w7_evict_store_probe.py"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run_w7_evict_store_v1.sh"
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
MEMORY_GUARD = ROOT / "scripts/03_memory_guard.py"
DRAND_VERIFIER = ROOT / "scripts/89_verify_drand_receipt.mjs"
DRAND_NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
BIN = Path("/home/bmarti44/.cache/glm52-w7-evict-store-fdd0fe0/ds4-server")
CANDIDATE_SRC = BIN.parent
MODEL = Path("/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf")
LIVE = Path("/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json")
PRIMARY = Path("/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json")
OUT_ROOT = Path("/home/bmarti44/.local/state/glm52-w7-evict-store-probe")
CAMPAIGN_LOCK = Path("/run/lock/frontier-at-home/inference.lock")

BINARY_SHA256 = "fbd889832abc5a04b44f16462ac7959bf77d73cc41283faa7ad6b5a02945130a"
BINARY_BYTES = 5_584_336
ENGINE_SOURCE_COMMIT = "fdd0fe05df6368de428cd5405929f49d590c9f52"
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
MODEL_BYTES = 211_075_856_448
LIVE_SHA256 = "d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a"
PRIMARY_SHA256 = "a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3"
BASE_SHA256 = "e2f6235cd5f94b67773e75cff0f4fbceaa264f5b88e3d12b45ae3bb1e31e6924"
CGROUP_SHA256 = "13fe1f8faa77020918faadc5214f4f5cdc955d6dfc6c158cc0f2d080a0e8985d"
SAFE_SHA256 = "2ddffb19f79b790c419db8ac53574d23ccf9f2c7699136fbaa55fc2a890b19e6"
MEMORY_GUARD_SHA256 = "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58"
SCORER_SHA256 = "1ccb1cf2d501cc5ec0132080fe7f60fed21a707bc741c28e03520a89d325fcbb"
DRAND_VERIFIER_SHA256 = "c191d301e1ff8460fffaea9dfeaab7d0fce0d63f92d3fdfcfa20442ccfdc2131"
DRAND_NODE_SHA256 = "3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90"
DRAND_FREEZE_FLOOR_ROUND = 6_358_539
FLAG = "DS4_KV_SKIP_PRELOAD_EVICT_STORE_DIAGNOSTIC"
ACTIVATION = "ds4-server: diagnostic preload evict-store bypass enabled"
SKIPPED = "ds4-server: diagnostic skipped preload evict store live=5055 prompt=5066 common=5045"
EXPECTED_STORE_RE = re.compile(
    r"ds4-server: kv cache stored tokens=5055 trimmed=0 reason=evict "
    r"key=token-text size=[0-9]+(?:\.[0-9]+)? MiB save=[0-9]+(?:\.[0-9]+)? ms$"
)
EXPECTED_HIT_RE = re.compile(
    r"ds4-server: kv cache hit text tokens=5044 text=15571 quant=2 key=token-text "
    r"load=[0-9]+(?:\.[0-9]+)? ms file=.*/([0-9a-f]{40})\.kv$"
)
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ACTIVE_ATTEMPT: Path | None = None
_ACTIVE_CANDIDATE: str | None = None
_ACTIVE_ROWS: list[dict[str, object]] = []


class ProbeError(RuntimeError):
    pass


def _load_base() -> Any:
    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
        raise ProbeError("frozen lifecycle module mismatch")
    spec = importlib.util.spec_from_file_location("w7_frozen_lifecycle", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ProbeError("cannot load frozen lifecycle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CGROUP = CGROUP
    module.CGROUP_SHA256 = CGROUP_SHA256
    module.SCORER = SCORER
    module.SCORER_SHA256 = SCORER_SHA256
    module.BIN = BIN
    module.BINARY_SHA256 = BINARY_SHA256
    module.ENGINE_SOURCE_COMMIT = ENGINE_SOURCE_COMMIT
    module.CANDIDATE_SRC = CANDIDATE_SRC
    module.OUT_ROOT = OUT_ROOT
    module.DRAND_FREEZE_FLOOR_ROUND = DRAND_FREEZE_FLOOR_ROUND
    return module


BASE = _load_base()


def git_bytes(candidate: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "show", f"{candidate}:{relative}"],
        capture_output=True, check=True,
    )
    return completed.stdout


def verify_candidate(candidate: str) -> None:
    if COMMIT_RE.fullmatch(candidate) is None:
        raise ProbeError("invalid candidate commit")
    head = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, check=True, text=True,
    ).stdout
    if head != candidate or dirty:
        raise ProbeError("candidate is not the clean executing HEAD")
    for relative, live_path in (
        ("scripts/93_run_w7_evict_store_probe.py", Path(__file__)),
        ("scripts/91_run_w7_cache_generation_campaign.py", BASE_PATH),
        ("scripts/92_score_w7_evict_store_probe.py", SCORER),
        ("results/glm52-gates/harness/glm_cgroup_run_w7_evict_store_v1.sh", CGROUP),
    ):
        if git_bytes(candidate, relative) != live_path.read_bytes():
            raise ProbeError(f"executing dependency differs from frozen candidate: {relative}")


def verify_dependencies() -> None:
    for path, digest in (
        (BASE_PATH, BASE_SHA256), (SCORER, SCORER_SHA256), (CGROUP, CGROUP_SHA256),
        (SAFE, SAFE_SHA256), (MEMORY_GUARD, MEMORY_GUARD_SHA256),
        (DRAND_VERIFIER, DRAND_VERIFIER_SHA256), (DRAND_NODE, DRAND_NODE_SHA256),
        (BIN, BINARY_SHA256), (LIVE, LIVE_SHA256), (PRIMARY, PRIMARY_SHA256),
    ):
        BASE.verify_file(path, digest)
    if BIN.stat().st_size != BINARY_BYTES:
        raise ProbeError("binary size mismatch")
    if MODEL.is_symlink() or not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES:
        raise ProbeError("model identity mismatch")


def derive_order(seed_sha256: str) -> list[str]:
    if SHA_RE.fullmatch(seed_sha256) is None:
        raise ValueError("seed must be lowercase SHA-256")
    bit = hashlib.sha256(b"W7-EVICT-STORE-PROBE-V1\0" + bytes.fromhex(seed_sha256)).digest()[0] & 1
    return ["off", "on"] if bit == 0 else ["on", "off"]


def environment_for_arm(
    arm: str, out: Path, request_sha256: str, engine_lock_path: str,
    lock_identity: str, campaign_lock_fd: int, memory_guard_path: str,
) -> tuple[dict[str, str], str]:
    env, _ = BASE.environment_for_arm(
        "off", out, request_sha256, engine_lock_path, lock_identity,
        campaign_lock_fd, memory_guard_path,
    )
    names = set(env["GLM_SAFE_PROVENANCE_ENV_ALLOWLIST"].split(","))
    if arm == "on":
        env[FLAG] = "1"
        names.add(FLAG)
    digest = hashlib.sha256(
        "".join(f"{name}={env[name]}\n" for name in sorted(names)).encode()
    ).hexdigest()
    env["GLM_SAFE_PROVENANCE_ENV_ALLOWLIST"] = ",".join(sorted(names))
    env["GLM_SAFE_EXPECTED_ENV_SHA256"] = digest
    return env, digest


def driver(
    arm: str, out: Path, request: Path, request_sha256: str, candidate: str,
    model_devino: str, model_descriptor: str, engine_lock: str,
) -> int:
    if arm not in {"off", "on"}:
        raise ProbeError("invalid arm")
    if os.environ.get(FLAG) != ("1" if arm == "on" else None):
        raise ProbeError("diagnostic environment/arm mismatch")
    BASE.verify_candidate = verify_candidate
    return BASE.driver(
        "off", out, request, request_sha256, candidate, model_devino,
        model_descriptor, engine_lock,
    )


def parse_arm(
    arm: str, position: int, out: Path, containment_rc: int,
    containment_stdout: str, request_sha256: str, config_sha256: str,
) -> tuple[dict[str, object], dict[str, str]]:
    BASE.BINARY_SHA256 = BINARY_SHA256
    BASE.MODEL_SHA256 = MODEL_SHA256
    parsed = BASE.parse_arm(
        "off", 0, position, out, containment_rc, containment_stdout,
        request_sha256, config_sha256,
    )
    done = BASE.DONE_RE.fullmatch(containment_stdout)
    if done is None:
        raise ProbeError("safe-run receipt missing during W7.2 binding")
    safety_main, _ = BASE.read_stable(out / "safety/main.log")
    if hashlib.sha256(safety_main).hexdigest() != done.group(2):
        raise ProbeError("W7.2 safety-main receipt mismatch")
    server_bytes, server_metadata = BASE.read_stable(out / "server.log")
    server_digest = hashlib.sha256(server_bytes).hexdigest()
    expected_server_binding = (
        f"final_artifact_verified path={out / 'server.log'} "
        f"sha256={server_digest} "
        f"device_inode={server_metadata.st_dev}:{server_metadata.st_ino}:{server_metadata.st_size}"
    ).encode()
    if safety_main.count(expected_server_binding) != 1:
        raise ProbeError("W7.2 server snapshot differs from receipt-bound artifact")
    server = server_bytes.decode("utf-8", errors="strict")
    logit_paths = sorted(
        out.glob("logits.sync*.start*.prompt*.suffix*"),
        key=lambda path: int(re.search(r"\.sync([0-9]+)\.", path.name).group(1)),
    )
    logit_records = []
    bound_artifacts = {str(out / "server.log"): server_digest}
    for path in logit_paths:
        payload, metadata = BASE.read_stable(path)
        digest = hashlib.sha256(payload).hexdigest()
        logit_records.append((path.name, digest, metadata.st_size))
        bound_artifacts[str(path)] = digest
    sequence_digest = hashlib.sha256(
        json.dumps(logit_records, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if (
        sequence_digest != parsed["logit_sequence_sha256"]
        or logit_records[-1][1] != parsed["final_logits_sha256"]
    ):
        raise ProbeError("W7.2 logit snapshot differs from frozen-parser evidence")
    generic_store = [
        line for line in server.splitlines()
        if "ds4-server: kv cache stored" in line and "reason=evict" in line
    ]
    generic_skip = [
        line for line in server.splitlines()
        if "ds4-server: diagnostic skipped preload evict store" in line
    ]
    generic_activation = [
        line for line in server.splitlines()
        if "ds4-server: diagnostic preload evict-store bypass" in line
    ]
    hit_lines = [line for line in server.splitlines() if "ds4-server: kv cache hit text " in line]
    hit_matches = [EXPECTED_HIT_RE.search(line) for line in hit_lines]
    if len(hit_matches) != 1 or hit_matches[0] is None:
        raise ProbeError("checkpoint hit cardinality or payload mismatch")
    if arm == "off":
        if (
            len(generic_store) != 1 or EXPECTED_STORE_RE.search(generic_store[0]) is None
            or generic_skip or generic_activation
        ):
            raise ProbeError("OFF evict-store event cardinality or payload mismatch")
    elif (
        generic_store or len(generic_skip) != 1 or generic_skip[0].endswith(SKIPPED) is False
        or len(generic_activation) != 1 or generic_activation[0].endswith(ACTIVATION) is False
    ):
        raise ProbeError("ON skip event cardinality or payload mismatch")
    safety = dict(parsed["safety"])
    safety.pop("false_generation_flushes")
    for name in ("block", "stable_remap", "final_logits_sha256", "logit_sequence_sha256"):
        parsed.pop(name)
    parsed.update({
        "arm": arm,
        "position": position,
        "diagnostic_skip": 1 if arm == "on" else 0,
        "logit_sha256s": [record[1] for record in logit_records],
        "selected_checkpoint_tokens": 5044,
        "checkpoint_id": f"token-text:{hit_matches[0].group(1)}",
        "evict_store_count": len(generic_store),
        "skip_marker_count": len(generic_skip),
        "activation_marker_count": len(generic_activation),
        "safety": safety,
    })
    return parsed, bound_artifacts


def load_scorer(frozen: bytes) -> Any:
    if hashlib.sha256(frozen).hexdigest() != SCORER_SHA256:
        raise ProbeError("retained scorer digest mismatch")
    module = types.ModuleType("w7_evict_store_probe_scorer")
    exec(compile(frozen, str(SCORER), "exec"), module.__dict__)
    return module


def collect_artifacts(
    attempt: Path, expected: dict[str, str] | None = None,
) -> dict[str, str]:
    """Hash regular attempt artifacts once and enforce authoritative snapshots."""
    expected = expected or {}
    artifacts: dict[str, str] = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_symlink():
            raise ProbeError(f"symlink in attempt evidence: {path}")
        if path.is_dir():
            continue
        if path.name in {"manifest.json", "raw.jsonl", "summary.json"}:
            continue
        payload, _ = BASE.read_stable(path)
        digest = hashlib.sha256(payload).hexdigest()
        if str(path) in expected and digest != expected[str(path)]:
            raise ProbeError(f"authoritative artifact changed before publication: {path}")
        artifacts[str(path.relative_to(attempt))] = digest
    if set(expected) - {str(attempt / name) for name in artifacts}:
        raise ProbeError("authoritative artifact disappeared before publication")
    return artifacts


def terminal_manifest_valid(attempt: Path, candidate: str) -> bool:
    try:
        payload, _ = BASE.read_stable(attempt / "manifest.json")
        manifest = json.loads(payload)
        if (
            not isinstance(manifest, dict) or manifest.get("candidate_hash") != candidate
            or manifest.get("verdict") not in {"PASS", "FAIL"}
            or not isinstance(manifest.get("artifacts"), dict)
            or not {"raw.jsonl", "summary.json"}.issubset(manifest["artifacts"])
        ):
            return False
        for relative, digest in manifest["artifacts"].items():
            if (
                not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(digest, str)
                or SHA_RE.fullmatch(digest) is None
            ):
                return False
            artifact, _ = BASE.read_stable(attempt / relative)
            if hashlib.sha256(artifact).hexdigest() != digest:
                return False
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def displace_existing(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    payload, _ = BASE.read_stable(path)
    for index in range(1, 100):
        target = path.with_name(f"{path.name}.pre-finalization-{index}")
        if not target.exists():
            os.rename(path, target)
            return target.name, hashlib.sha256(payload).hexdigest()
    raise ProbeError(f"too many displaced finalization artifacts: {path.name}")


def finalize_failure_triplet(error: BaseException) -> None:
    """Preserve a fail-closed triplet if execution escapes after attempt creation."""
    attempt = _ACTIVE_ATTEMPT or BASE._ACTIVE_ATTEMPT
    candidate = _ACTIVE_CANDIDATE or BASE._ACTIVE_CANDIDATE
    if attempt is None or candidate is None:
        return
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    manifest_path = attempt / "manifest.json"
    try:
        if manifest_path.exists() and terminal_manifest_valid(attempt, candidate):
            return
        displaced_manifest = displace_existing(manifest_path)
        displaced_raw = displace_existing(attempt / "raw.jsonl")
        displaced_summary = displace_existing(attempt / "summary.json")
        raw_bytes = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in _ACTIVE_ROWS
        )
        failure = f"{type(error).__name__}: {error}"
        summary_bytes = (json.dumps({"failure": failure, "verdict": "FAIL"}, sort_keys=True, separators=(",", ":")) + "\n").encode()
        artifacts = collect_artifacts(attempt)
        BASE.write_new(attempt / "raw.jsonl", raw_bytes)
        BASE.write_new(attempt / "summary.json", summary_bytes)
        artifacts["raw.jsonl"] = hashlib.sha256(raw_bytes).hexdigest()
        artifacts["summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
        for displaced in (displaced_manifest, displaced_raw, displaced_summary):
            if displaced is not None:
                artifacts[displaced[0]] = displaced[1]
        BASE.write_json_new(manifest_path, {
            "schema": "glm52-w7-evict-store-probe-failure-v1",
            "candidate_hash": candidate,
            "failure": failure,
            "runner_sha256": BASE.sha256_file(Path(__file__)),
            "scorer_sha256": SCORER_SHA256,
            "binary_sha256": BINARY_SHA256,
            "model_sha256": MODEL_SHA256,
            "completed_rows": len(_ACTIVE_ROWS),
            "artifacts": artifacts,
            "verdict": "FAIL",
        })
        if not terminal_manifest_valid(attempt, candidate):
            raise ProbeError("published failure triplet did not validate")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def run_probe(candidate: str, randomness_receipt: Path) -> int:
    global _ACTIVE_ATTEMPT, _ACTIVE_CANDIDATE, _ACTIVE_ROWS
    verify_dependencies()
    verify_candidate(candidate)
    if BASE.server_pids() or subprocess.run(["/usr/bin/pgrep", "-x", "fio"], capture_output=True).returncode == 0:
        raise ProbeError("engine or fio already active")
    available_kb = int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    ))
    if available_kb < 110 * 1024 * 1024:
        raise ProbeError("less than 110 GiB available")
    campaign_lock_fd = os.open(CAMPAIGN_LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(campaign_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(campaign_lock_fd)
        raise ProbeError("global inference lock is already held") from error
    memory_guard_fd = model_fd = engine_lock_fd = -1
    try:
        frozen_scorer = git_bytes(candidate, "scripts/92_score_w7_evict_store_probe.py")
        if frozen_scorer != BASE.read_stable(SCORER)[0] or hashlib.sha256(frozen_scorer).hexdigest() != SCORER_SHA256:
            raise ProbeError("frozen scorer identity mismatch")
        memory_guard_fd = os.open(MEMORY_GUARD, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        if BASE.sha256_descriptor(memory_guard_fd) != MEMORY_GUARD_SHA256:
            raise ProbeError("retained memory guard mismatch")
        memory_guard_path = f"/proc/{os.getpid()}/fd/{memory_guard_fd}"
        model_fd = os.open(MODEL, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        before = os.fstat(model_fd)
        if before.st_size != MODEL_BYTES or BASE.sha256_descriptor(model_fd) != MODEL_SHA256:
            raise ProbeError("model content identity mismatch")
        after = os.fstat(model_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ProbeError("model changed during identity scan")
        model_devino = f"{after.st_dev}:{after.st_ino}"
        model_descriptor = f"/proc/{os.getpid()}/fd/{model_fd}"
        seed, receipt_sha, receipt_bytes = BASE.verify_public_randomness_receipt(randomness_receipt, candidate)
        order = derive_order(seed)
        OUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            attempt = BASE.create_and_activate_attempt(OUT_ROOT, candidate, uuid.uuid4().hex)
            _ACTIVE_ATTEMPT = attempt
            _ACTIVE_CANDIDATE = candidate
            _ACTIVE_ROWS = []
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        engine_lock_fd = os.open(attempt / "engine.lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.unlink(attempt / "engine.lock")
        lock_stat = os.fstat(engine_lock_fd)
        engine_lock_path = f"/proc/{os.getpid()}/fd/{engine_lock_fd}"
        lock_identity = f"{lock_stat.st_dev}:{lock_stat.st_ino}"
        if shutil.disk_usage(attempt).free < 3 * 1024**3:
            raise ProbeError("less than 3 GiB free for probe evidence")
        BASE.write_new(attempt / "randomness-receipt.json", receipt_bytes)
        request_path = attempt / "primary-request.json"
        request_sha = BASE.make_primary_request(request_path)
        config = {
            "binary_sha256": BINARY_SHA256, "model_sha256": MODEL_SHA256,
            "context": 8192, "cache_gib": 40, "fetch_threads": 6,
            "boundary_align": 4, "boundary_trim": 8, "max_tokens": 160,
            "diagnostic_flag": FLAG,
            "containment": {"MemorySwapMax": 0, "kill_floor_GiB": 24, "minimum_start_GiB": 110, "timeout_s": 2400},
        }
        config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows: list[dict[str, object]] = []
        _ACTIVE_ROWS = rows
        authoritative_bindings: dict[str, str] = {}
        failure: str | None = None
        try:
            for position, arm in enumerate(order):
                verify_candidate(candidate)
                out = attempt / f"p{position}-{arm}-{uuid.uuid4().hex[:12]}"
                out.mkdir(mode=0o700)
                env, env_sha = environment_for_arm(
                    arm, out, request_sha, engine_lock_path, lock_identity,
                    campaign_lock_fd, memory_guard_path,
                )
                tag = f"w7e-p{position}-{uuid.uuid4().hex[:10]}"
                command = [
                    str(CGROUP), "--tag", tag, "--", "/usr/bin/python3", str(Path(__file__)),
                    "--driver", arm, str(out), str(request_path), request_sha, candidate,
                    model_devino, model_descriptor, engine_lock_path,
                ]
                completed = BASE.run_contained_command(command, env, tag)
                BASE.write_new(out / "containment.stdout", completed.stdout.encode())
                BASE.write_new(out / "containment.stderr", completed.stderr.encode())
                BASE.write_new(out / "containment.rc", f"{completed.returncode}\n".encode())
                row, arm_bindings = parse_arm(
                    arm, position, out, completed.returncode, completed.stdout,
                    request_sha, config_sha,
                )
                row["executed_environment_sha256"] = env_sha
                rows.append(row)
                authoritative_bindings.update(arm_bindings)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        scorer_rows = []
        for row in rows:
            scorer_row = dict(row)
            scorer_row.pop("executed_environment_sha256")
            scorer_rows.append(scorer_row)
        summary = load_scorer(frozen_scorer).score_probe_rows(scorer_rows, order)
        if failure is not None:
            summary = dict(summary, runtime_failure=failure, verdict="FAIL")
        raw_bytes = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in rows
        )
        summary_bytes = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
        manifest = {
            "schema": "glm52-w7-evict-store-probe-v1",
            "candidate_hash": candidate,
            "runner_sha256": BASE.sha256_file(Path(__file__)),
            "base_lifecycle_sha256": BASE_SHA256,
            "scorer_sha256": SCORER_SHA256,
            "cgroup_sha256": CGROUP_SHA256,
            "safe_run_sha256": SAFE_SHA256,
            "memory_guard_sha256": MEMORY_GUARD_SHA256,
            "binary_sha256": BINARY_SHA256,
            "engine_source_commit": ENGINE_SOURCE_COMMIT,
            "model_sha256": MODEL_SHA256,
            "model_bytes": MODEL_BYTES,
            "live_request_sha256": LIVE_SHA256,
            "primary_source_sha256": PRIMARY_SHA256,
            "executed_request_sha256": request_sha,
            "configuration": config,
            "configuration_sha256": config_sha,
            "public_randomness_sha256": seed,
            "public_randomness_receipt_sha256": receipt_sha,
            "arm_order": order,
            "completed_rows": len(rows),
            "artifacts": {},
        }
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            artifacts = collect_artifacts(attempt, authoritative_bindings)
            artifacts["raw.jsonl"] = hashlib.sha256(raw_bytes).hexdigest()
            artifacts["summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
            manifest["artifacts"] = artifacts
            BASE.write_new(attempt / "raw.jsonl", raw_bytes)
            BASE.write_new(attempt / "summary.json", summary_bytes)
            BASE.write_json_new(attempt / "manifest.json", manifest)
            directory_fd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if not terminal_manifest_valid(attempt, candidate):
                raise ProbeError("published terminal triplet did not validate")
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        print(f"W7_EVICT_STORE_PROBE_ATTEMPT={attempt}")
        _ACTIVE_ATTEMPT = None
        _ACTIVE_CANDIDATE = None
        _ACTIVE_ROWS = []
        BASE._ACTIVE_ATTEMPT = None
        BASE._ACTIVE_CANDIDATE = None
        return 0 if summary["verdict"] == "PASS" else 1
    finally:
        for descriptor in (engine_lock_fd, model_fd, memory_guard_fd):
            if descriptor >= 0:
                os.close(descriptor)
        fcntl.flock(campaign_lock_fd, fcntl.LOCK_UN)
        os.close(campaign_lock_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--randomness-receipt", type=Path)
    parser.add_argument("--driver", nargs=8)
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.candidate or args.randomness_receipt or args.driver:
                return 2
            verify_dependencies()
            if derive_order("a" * 64) == derive_order("b" * 64):
                raise ProbeError("order derivation self-test did not exercise both orders")
            print("W7_EVICT_STORE_PROBE_SELFTEST_OK")
            return 0
        if args.driver:
            if args.candidate or args.randomness_receipt:
                return 2
            arm, out, request, request_sha, candidate, devino, descriptor, lock = args.driver
            return driver(arm, Path(out), Path(request), request_sha, candidate, devino, descriptor, lock)
        if not args.candidate or args.randomness_receipt is None:
            parser.error("--candidate and --randomness-receipt are required")
        previous_handlers = BASE.install_campaign_signal_handlers()
        try:
            return run_probe(args.candidate, args.randomness_receipt)
        finally:
            BASE.restore_campaign_signal_handlers(previous_handlers)
    except Exception as error:
        try:
            finalize_failure_triplet(error)
        except Exception as finalization_error:
            print(
                f"W7_EVICT_STORE_PROBE_FINALIZATION_FAIL: {finalization_error}",
                file=sys.stderr,
            )
        print(f"W7_EVICT_STORE_PROBE_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
