#!/usr/bin/env python3
"""Run the contained, fresh-server W7.1 stable-remap OFF/ON campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import stat
import subprocess
import sys
import time
import types
import urllib.request
import uuid
from typing import Any


ROOT = Path("/home/bmarti44/spark-deepseek-v4-flash")
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
SCORER = ROOT / "scripts/90_score_w7_cache_generation_campaign.py"
MEMORY_GUARD = ROOT / "scripts/03_memory_guard.py"
DRAND_VERIFIER = ROOT / "scripts/89_verify_drand_receipt.mjs"
DRAND_NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
BIN = Path("/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6/ds4-server")
CANDIDATE_SRC = BIN.parent
MODEL = Path("/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf")
LIVE = Path("/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json")
PRIMARY = Path("/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json")
OUT_ROOT = Path("/home/bmarti44/.local/state/glm52-w7-cache-generation-campaign")
CAMPAIGN_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
PORT = 8097

BINARY_SHA256 = "eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2"
ENGINE_SOURCE_COMMIT = "bccf0b66e196e88213add7e0f81571bb7e558670"
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
MODEL_BYTES = 211075856448
LOGIT_BYTES = 154880 * 4
LIVE_SHA256 = "d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a"
PRIMARY_SHA256 = "a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3"
CGROUP_SHA256 = "a0cdae4fbd78e770ef971c65eaf95a38917de0150c8c3452a3876d9e894793fb"
SAFE_SHA256 = "2ddffb19f79b790c419db8ac53574d23ccf9f2c7699136fbaa55fc2a890b19e6"
MEMORY_GUARD_SHA256 = "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58"
SCORER_SHA256 = "721108911ce3bdc7bcae722605603e517ed4b07cfb9aa8142152860caf16ce5e"
DRAND_VERIFIER_SHA256 = "c191d301e1ff8460fffaea9dfeaab7d0fce0d63f92d3fdfcfa20442ccfdc2131"
DRAND_NODE_SHA256 = "3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90"
DRAND_FREEZE_FLOOR_ROUND = 6358059
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
TIMING_RE = re.compile(
    r"^DS4_TOKEN_TIMING request=(\S+) index=(\d+) monotonic_ns=(\d+) token=(-?\d+)$"
)
DONE_RE = re.compile(
    r"SAFE_RUN_DONE rc=0 killed=no dir=(/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+) "
    r"main_sha256=([0-9a-f]{64}) samples_sha256=([0-9a-f]{64}) kernel_sha256=([0-9a-f]{64})\s*\Z"
)
FALSE_FLUSH = "CUDA persistent expert cache flushed (model load generation changed)"
LISTENER = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"
_ACTIVE_ATTEMPT: Path | None = None
_ACTIVE_CANDIDATE: str | None = None
_ACTIVE_CONTAINMENT: subprocess.Popen[str] | None = None


class CampaignError(RuntimeError):
    pass


class CampaignInterrupted(CampaignError):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"campaign interrupted by signal {signum}")


def _raise_campaign_interrupt(signum: int, _frame: object) -> None:
    raise CampaignInterrupted(signum)


def install_campaign_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for selected in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[selected] = signal.getsignal(selected)
        signal.signal(selected, _raise_campaign_interrupt)
    return previous


def restore_campaign_signal_handlers(previous: dict[int, Any]) -> None:
    for selected, handler in previous.items():
        signal.signal(selected, handler)


def _terminate_and_reap(process: subprocess.Popen[str]) -> None:
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def containment_unit_name(tag: str, launcher_pid: int) -> str:
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", tag) is None
        or type(launcher_pid) is not int or launcher_pid <= 1
    ):
        raise CampaignError("invalid containment unit identity")
    return f"glm52-{tag.replace('.', '-')}-{launcher_pid}.service"


def _unit_state(unit: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "/usr/bin/systemctl", "--user", "show", unit, "--no-pager",
            "--property=LoadState", "--property=ActiveState", "--property=SubState",
            "--property=MainPID", "--property=ControlPID",
        ],
        capture_output=True, text=True, check=False, timeout=15,
    )
    if completed.returncode != 0:
        return {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0", "ControlPID": "0"}
    state: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            state[name] = value
    return state


def _unit_is_stopped(unit: str) -> bool:
    state = _unit_state(unit)
    return (
        state.get("LoadState") in {"not-found", "masked"}
        or (
            state.get("ActiveState") in {"inactive", "failed"}
            and state.get("MainPID", "0") == "0"
            and state.get("ControlPID", "0") == "0"
        )
    )


def stop_exact_containment_unit(unit: str) -> None:
    if re.fullmatch(r"glm52-[A-Za-z0-9][A-Za-z0-9._-]{0,80}-[1-9][0-9]*\.service", unit) is None:
        raise CampaignError("refusing untrusted containment unit")
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "stop", unit],
        capture_output=True, text=True, check=False, timeout=60,
    )
    for _ in range(120):
        if _unit_is_stopped(unit):
            return
        time.sleep(0.25)
    subprocess.run(
        [
            "/usr/bin/systemctl", "--user", "kill", "--kill-whom=all",
            "--signal=SIGKILL", unit,
        ],
        capture_output=True, text=True, check=False, timeout=30,
    )
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "stop", unit],
        capture_output=True, text=True, check=False, timeout=30,
    )
    for _ in range(40):
        if _unit_is_stopped(unit):
            return
        time.sleep(0.25)
    raise CampaignError(f"containment unit did not stop: {unit}")


def _listener_is_active() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", PORT)) == 0


def _cleanup_interrupted_containment(process: subprocess.Popen[str], unit: str) -> None:
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        try:
            stop_exact_containment_unit(unit)
        finally:
            _terminate_and_reap(process)
        if server_pids() or _listener_is_active():
            raise CampaignError("containment cleanup left a server or listener")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def run_contained_command(
    command: list[str], environment: dict[str, str], tag: str,
) -> subprocess.CompletedProcess[str]:
    """Run one launcher and synchronously reap it before propagating interruption."""
    global _ACTIVE_CONTAINMENT
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        _ACTIVE_CONTAINMENT = process
        unit = containment_unit_name(tag, process.pid)
    finally:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        except BaseException:
            if process is not None:
                _cleanup_interrupted_containment(process, containment_unit_name(tag, process.pid))
            _ACTIVE_CONTAINMENT = None
            raise
    try:
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        _cleanup_interrupted_containment(process, unit)
        raise
    finally:
        _ACTIVE_CONTAINMENT = None


def finalize_failure_triplet(error: BaseException) -> None:
    """Fail closed if an exception escapes after an attempt directory exists."""
    if _ACTIVE_ATTEMPT is None or _ACTIVE_CANDIDATE is None:
        return
    raw_path = _ACTIVE_ATTEMPT / "raw.jsonl"
    summary_path = _ACTIVE_ATTEMPT / "summary.json"
    manifest_path = _ACTIVE_ATTEMPT / "manifest.json"
    if manifest_path.exists():
        return
    failure = f"{type(error).__name__}: {error}"
    raw_bytes = raw_path.read_bytes() if raw_path.is_file() else b""
    summary_bytes = (
        json.dumps({"failure": failure, "verdict": "FAIL"}, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    if not raw_path.exists():
        write_new(raw_path, raw_bytes)
    displaced_summary: tuple[str, str] | None = None
    if summary_path.exists():
        prior_bytes, _ = read_stable(summary_path)
        prior_path = _ACTIVE_ATTEMPT / "summary.pre-finalization.json"
        os.rename(summary_path, prior_path)
        displaced_summary = (prior_path.name, hashlib.sha256(prior_bytes).hexdigest())
    write_new(summary_path, summary_bytes)
    manifest = {
        "schema": "glm52-w7-cache-generation-campaign-failure-v1",
        "candidate_hash": _ACTIVE_CANDIDATE,
        "failure": failure,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": SCORER_SHA256,
        "binary_sha256": BINARY_SHA256,
        "model_sha256": MODEL_SHA256,
        "live_request_sha256": LIVE_SHA256,
        "primary_source_sha256": PRIMARY_SHA256,
        "configuration": "unavailable_or_incomplete",
        "public_randomness_receipt_sha256": (
            sha256_file(_ACTIVE_ATTEMPT / "randomness-receipt.json")
            if (_ACTIVE_ATTEMPT / "randomness-receipt.json").is_file() else None
        ),
        "artifacts": {
            "raw.jsonl": hashlib.sha256(raw_bytes).hexdigest(),
            "summary.json": hashlib.sha256(summary_bytes).hexdigest(),
        },
        "verdict": "FAIL",
    }
    if displaced_summary is not None:
        manifest["artifacts"][displaced_summary[0]] = displaced_summary[1]
    write_json_new(manifest_path, manifest)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 16 * 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def process_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
    if len(fields) < 22:
        raise CampaignError("cannot bind campaign process identity")
    return int(fields[21])


def lock_kernel_key(metadata: os.stat_result) -> str:
    return f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}:{metadata.st_ino}"


def create_and_activate_attempt(parent: Path, candidate: str, nonce: str) -> Path:
    """Create an attempt while deferring termination signals until it is tracked."""
    global _ACTIVE_ATTEMPT, _ACTIVE_CANDIDATE
    if COMMIT_RE.fullmatch(candidate) is None or re.fullmatch(r"[A-Za-z0-9._-]+", nonce) is None:
        raise CampaignError("invalid attempt activation binding")
    attempt = parent / f"attempt-{nonce}"
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        attempt.mkdir(mode=0o700)
        _ACTIVE_ATTEMPT = attempt
        _ACTIVE_CANDIDATE = candidate
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    return attempt


def read_stable(path: Path) -> tuple[bytes, os.stat_result]:
    """Read one regular file once, through one no-follow descriptor."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CampaignError(f"unsafe evidence file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or sum(map(len, chunks)) != before.st_size:
            raise CampaignError(f"evidence changed while read: {path}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def write_json_new(path: Path, value: object) -> None:
    write_new(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def verify_file(path: Path, digest: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
        raise CampaignError(f"dependency mismatch: {path}")


def verify_dependencies() -> None:
    verify_file(BIN, BINARY_SHA256)
    verify_file(LIVE, LIVE_SHA256)
    verify_file(PRIMARY, PRIMARY_SHA256)
    verify_file(CGROUP, CGROUP_SHA256)
    verify_file(SAFE, SAFE_SHA256)
    verify_file(MEMORY_GUARD, MEMORY_GUARD_SHA256)
    verify_file(SCORER, SCORER_SHA256)
    verify_file(DRAND_VERIFIER, DRAND_VERIFIER_SHA256)
    verify_file(DRAND_NODE, DRAND_NODE_SHA256)
    if MODEL.is_symlink() or not MODEL.is_file() or MODEL.stat().st_size != MODEL_BYTES:
        raise CampaignError("model identity mismatch")


def derive_schedules(seed_sha256: str) -> list[str]:
    if SHA_RE.fullmatch(seed_sha256) is None:
        raise ValueError("seed must be lowercase SHA-256")
    seed = bytes.fromhex(seed_sha256)
    domain = b"W7-CACHE-GENERATION-SCHEDULE-V1\0"
    return [
        "ABBA" if hashlib.sha256(domain + seed + bytes([block])).digest()[0] & 1 == 0 else "BAAB"
        for block in range(5)
    ]


def verify_public_randomness_receipt(path: Path, candidate: str) -> tuple[str, str, bytes]:
    if not path.is_file() or path.is_symlink():
        raise CampaignError("public randomness receipt is not a regular file")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("invalid public randomness receipt JSON") from error
    required = {
        "round", "freeze_floor_round", "randomness", "signature", "previous_signature",
        "frozen_gate_commit", "relay_agreement",
    }
    if not isinstance(doc, dict) or not required.issubset(doc):
        raise CampaignError("public randomness receipt fields missing")
    if (
        type(doc["round"]) is not int or type(doc["freeze_floor_round"]) is not int
        or doc["freeze_floor_round"] != DRAND_FREEZE_FLOOR_ROUND
        or doc["round"] <= DRAND_FREEZE_FLOOR_ROUND
        or doc["frozen_gate_commit"] != candidate
        or doc["relay_agreement"] != ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"]
        or SHA_RE.fullmatch(doc["randomness"]) is None
        or re.fullmatch(r"[0-9a-f]{192}", doc["signature"] or "") is None
        or re.fullmatch(r"[0-9a-f]{192}", doc["previous_signature"] or "") is None
    ):
        raise CampaignError("public randomness receipt is not post-freeze and three-relay bound")
    completed = subprocess.run(
        [
            str(DRAND_NODE), str(DRAND_VERIFIER), str(doc["round"]), doc["randomness"],
            doc["signature"], doc["previous_signature"],
        ],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or completed.stdout != "DRAND_BLS_RECEIPT_OK\n":
        raise CampaignError("public randomness BLS verification failed")
    return doc["randomness"], hashlib.sha256(raw).hexdigest(), raw


def git_bytes(candidate: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "show", f"{candidate}:{relative}"],
        check=True, capture_output=True,
    )
    return completed.stdout


def verify_candidate(candidate: str) -> None:
    if COMMIT_RE.fullmatch(candidate) is None:
        raise CampaignError("invalid candidate commit")
    head = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    if head != candidate or dirty:
        raise CampaignError("candidate is not the clean executing HEAD")
    relative = "scripts/91_run_w7_cache_generation_campaign.py"
    if git_bytes(candidate, relative) != Path(__file__).read_bytes():
        raise CampaignError("executing runner differs from frozen candidate")


def server_pids() -> list[int]:
    completed = subprocess.run(
        ["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True, check=False,
    )
    return [int(value) for value in completed.stdout.split()]


def make_primary_request(output: Path) -> str:
    source = json.loads(PRIMARY.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("max_tokens") != 0:
        raise CampaignError("unexpected primary fixture")
    source["max_tokens"] = 160
    source["stream"] = True
    source["stream_options"] = {"include_usage": True}
    source["ignore_eos"] = True
    encoded = (json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n").encode()
    write_new(output, encoded)
    return hashlib.sha256(encoded).hexdigest()


def wait_ready(process: subprocess.Popen[bytes]) -> None:
    for _ in range(600):
        if process.poll() is not None:
            raise CampaignError("server died before listener readiness")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise CampaignError("listener readiness timeout")


def post_json(path: Path, *, stream: bool) -> tuple[bytes, dict[str, Any] | None]:
    body = path.read_bytes()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=1800) as response:
        if response.status != 200:
            raise CampaignError(f"request returned HTTP {response.status}")
        if not stream:
            return response.read(), None
        response_ids: set[str] = set()
        text_parts: list[str] = []
        finish_reasons: list[str] = []
        usage: dict[str, Any] | None = None
        done = False
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:") or done:
                raise CampaignError("malformed SSE stream")
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            if not isinstance(event, dict):
                raise CampaignError("SSE event is not an object")
            response_id = event.get("id")
            if response_id is not None:
                if not isinstance(response_id, str) or not response_id:
                    raise CampaignError("invalid SSE response id")
                response_ids.add(response_id)
            event_usage = event.get("usage")
            if event_usage is not None:
                if usage is not None or not isinstance(event_usage, dict):
                    raise CampaignError("invalid duplicate SSE usage")
                usage = event_usage
            choices = event.get("choices", [])
            if not isinstance(choices, list):
                raise CampaignError("invalid SSE choices")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise CampaignError("invalid SSE choice")
                fragment = choice.get("text", "")
                if not isinstance(fragment, str):
                    raise CampaignError("invalid SSE text")
                text_parts.append(fragment)
                reason = choice.get("finish_reason")
                if reason is not None:
                    if not isinstance(reason, str):
                        raise CampaignError("invalid finish reason")
                    finish_reasons.append(reason)
        if not done or len(response_ids) != 1 or len(finish_reasons) != 1 or usage is None:
            raise CampaignError("incomplete SSE stream")
        return b"", {
            "request_start_ns": started,
            "response_id": next(iter(response_ids)),
            "generated_text": "".join(text_parts),
            "finish_reason": finish_reasons[0],
            "usage": usage,
            "done": done,
        }


def validate_warm_response(raw: bytes) -> None:
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("invalid warm response") from error
    usage = doc.get("usage") if isinstance(doc, dict) else None
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    choices = doc.get("choices") if isinstance(doc, dict) else None
    if (
        not isinstance(choices, list) or len(choices) != 1
        or not isinstance(choices[0], dict) or choices[0].get("finish_reason") != "length"
        or choices[0].get("text") != "" or not isinstance(usage, dict)
        or usage.get("prompt_tokens") != 5055 or usage.get("completion_tokens") != 0
        or usage.get("total_tokens") != 5055 or not isinstance(details, dict)
        or details.get("cached_tokens") != 0 or details.get("cache_write_tokens") != 5055
    ):
        raise CampaignError("warm response does not bind the 5,055-token cache population")


def remove_kv_scratch(kv: Path) -> None:
    if not kv.is_dir() or kv.is_symlink():
        raise CampaignError("invalid KV scratch directory")
    for child in kv.iterdir():
        if child.is_symlink() or not child.is_file() or re.fullmatch(r"[0-9a-f]{40}\.kv", child.name) is None:
            raise CampaignError("unexpected KV scratch artifact")
        child.unlink()
    kv.rmdir()


def stop_server(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        return process.wait(timeout=45)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait(timeout=15)
        raise CampaignError("server required SIGKILL") from error


def driver(
    arm: str, out: Path, request_path: Path, request_sha256: str, candidate: str,
    expected_model_devino: str, model_descriptor_path: str, engine_lock_path: str,
) -> int:
    if arm not in {"off", "on"}:
        raise CampaignError("invalid driver arm")
    verify_dependencies()
    verify_candidate(candidate)
    if server_pids():
        raise CampaignError("pre-existing ds4-server")
    if not out.is_dir() or out.is_symlink() or any(out.iterdir()):
        raise CampaignError("driver output is not a fresh empty directory")
    if SHA_RE.fullmatch(request_sha256) is None or sha256_file(request_path) != request_sha256:
        raise CampaignError("request digest mismatch")
    stable = os.environ.get("DS4_CUDA_STABLE_MODEL_REMAP")
    if stable != ("1" if arm == "on" else None):
        raise CampaignError("stable-remap environment/arm mismatch")

    if re.fullmatch(r"/proc/[1-9][0-9]*/fd/[0-9]+", model_descriptor_path) is None:
        raise CampaignError("invalid retained model descriptor path")
    model_fd = os.open(model_descriptor_path, os.O_RDONLY | os.O_CLOEXEC)
    model_stat = os.fstat(model_fd)
    model_devino = f"{model_stat.st_dev}:{model_stat.st_ino}"
    if model_devino != expected_model_devino or model_stat.st_size != MODEL_BYTES:
        os.close(model_fd)
        raise CampaignError("model descriptor identity mismatch")
    try:
        lock_stat = os.stat(engine_lock_path)
    except OSError as error:
        os.close(model_fd)
        raise CampaignError("retained engine lock unavailable") from error
    if not stat.S_ISREG(lock_stat.st_mode):
        os.close(model_fd)
        raise CampaignError("retained engine lock is not regular")

    kv = out / "kv"
    kv.mkdir(mode=0o700)
    server_log = out / "server.log"
    command = [
        str(BIN), "--cuda", "-m", f"/proc/self/fd/{model_fd}", "-c", "8192", "--host", "127.0.0.1",
        "--port", str(PORT), "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
        "--kv-disk-dir", str(kv), "--kv-disk-space-mb", "4096",
        "--kv-cache-boundary-align-tokens", "4", "--kv-cache-boundary-trim-tokens", "8",
    ]
    process: subprocess.Popen[bytes] | None = None
    with server_log.open("xb", buffering=0) as log:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(model_fd,)
            )
            wait_ready(process)
            live_response, _ = post_json(LIVE, stream=False)
            validate_warm_response(live_response)
            write_new(out / "live-response.json", live_response)
            _, client = post_json(request_path, stream=True)
            if client is None:
                raise CampaignError("missing streaming client evidence")
            write_json_new(out / "primary-client.json", client)
            exit_status = stop_server(process)
            process = None
            if exit_status != 0:
                raise CampaignError(f"server exited {exit_status}")
            write_json_new(
                out / "child-exit.json",
                {"shutdown_requested": True, "forced_kill": False, "exit_status": 0},
            )
            remove_kv_scratch(kv)
        finally:
            if process is not None:
                try:
                    stop_server(process)
                except Exception:
                    pass
    os.close(model_fd)
    os.sync()
    return 0


def environment_for_arm(
    arm: str, out: Path, request_sha256: str,
    engine_lock_path: str = "/proc/self/fd/999", lock_identity: str = "0:0",
    campaign_lock_fd: int | None = None,
    memory_guard_path: str = str(MEMORY_GUARD),
) -> tuple[dict[str, str], str]:
    measured = {
        "DS4_CUDA_EXPERT_CACHE_GB": "40",
        "DS4_CUDA_EXPERT_CACHE_PIN": "1",
        "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
        "DS4_CUDA_FETCH_THREADS": "6",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
        "DS4_GLM_LOGIT_DUMP": str(out / "logits"),
        "DS4_GLM_LOGIT_DUMP_ALL": "1",
        "DS4_GLM_SYNC_TRACE": "1",
        "DS4_LOCK_EXPECTED_DEV_INO": lock_identity,
        "DS4_LOCK_FILE": engine_lock_path,
        "DS4_TOKEN_TIMING_LOG": "1",
    }
    if arm == "on":
        measured["DS4_CUDA_STABLE_MODEL_REMAP"] = "1"
    digest = hashlib.sha256(
        "".join(f"{name}={measured[name]}\n" for name in sorted(measured)).encode()
    ).hexdigest()
    env = {
        "HOME": "/home/bmarti44",
        "USER": "bmarti44",
        "LOGNAME": "bmarti44",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
    }
    env.update(measured)
    env.update({
        "GLM_CANDIDATE_SRC": str(CANDIDATE_SRC),
        "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
        "GLM_SAFE_KILL_FLOOR_GIB": "24",
        "GLM_SAFE_MIN_START_GIB": "110",
        "GLM_SAFE_TIMEOUT_S": "2400",
        "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
        "GLM_SAFE_EXPECTED_BINARY_SHA256": BINARY_SHA256,
        "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(sorted(measured)),
        "GLM_SAFE_EXPECTED_ENV_SHA256": digest,
        "GLM_SAFE_MEMORY_GUARD_PATH": memory_guard_path,
        "GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256": MEMORY_GUARD_SHA256,
        "GLM_SAFE_FINAL_ARTIFACTS": ",".join(str(out / name) for name in (
            "server.log", "live-response.json", "primary-client.json", "child-exit.json"
        )),
        "GLM_SAFE_DONE_DIGESTS": "1",
    })
    if campaign_lock_fd is not None:
        campaign_lock_stat = os.fstat(campaign_lock_fd)
        env.update({
            "GLM_SAFE_PARENT_LOCK_PID": str(os.getpid()),
            "GLM_SAFE_PARENT_LOCK_START_TICKS": str(process_start_ticks(os.getpid())),
            "GLM_SAFE_PARENT_LOCK_FD": str(campaign_lock_fd),
            "GLM_SAFE_PARENT_LOCK_DEV_INO": f"{campaign_lock_stat.st_dev}:{campaign_lock_stat.st_ino}",
            "GLM_SAFE_PARENT_LOCK_KERNEL_KEY": lock_kernel_key(campaign_lock_stat),
        })
    return env, digest


def parse_arm(
    arm: str, block: int, position: int, out: Path, containment_rc: int,
    containment_stdout: str, request_sha256: str, config_sha256: str,
) -> dict[str, object]:
    if containment_rc != 0:
        raise CampaignError(f"containment failed rc={containment_rc}")
    done = DONE_RE.fullmatch(containment_stdout)
    if done is None:
        raise CampaignError("safe-run receipt missing or malformed")
    crash = Path(done.group(1))
    safety_dir = out / "safety"
    safety_dir.mkdir(mode=0o700)
    safety_bytes: dict[str, bytes] = {}
    for offset, name in enumerate(("main.log", "samples.log", "kernel.log"), start=2):
        payload, _ = read_stable(crash / name)
        if hashlib.sha256(payload).hexdigest() != done.group(offset):
            raise CampaignError(f"safe-run receipt digest mismatch: {name}")
        write_new(safety_dir / name, payload)
        safety_bytes[name] = payload
    try:
        main = safety_bytes["main.log"].decode("utf-8", errors="strict")
        samples = safety_bytes["samples.log"].decode("utf-8", errors="strict")
        kernel = safety_bytes["kernel.log"].decode("utf-8", errors="strict")
        arm_artifacts: dict[str, tuple[bytes, os.stat_result]] = {}
        for name in ("server.log", "live-response.json", "primary-client.json", "child-exit.json"):
            arm_artifacts[name] = read_stable(out / name)
        server_bytes, _ = arm_artifacts["server.log"]
        client_bytes, _ = arm_artifacts["primary-client.json"]
        server = server_bytes.decode("utf-8", errors="strict")
        client = json.loads(client_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("invalid UTF-8 or JSON arm evidence") from error
    for name, (payload, metadata) in arm_artifacts.items():
        expected = (
            f"final_artifact_verified path={out / name} "
            f"sha256={hashlib.sha256(payload).hexdigest()} "
            f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}"
        )
        if main.count(expected) != 1:
            raise CampaignError(f"safe-run final artifact binding mismatch: {name}")

    timings: list[tuple[int, int, int]] = []
    for line in server.splitlines():
        if match := TIMING_RE.fullmatch(line):
            if match.group(1) == client["response_id"]:
                timings.append((int(match.group(2)), int(match.group(3)), int(match.group(4))))
    if len(timings) < 128 or [item[0] for item in timings] != list(range(1, len(timings) + 1)):
        raise CampaignError("missing or non-contiguous raw token timing")

    logit_paths = sorted(
        out.glob("logits.sync*.start*.prompt*.suffix*"),
        key=lambda path: int(re.search(r"\.sync([0-9]+)\.", path.name).group(1)),
    )
    if len(logit_paths) != len(timings) + 1:
        raise CampaignError("incomplete full-logit sequence")
    logit_records = []
    for path in logit_paths:
        payload, metadata = read_stable(path)
        logit_records.append((path.name, hashlib.sha256(payload).hexdigest(), metadata.st_size))
    logit_indices = [int(re.search(r"\.sync([0-9]+)\.", name).group(1)) for name, _, _ in logit_records]
    if logit_indices != list(range(1, len(logit_records) + 1)) or any(
        size != LOGIT_BYTES for _, _, size in logit_records
    ):
        raise CampaignError("logit sequence is non-contiguous or has the wrong tensor size")
    sequence_bytes = json.dumps(logit_records, separators=(",", ":")).encode("ascii")

    memory_values = [int(value) for value in re.findall(r"\bmem_avail_kb=([0-9]+)\b", samples)]
    swap_values = [int(value) for value in re.findall(r"\bcgroup_swap_current_bytes=([0-9]+)\b", samples)]
    final = re.findall(
        r"cgroup_final .* swap_current_bytes=([0-9]+) events=low [0-9]+,high [0-9]+,"
        r"max ([0-9]+),oom ([0-9]+),oom_kill ([0-9]+),oom_group_kill ([0-9]+),",
        main,
    )
    if not memory_values or not swap_values or len(final) != 1:
        raise CampaignError("incomplete safety counters")
    listener = server.find(LISTENER)
    shutdown = server.rfind(SHUTDOWN)
    if listener < 0 or shutdown <= listener:
        raise CampaignError("listener/shutdown window missing")
    false_flushes = server[listener:shutdown].count(FALSE_FLUSH)
    max_delta, oom_delta, oom_kill_delta, oom_group = map(int, final[0][1:])
    if oom_group != 0:
        raise CampaignError("cgroup OOM group kill recorded")
    usage = client.get("usage")
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    if (
        client.get("done") is not True or client.get("finish_reason") != "length"
        or not isinstance(usage, dict) or usage.get("completion_tokens") != len(timings)
        or usage.get("prompt_tokens") != 5066 or usage.get("total_tokens") != 5066 + len(timings)
        or not isinstance(details, dict) or details.get("cached_tokens") != 5044
        or details.get("cache_write_tokens") != 22
    ):
        raise CampaignError("stream usage does not bind the intended 22-token append")
    token_ids = [item[2] for item in timings]
    output_sha = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("ascii")).hexdigest()
    generated_text = client.get("generated_text")
    if not isinstance(generated_text, str):
        raise CampaignError("streaming response lacks generated text")
    generated_text_bytes = generated_text.encode("utf-8", errors="strict")
    return {
        "block": block,
        "position": position,
        "arm": arm,
        "run_id": out.name,
        "binary_sha256": BINARY_SHA256,
        "model_sha256": MODEL_SHA256,
        "common_config_sha256": config_sha256,
        "request_sha256": request_sha256,
        "stable_remap": 1 if arm == "on" else 0,
        "request_start_ns": client["request_start_ns"],
        "token_timestamps_ns": [item[1] for item in timings],
        "output_token_ids": token_ids,
        "output_sha256": output_sha,
        "generated_text_sha256": hashlib.sha256(generated_text_bytes).hexdigest(),
        "generated_text_bytes": len(generated_text_bytes),
        "final_logits_sha256": logit_records[-1][1],
        "logit_sequence_sha256": hashlib.sha256(sequence_bytes).hexdigest(),
        "server_fresh": server.count(LISTENER) == 1,
        "safety": {
            "containment_rc": containment_rc,
            "minimum_mem_available_kb": min(memory_values),
            "swap_growth_bytes": max(swap_values + [int(final[0][0])]),
            "cgroup_max_delta": max_delta,
            "cgroup_oom_delta": oom_delta,
            "cgroup_oom_kill_delta": oom_kill_delta,
            "xid_count": len(re.findall(r"\bXid\b", kernel, flags=re.IGNORECASE)),
            "surviving_descendants": len(server_pids()),
            "false_generation_flushes": false_flushes,
        },
    }


def load_scorer(frozen_scorer_bytes: bytes, expected_sha256: str) -> Any:
    if hashlib.sha256(frozen_scorer_bytes).hexdigest() != expected_sha256:
        raise CampaignError("retained scorer digest mismatch")
    module = types.ModuleType("w7_cache_campaign_scorer")
    try:
        exec(compile(frozen_scorer_bytes, str(SCORER), "exec"), module.__dict__)
    except Exception as error:
        raise CampaignError("cannot load retained scorer") from error
    return module


def campaign(candidate: str, randomness_receipt: Path) -> int:
    global _ACTIVE_ATTEMPT, _ACTIVE_CANDIDATE
    previous_signal_handlers = install_campaign_signal_handlers()
    verify_dependencies()
    verify_candidate(candidate)
    if server_pids() or subprocess.run(["/usr/bin/pgrep", "-x", "fio"], capture_output=True).returncode == 0:
        raise CampaignError("engine or fio already active")
    available_kb = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
    if available_kb < 110 * 1024 * 1024:
        raise CampaignError("less than 110 GiB available")
    campaign_lock_fd = os.open(CAMPAIGN_LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(campaign_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(campaign_lock_fd)
        raise CampaignError("global inference lock is already held") from error
    frozen_scorer_bytes = git_bytes(candidate, "scripts/90_score_w7_cache_generation_campaign.py")
    if (
        hashlib.sha256(frozen_scorer_bytes).hexdigest() != SCORER_SHA256
        or frozen_scorer_bytes != read_stable(SCORER)[0]
    ):
        raise CampaignError("frozen scorer identity mismatch")
    memory_guard_fd = os.open(MEMORY_GUARD, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    if sha256_descriptor(memory_guard_fd) != MEMORY_GUARD_SHA256:
        raise CampaignError("retained memory guard identity mismatch")
    memory_guard_path = f"/proc/{os.getpid()}/fd/{memory_guard_fd}"
    model_fd = os.open(MODEL, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    model_before = os.fstat(model_fd)
    if model_before.st_size != MODEL_BYTES or sha256_descriptor(model_fd) != MODEL_SHA256:
        raise CampaignError("model content identity mismatch")
    model_after = os.fstat(model_fd)
    if (model_before.st_dev, model_before.st_ino, model_before.st_size, model_before.st_mtime_ns) != (
        model_after.st_dev, model_after.st_ino, model_after.st_size, model_after.st_mtime_ns
    ):
        raise CampaignError("model changed during identity scan")
    model_devino = f"{model_after.st_dev}:{model_after.st_ino}"
    model_descriptor_path = f"/proc/{os.getpid()}/fd/{model_fd}"
    seed_sha256, randomness_receipt_sha256, randomness_receipt_bytes = verify_public_randomness_receipt(
        randomness_receipt, candidate
    )
    schedules = derive_schedules(seed_sha256)
    OUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = create_and_activate_attempt(OUT_ROOT, candidate, uuid.uuid4().hex)
    engine_lock_fd = os.open(
        attempt / "engine.lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    os.unlink(attempt / "engine.lock")
    engine_lock_stat = os.fstat(engine_lock_fd)
    engine_lock_path = f"/proc/{os.getpid()}/fd/{engine_lock_fd}"
    engine_lock_identity = f"{engine_lock_stat.st_dev}:{engine_lock_stat.st_ino}"
    if shutil.disk_usage(attempt).free < 8 * 1024**3:
        raise CampaignError("less than 8 GiB free for campaign evidence")
    write_new(attempt / "randomness-receipt.json", randomness_receipt_bytes)
    request_path = attempt / "primary-request.json"
    request_sha256 = make_primary_request(request_path)
    config = {
        "binary_sha256": BINARY_SHA256, "model_sha256": MODEL_SHA256,
        "context": 8192, "cache_gib": 40, "fetch_threads": 6,
        "boundary_align": 4, "boundary_trim": 8, "max_tokens": 160,
        "temperature": 0, "containment": {
            "MemoryHigh": "derived_from_start_available_minus_kill_floor_minus_8_GiB",
            "MemoryMax": "derived_from_start_available_minus_kill_floor_minus_4_GiB",
            "MemorySwapMax": 0,
            "kill_floor_GiB": 24, "minimum_start_GiB": 110, "timeout_s": 2400,
        },
    }
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows: list[dict[str, object]] = []
    failure: str | None = None
    try:
        for block, schedule in enumerate(schedules):
            for position, letter in enumerate(schedule):
                verify_candidate(candidate)
                arm = "off" if letter == "A" else "on"
                run_id = f"b{block}-p{position}-{arm}-{uuid.uuid4().hex[:12]}"
                out = attempt / run_id
                out.mkdir(mode=0o700)
                env, environment_sha = environment_for_arm(
                    arm, out, request_sha256, engine_lock_path, engine_lock_identity,
                    campaign_lock_fd, memory_guard_path,
                )
                tag = f"w7p-b{block}p{position}-{uuid.uuid4().hex[:10]}"
                command = [
                    str(CGROUP), "--tag", tag, "--", "/usr/bin/python3", str(Path(__file__)),
                    "--driver", arm, str(out), str(request_path), request_sha256, candidate,
                    model_devino, model_descriptor_path, engine_lock_path,
                ]
                completed = run_contained_command(command, env, tag)
                write_new(out / "containment.stdout", completed.stdout.encode())
                write_new(out / "containment.stderr", completed.stderr.encode())
                write_new(out / "containment.rc", f"{completed.returncode}\n".encode())
                row = parse_arm(
                    arm, block, position, out, completed.returncode, completed.stdout,
                    request_sha256, config_sha256,
                )
                row["executed_environment_sha256"] = environment_sha
                rows.append(row)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"

    # The scorer has an exact schema; runtime-only environment bindings live in raw metadata.
    scorer_rows = []
    for row in rows:
        scorer_row = dict(row)
        scorer_row.pop("executed_environment_sha256")
        scorer_rows.append(scorer_row)
    scorer = load_scorer(frozen_scorer_bytes, SCORER_SHA256)
    summary = scorer.score_campaign_rows(scorer_rows, schedules)
    if failure is not None:
        summary = dict(summary)
        summary["runtime_failure"] = failure
        summary["verdict"] = "FAIL"
    raw_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )
    summary_bytes = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    artifacts: dict[str, str] = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "raw.jsonl", "summary.json"}:
            artifacts[str(path.relative_to(attempt))] = sha256_file(path)
    manifest = {
        "schema": "glm52-w7-cache-generation-campaign-v1",
        "candidate_hash": candidate,
        "runner_sha256": sha256_file(Path(__file__)),
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
        "executed_request_sha256": request_sha256,
        "configuration": config,
        "configuration_sha256": config_sha256,
        "public_randomness_sha256": seed_sha256,
        "public_randomness_receipt_sha256": randomness_receipt_sha256,
        "schedules": schedules,
        "completed_rows": len(rows),
        "artifacts": artifacts,
    }
    manifest["artifacts"]["raw.jsonl"] = hashlib.sha256(raw_bytes).hexdigest()
    manifest["artifacts"]["summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    write_new(attempt / "raw.jsonl", raw_bytes)
    write_new(attempt / "summary.json", summary_bytes)
    write_new(attempt / "manifest.json", manifest_bytes)
    directory_fd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"W7_CACHE_GENERATION_CAMPAIGN_ATTEMPT={attempt}")
    result = 0 if summary["verdict"] == "PASS" else 1
    os.close(engine_lock_fd)
    os.close(model_fd)
    os.close(memory_guard_fd)
    fcntl.flock(campaign_lock_fd, fcntl.LOCK_UN)
    os.close(campaign_lock_fd)
    _ACTIVE_ATTEMPT = None
    _ACTIVE_CANDIDATE = None
    restore_campaign_signal_handlers(previous_signal_handlers)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--randomness-receipt", type=Path)
    parser.add_argument(
        "--driver", nargs=8,
        metavar=(
            "ARM", "OUT", "REQUEST", "REQUEST_SHA256", "CANDIDATE",
            "MODEL_DEVINO", "MODEL_DESCRIPTOR", "ENGINE_LOCK",
        ),
    )
    args = parser.parse_args()
    if args.self_test:
        if args.candidate or args.randomness_receipt or args.driver:
            return 2
        verify_dependencies()
        print("W7_CACHE_GENERATION_CAMPAIGN_SELFTEST_OK")
        return 0
    if args.driver:
        if args.candidate or args.randomness_receipt:
            return 2
        arm, out, request, request_sha256, candidate, model_devino, model_descriptor, engine_lock = args.driver
        return driver(
            arm, Path(out), Path(request), request_sha256, candidate,
            model_devino, model_descriptor, engine_lock,
        )
    if args.candidate and args.randomness_receipt:
        return campaign(args.candidate, args.randomness_receipt)
    return 2


if __name__ == "__main__":
    try:
        return_code = main()
    except BaseException as error:
        try:
            finalize_failure_triplet(error)
        except Exception as finalization_error:
            print(f"W7_CAMPAIGN_FINALIZATION_FAIL: {finalization_error}", file=sys.stderr)
        if isinstance(error, CampaignError):
            print(f"W7_CAMPAIGN_FAIL: {error}", file=sys.stderr)
            raise SystemExit(2)
        raise
    raise SystemExit(return_code)
