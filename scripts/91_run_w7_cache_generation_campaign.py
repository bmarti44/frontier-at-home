#!/usr/bin/env python3
"""Run the contained, fresh-server W7.1 stable-remap OFF/ON campaign."""

from __future__ import annotations

import argparse
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
LOCK = Path("/run/user/1000/ds4-engine.lock")
PORT = 8097

BINARY_SHA256 = "eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2"
ENGINE_SOURCE_COMMIT = "bccf0b66e196e88213add7e0f81571bb7e558670"
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
MODEL_BYTES = 211075856448
LOGIT_BYTES = 154880 * 4
LIVE_SHA256 = "d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a"
PRIMARY_SHA256 = "a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3"
CGROUP_SHA256 = "fe298d8fb24db1155d0b2f996879ec4ae32e6f4b6e5183d9e26240d10d42d0ea"
SAFE_SHA256 = "2ddffb19f79b790c419db8ac53574d23ccf9f2c7699136fbaa55fc2a890b19e6"
MEMORY_GUARD_SHA256 = "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58"
SCORER_SHA256 = "b02654462877ebf47a0449dc681d8f1172b5c26df3694e3619b1c7c94b2342d4"
DRAND_VERIFIER_SHA256 = "c191d301e1ff8460fffaea9dfeaab7d0fce0d63f92d3fdfcfa20442ccfdc2131"
DRAND_NODE_SHA256 = "3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
TIMING_RE = re.compile(
    r"^DS4_TOKEN_TIMING request=(\S+) index=(\d+) monotonic_ns=(\d+) token=(-?\d+)$"
)
DONE_RE = re.compile(
    r"SAFE_RUN_DONE rc=0 killed=no dir=(/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+) "
    r"main_sha256=[0-9a-f]{64} samples_sha256=[0-9a-f]{64} kernel_sha256=[0-9a-f]{64}\s*\Z"
)
FALSE_FLUSH = "CUDA persistent expert cache flushed (model load generation changed)"
LISTENER = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"


class CampaignError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    lock_stat = LOCK.stat()
    if LOCK.is_symlink() or (lock_stat.st_mode & 0o777) != 0o600 or lock_stat.st_uid != os.getuid():
        raise CampaignError("engine lock identity mismatch")


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
        or doc["round"] <= doc["freeze_floor_round"]
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


def driver(arm: str, out: Path, request_path: Path, request_sha256: str, candidate: str) -> int:
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

    kv = out / "kv"
    kv.mkdir(mode=0o700)
    server_log = out / "server.log"
    command = [
        str(BIN), "--cuda", "-m", str(MODEL), "-c", "8192", "--host", "127.0.0.1",
        "--port", str(PORT), "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
        "--kv-disk-dir", str(kv), "--kv-disk-space-mb", "4096",
        "--kv-cache-boundary-align-tokens", "4", "--kv-cache-boundary-trim-tokens", "8",
    ]
    process: subprocess.Popen[bytes] | None = None
    with server_log.open("xb", buffering=0) as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
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
    os.sync()
    return 0


def environment_for_arm(arm: str, out: Path, request_sha256: str) -> tuple[dict[str, str], str]:
    lock_identity = f"{LOCK.stat().st_dev}:{LOCK.stat().st_ino}"
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
        "DS4_LOCK_FILE": str(LOCK),
        "DS4_TOKEN_TIMING_LOG": "1",
    }
    if arm == "on":
        measured["DS4_CUDA_STABLE_MODEL_REMAP"] = "1"
    digest = hashlib.sha256(
        "".join(f"{name}={measured[name]}\n" for name in sorted(measured)).encode()
    ).hexdigest()
    env = os.environ.copy()
    env.update(measured)
    env.update({
        "GLM_CANDIDATE_SRC": str(CANDIDATE_SRC),
        "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
        "GLM_SAFE_MEMORY_HIGH_GIB": "78",
        "GLM_SAFE_KILL_FLOOR_GIB": "24",
        "GLM_SAFE_MIN_START_GIB": "110",
        "GLM_SAFE_TIMEOUT_S": "2400",
        "GLM_SAFE_ALLOW_CGROUP_HIGH": "1",
        "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
        "GLM_SAFE_EXPECTED_BINARY_SHA256": BINARY_SHA256,
        "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(sorted(measured)),
        "GLM_SAFE_EXPECTED_ENV_SHA256": digest,
        "GLM_SAFE_MEMORY_GUARD_PATH": str(MEMORY_GUARD),
        "GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256": MEMORY_GUARD_SHA256,
        "GLM_SAFE_FINAL_ARTIFACTS": ",".join(str(out / name) for name in (
            "server.log", "live-response.json", "primary-client.json", "child-exit.json"
        )),
        "GLM_SAFE_DONE_DIGESTS": "1",
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
    main = (crash / "main.log").read_text(encoding="utf-8")
    samples = (crash / "samples.log").read_text(encoding="utf-8")
    kernel = (crash / "kernel.log").read_text(encoding="utf-8")
    server = (out / "server.log").read_text(encoding="utf-8")
    client = json.loads((out / "primary-client.json").read_text(encoding="utf-8"))

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
    logit_records = [(path.name, sha256_file(path), path.stat().st_size) for path in logit_paths]
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


def load_scorer() -> Any:
    spec = importlib.util.spec_from_file_location("w7_cache_campaign_scorer", SCORER)
    if spec is None or spec.loader is None:
        raise CampaignError("cannot load scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaign(candidate: str, randomness_receipt: Path) -> int:
    verify_dependencies()
    verify_candidate(candidate)
    if server_pids() or subprocess.run(["/usr/bin/pgrep", "-x", "fio"], capture_output=True).returncode == 0:
        raise CampaignError("engine or fio already active")
    available_kb = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
    if available_kb < 110 * 1024 * 1024:
        raise CampaignError("less than 110 GiB available")
    seed_sha256, randomness_receipt_sha256, randomness_receipt_bytes = verify_public_randomness_receipt(
        randomness_receipt, candidate
    )
    schedules = derive_schedules(seed_sha256)
    OUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = OUT_ROOT / f"attempt-{uuid.uuid4().hex}"
    attempt.mkdir(mode=0o700)
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
            "MemoryHigh_GiB": 78, "MemoryMax_GiB": 80, "MemorySwapMax": 0,
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
                env, environment_sha = environment_for_arm(arm, out, request_sha256)
                tag = f"w7p-b{block}p{position}-{uuid.uuid4().hex[:10]}"
                command = [
                    str(CGROUP), "--tag", tag, "--", "/usr/bin/python3", str(Path(__file__)),
                    "--driver", arm, str(out), str(request_path), request_sha256, candidate,
                ]
                completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
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
    scorer = load_scorer()
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
    return 0 if summary["verdict"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--randomness-receipt", type=Path)
    parser.add_argument(
        "--driver", nargs=5,
        metavar=("ARM", "OUT", "REQUEST", "REQUEST_SHA256", "CANDIDATE"),
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
        arm, out, request, request_sha256, candidate = args.driver
        return driver(arm, Path(out), Path(request), request_sha256, candidate)
    if args.candidate and args.randomness_receipt:
        return campaign(args.candidate, args.randomness_receipt)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as error:
        print(f"W7_CAMPAIGN_FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
