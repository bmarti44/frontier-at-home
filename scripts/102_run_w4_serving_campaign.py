#!/usr/bin/env python3
"""Run W4's contained, matched, fresh-server prefill campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import ssl
import stat
import subprocess
import sys
import time
import types
import urllib.request
import uuid
from datetime import datetime
from typing import Any


ROOT = Path("/home/bmarti44/spark-deepseek-v4-flash")
BASE_PATH = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
MEMORY_GUARD = ROOT / "scripts/03_memory_guard.py"
SCORER = ROOT / "scripts/101_score_w4_serving_campaign.py"
MICROGATE = ROOT / "results/glm52-gates/W4-topk-candidate5-pass/summary.json"
FIXTURE = ROOT / "fixtures/ctx-32k.txt"
DRAND_VERIFIER = ROOT / "scripts/89_verify_drand_receipt.mjs"
DRAND_BUNDLE = ROOT / "scripts/103_verify_drand_receipt_bundle.mjs"
DRAND_NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
SYSTEM_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
BIN = Path("/home/bmarti44/.cache/glm52-w4-topk-c5-serving/ds4-server")
CANDIDATE_SRC = BIN.parent
MODEL = Path("/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf")
OUT_ROOT = Path("/home/bmarti44/.local/state/glm52-w4-serving-campaign")
CAMPAIGN_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
PORT = 8098

BINARY_SHA256 = "620fd8fa2b6cd0885f11c70cebfecf0ca128580a5dd2e27f05822d4ff4b4651f"
ENGINE_SOURCE_COMMIT = "0424a6b406e4f6e125be3269104f3d16ad39c951"
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
MODEL_BYTES = 211075856448
LOGIT_BYTES = 154880 * 4
BASE_SHA256 = "e2f6235cd5f94b67773e75cff0f4fbceaa264f5b88e3d12b45ae3bb1e31e6924"
CGROUP_SHA256 = "d604c4e64f102ce03a7d6660b887e5b6c78091eeea72eab82874f34f9f4efb14"
SAFE_SHA256 = "2ddffb19f79b790c419db8ac53574d23ccf9f2c7699136fbaa55fc2a890b19e6"
MEMORY_GUARD_SHA256 = "3928675ff7ab496910d80775f536cceb6ee9b28f40b33ebbbd634e219a08cf58"
SCORER_SHA256 = "077e89d287e06b09a37b68d0188982d3fd825051a502c7881f85cb0cdb7606a8"
MICROGATE_SHA256 = "9aaf51b0722ec2573876d6a35ce733e6e574bb1349daf6f72f61100995c39bde"
FIXTURE_SHA256 = "2d31aeb3156ae01ab7213cdf50eb7660df8e869de12be7646a6b19aaf3405031"
DRAND_VERIFIER_SHA256 = "c191d301e1ff8460fffaea9dfeaab7d0fce0d63f92d3fdfcfa20442ccfdc2131"
DRAND_BUNDLE_SHA256 = "b3729f3b9d213a9c040ea77324ee96d419c3bf4b8a80e2be9f54e6d8ebcff2bc"
DRAND_NODE_SHA256 = "3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90"
SYSTEM_CA_BUNDLE_SHA256 = "6602a85a36afc2e51c66a0df5ae3d383c5b7c2fed93339ccef7d37e01faf09e8"
DRAND_FREEZE_FLOOR_ROUND = 6359295
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
DONE_RE = re.compile(
    r"SAFE_RUN_DONE rc=0 killed=no dir=(/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+) "
    r"main_sha256=([0-9a-f]{64}) samples_sha256=([0-9a-f]{64}) kernel_sha256=([0-9a-f]{64})\s*\Z"
)
TOPK_MARKER = "ds4: CUDA exact top-2048 CUB enabled chunk=8192 merge=2"
LISTENER = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"
SYNC_RE = re.compile(
    r"^ds4: GLM sync start=(\d+) prompt=(\d+) suffix=(\d+) checkpoint=(\d+) "
    r"dense_len=\d+ ctx_cap=\d+ dense_fit=\d+ resume_min=\d+ dense_gap=\d+ "
    r"indexed_keep=\d+ indexed_batch=\d+ batch_ffn=\d+$"
)


def _load_base() -> Any:
    injected = globals().get("_W4_INJECTED_BASE_BYTES")
    if injected is not None:
        if not isinstance(injected, bytes) or hashlib.sha256(injected).hexdigest() != BASE_SHA256:
            raise RuntimeError("injected W7 campaign base differs")
        module = types.ModuleType("w7_campaign_base")
        module.__file__ = str(BASE_PATH)
        exec(compile(injected, str(BASE_PATH), "exec"), module.__dict__)
        module.PORT = PORT
        return module
    spec = importlib.util.spec_from_file_location("w7_campaign_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load W7 campaign base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PORT = PORT
    return module


BASE = _load_base()
CampaignError = BASE.CampaignError


def sha256_file(path: Path) -> str:
    return BASE.sha256_file(path)


def verify_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise CampaignError(f"dependency mismatch: {path}")


SEALS = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK |
         fcntl.F_SEAL_SEAL)


def _sealed_memfd(name: str, payload: bytes, mode: int) -> int:
    descriptor = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            offset += os.write(descriptor, view[offset:])
        os.fchmod(descriptor, mode)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, SEALS)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != SEALS:
            raise CampaignError("sealed verifier descriptor differs")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _sealed_bls_verify(arguments: list[str]) -> bool:
    bundle_bytes, _ = BASE.read_stable(DRAND_BUNDLE)
    if hashlib.sha256(bundle_bytes).hexdigest() != DRAND_BUNDLE_SHA256:
        raise CampaignError("sealed drand verifier dependency mismatch")
    node_bytes, _ = BASE.read_stable(DRAND_NODE)
    if hashlib.sha256(node_bytes).hexdigest() != DRAND_NODE_SHA256:
        raise CampaignError("sealed drand verifier dependency mismatch")
    node_fd = _sealed_memfd("w4-node", node_bytes, 0o500)
    try:
        checked = subprocess.run(
            [f"/proc/self/fd/{node_fd}", "-", *arguments],
            input=bundle_bytes, capture_output=True, check=False, timeout=30,
            pass_fds=(node_fd,),
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin",
                 "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        return (checked.returncode == 0 and checked.stdout == b"DRAND_BLS_RECEIPT_OK\n"
                and not checked.stderr)
    finally:
        os.close(node_fd)


def user_systemd_available() -> bool:
    completed = subprocess.run(
        ["/usr/bin/systemctl", "--user", "show-environment"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    return completed.returncode == 0


def verify_dependencies() -> None:
    for path, digest in (
        (BASE_PATH, BASE_SHA256), (BIN, BINARY_SHA256), (CGROUP, CGROUP_SHA256),
        (SAFE, SAFE_SHA256), (MEMORY_GUARD, MEMORY_GUARD_SHA256),
        (SCORER, SCORER_SHA256), (MICROGATE, MICROGATE_SHA256),
        (FIXTURE, FIXTURE_SHA256), (DRAND_VERIFIER, DRAND_VERIFIER_SHA256),
        (DRAND_BUNDLE, DRAND_BUNDLE_SHA256),
        (DRAND_NODE, DRAND_NODE_SHA256), (SYSTEM_CA_BUNDLE, SYSTEM_CA_BUNDLE_SHA256),
    ):
        verify_file(path, digest)
    metadata = MODEL.stat()
    if MODEL.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != MODEL_BYTES:
        raise CampaignError("model identity mismatch")


def git_bytes(candidate: str, relative: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-C", str(ROOT), "show",
         f"{candidate}:{relative}"], check=True, capture_output=True,
    ).stdout


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
    relative = "scripts/102_run_w4_serving_campaign.py"
    if head != candidate or dirty or git_bytes(candidate, relative) != Path(__file__).read_bytes():
        raise CampaignError("candidate is not the clean executing HEAD")


def derive_schedules(seed_sha256: str) -> list[str]:
    if SHA_RE.fullmatch(seed_sha256) is None:
        raise ValueError("seed must be lowercase SHA-256")
    seed = bytes.fromhex(seed_sha256)
    domain = b"W4-SERVING-PREFILL-SCHEDULE-V1\0"
    return [
        "ABBA" if hashlib.sha256(domain + seed + bytes([block])).digest()[0] & 1 == 0
        else "BAAB" for block in range(5)
    ]


def drand_publication_time(round_number: int) -> int:
    if type(round_number) is not int or round_number < 1:
        raise ValueError("invalid drand round")
    return 1595431050 + (round_number - 1) * 30


def first_drand_round_after(timestamp: int) -> int:
    if type(timestamp) is not int or timestamp < 1595431050:
        raise ValueError("invalid publication timestamp")
    return (timestamp - 1595431050) // 30 + 2


def fetch_publication_receipt(candidate: str) -> dict[str, object]:
    if COMMIT_RE.fullmatch(candidate) is None:
        raise CampaignError("invalid publication candidate")
    request = urllib.request.Request(
        "https://api.github.com/repos/bmarti44/frontier-at-home/events?per_page=100",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "frontier-at-home-w4-gate"})
    ca_bytes, _ = BASE.read_stable(SYSTEM_CA_BUNDLE)
    if hashlib.sha256(ca_bytes).hexdigest() != SYSTEM_CA_BUNDLE_SHA256:
        raise CampaignError("system CA bundle mismatch")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=ca_bytes.decode("ascii"))
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
    with opener.open(request, timeout=30) as response:
        events = json.loads(response.read())
    if not isinstance(events, list):
        raise CampaignError("GitHub publication response is malformed")
    matches = []
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if (event.get("type") == "PushEvent" and payload.get("head") == candidate
                and payload.get("ref") == "refs/heads/glm52-rung0-io-submission"):
            matches.append(event)
    if len(matches) != 1:
        raise CampaignError("candidate lacks one recent GitHub publication event")
    event = matches[0]
    created = event.get("created_at")
    if not isinstance(created, str):
        raise CampaignError("GitHub publication timestamp is missing")
    try:
        timestamp = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
    except ValueError as error:
        raise CampaignError("GitHub publication timestamp is invalid") from error
    return {"event_id": event.get("id"), "created_at": created,
            "created_at_unix": timestamp, "candidate_hash": candidate,
            "ref": "refs/heads/glm52-rung0-io-submission",
            "repository": "bmarti44/frontier-at-home"}


def verify_randomness_bytes(raw: bytes, candidate: str,
                            publication: dict[str, object]) -> tuple[str, str, bytes]:
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("invalid randomness receipt") from error
    required = {"round", "freeze_floor_round", "randomness", "signature",
                "previous_signature", "frozen_gate_commit", "relay_agreement"}
    if not isinstance(doc, dict) or not required.issubset(doc):
        raise CampaignError("randomness receipt fields missing")
    if (type(doc["round"]) is not int or doc["round"] <= DRAND_FREEZE_FLOOR_ROUND
            or doc["freeze_floor_round"] != DRAND_FREEZE_FLOOR_ROUND
            or doc["frozen_gate_commit"] != candidate
            or doc["relay_agreement"] != ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"]
            or SHA_RE.fullmatch(doc["randomness"] or "") is None
            or re.fullmatch(r"[0-9a-f]{192}", doc["signature"] or "") is None
            or re.fullmatch(r"[0-9a-f]{192}", doc["previous_signature"] or "") is None):
        raise CampaignError("randomness is not post-freeze and three-relay bound")
    if not _sealed_bls_verify([
            str(doc["round"]), doc["randomness"], doc["signature"],
            doc["previous_signature"]]):
        raise CampaignError("randomness BLS verification failed")
    if (publication.get("candidate_hash") != candidate
            or type(publication.get("created_at_unix")) is not int):
        raise CampaignError("candidate publication receipt is invalid")
    required_round = first_drand_round_after(publication["created_at_unix"])
    if doc["round"] != required_round:
        raise CampaignError("randomness is not the first eligible post-publication round")
    return doc["randomness"], hashlib.sha256(raw).hexdigest(), raw


def verify_randomness(path: Path, candidate: str,
                      publication: dict[str, object]) -> tuple[str, str, bytes]:
    raw, _ = BASE.read_stable(path)
    return verify_randomness_bytes(raw, candidate, publication)


def make_request(path: Path) -> str:
    text = FIXTURE.read_text(encoding="utf-8")
    prompt = "Reference document follows.\n" + text[:int(len(text) * 0.46)] + \
             "\nReturn no text."
    request = {"model": "glm-5.2", "prompt": prompt, "max_tokens": 0,
               "temperature": 0, "stream": False}
    payload = (json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n").encode()
    BASE.write_new(path, payload)
    return hashlib.sha256(payload).hexdigest()


def semantic_response(raw: bytes) -> tuple[dict[str, Any], str]:
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("response is not valid JSON") from error
    choices = doc.get("choices") if isinstance(doc, dict) else None
    usage = doc.get("usage") if isinstance(doc, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    if (not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or choices[0].get("text") != ""
            or choices[0].get("finish_reason") != "length"
            or not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
            or usage["prompt_tokens"] < 16_000 or usage.get("completion_tokens") != 0
            or usage.get("total_tokens") != usage["prompt_tokens"]
            or not isinstance(details, dict) or details.get("cached_tokens") != 0
            or details.get("cache_write_tokens") != usage["prompt_tokens"]):
        raise CampaignError("response does not bind a novel >=16K-token prefill")
    semantic = {
        "choices": [{"text": "", "finish_reason": "length"}],
        "usage": {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": 0,
                  "total_tokens": usage["prompt_tokens"],
                  "prompt_tokens_details": {
                      "cached_tokens": 0, "cache_write_tokens": usage["prompt_tokens"]}},
    }
    digest = hashlib.sha256(json.dumps(
        semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return semantic, digest


def validate_novel_sync_trace(server_log: str, expected_prompt_tokens: int) -> None:
    matches = [SYNC_RE.fullmatch(line) for line in server_log.splitlines()]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1 or tuple(map(int, matches[0].groups()[:4])) != (
            0, expected_prompt_tokens, expected_prompt_tokens, 0):
        raise CampaignError("sync trace does not prove a novel complete prefill")


def measured_environment(arm: str, out: Path, lock_path: str,
                         lock_identity: str) -> dict[str, str]:
    measured = {
        "DS4_CUDA_EXPERT_CACHE_GB": "40", "DS4_CUDA_EXPERT_CACHE_PIN": "1",
        "DS4_CUDA_EXPERT_CACHE_SLRU": "1", "DS4_CUDA_FETCH_THREADS": "6",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1", "DS4_CUDA_STABLE_MODEL_REMAP": "1",
        "DS4_GLM_LOGIT_DUMP": str(out / "logits"), "DS4_GLM_LOGIT_DUMP_ALL": "1",
        "DS4_GLM_SYNC_TRACE": "1", "DS4_LOCK_EXPECTED_DEV_INO": lock_identity,
        "DS4_LOCK_FILE": lock_path,
    }
    if arm == "on":
        measured["DS4_CUDA_TOPK2048_CUB"] = "1"
    return measured


def validate_environment_artifact(arm: str, out: Path, observed: object) -> str:
    if not isinstance(observed, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in observed.items()):
        raise CampaignError("environment artifact is malformed")
    lock_path = observed.get("DS4_LOCK_FILE", "")
    lock_identity = observed.get("DS4_LOCK_EXPECTED_DEV_INO", "")
    if (re.fullmatch(r"/proc/[1-9][0-9]*/fd/[0-9]+", lock_path) is None
            or re.fullmatch(r"[0-9]+:[0-9]+", lock_identity) is None
            or observed != measured_environment(arm, out, lock_path, lock_identity)):
        raise CampaignError("environment artifact differs from the fixed arm")
    return hashlib.sha256("".join(
        f"{name}={observed[name]}\n" for name in sorted(observed)).encode()).hexdigest()


def environment_for_arm(arm: str, out: Path, lock_path: str, lock_identity: str,
                        campaign_lock_fd: int | None = None,
                        memory_guard_path: str = str(MEMORY_GUARD)) -> tuple[dict[str, str], str]:
    if arm not in {"off", "on"}:
        raise CampaignError("invalid arm")
    measured = measured_environment(arm, out, lock_path, lock_identity)
    digest = hashlib.sha256("".join(
        f"{name}={measured[name]}\n" for name in sorted(measured)).encode()).hexdigest()
    env = {
        "HOME": "/home/bmarti44", "USER": "bmarti44", "LOGNAME": "bmarti44",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        **measured,
        "GLM_CANDIDATE_SRC": str(CANDIDATE_SRC), "GLM_PORT": str(PORT),
        "GLM_EXPERT_CACHE_GB": "40", "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
        "GLM_SAFE_KILL_FLOOR_GIB": "24", "GLM_SAFE_MIN_START_GIB": "110",
        "GLM_SAFE_TIMEOUT_S": "3600", "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
        "GLM_SAFE_EXPECTED_BINARY_SHA256": BINARY_SHA256,
        "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(sorted(measured)),
        "GLM_SAFE_EXPECTED_ENV_SHA256": digest,
        "GLM_SAFE_MEMORY_GUARD_PATH": memory_guard_path,
        "GLM_SAFE_EXPECTED_MEMORY_GUARD_SHA256": MEMORY_GUARD_SHA256,
        "GLM_SAFE_FINAL_ARTIFACTS": ",".join(str(out / name) for name in (
            "server.log", "response.json", "observation.json", "environment.json",
            "child-exit.json")),
        "GLM_SAFE_DONE_DIGESTS": "1",
    }
    if campaign_lock_fd is not None:
        metadata = os.fstat(campaign_lock_fd)
        env.update({
            "GLM_SAFE_PARENT_LOCK_PID": str(os.getpid()),
            "GLM_SAFE_PARENT_LOCK_START_TICKS": str(BASE.process_start_ticks(os.getpid())),
            "GLM_SAFE_PARENT_LOCK_FD": str(campaign_lock_fd),
            "GLM_SAFE_PARENT_LOCK_DEV_INO": f"{metadata.st_dev}:{metadata.st_ino}",
            "GLM_SAFE_PARENT_LOCK_KERNEL_KEY": BASE.lock_kernel_key(metadata),
        })
    return env, digest


def driver(arm: str, out: Path, request_path: Path, request_sha256: str, candidate: str,
           expected_model_devino: str, model_descriptor_path: str, lock_path: str) -> int:
    verify_dependencies()
    verify_candidate(candidate)
    if BASE.server_pids() or any(out.iterdir()):
        raise CampaignError("driver requires a fresh output and no server")
    if sha256_file(request_path) != request_sha256:
        raise CampaignError("request digest mismatch")
    if os.environ.get("DS4_CUDA_TOPK2048_CUB") != ("1" if arm == "on" else None):
        raise CampaignError("top-k flag/arm mismatch")
    observed_environment = measured_environment(
        arm, out, os.environ.get("DS4_LOCK_FILE", ""),
        os.environ.get("DS4_LOCK_EXPECTED_DEV_INO", ""))
    if any(os.environ.get(name) != value for name, value in observed_environment.items()):
        raise CampaignError("executed environment differs from fixed arm")
    BASE.write_json_new(out / "environment.json", observed_environment)
    model_fd = os.open(model_descriptor_path, os.O_RDONLY | os.O_CLOEXEC)
    metadata = os.fstat(model_fd)
    if f"{metadata.st_dev}:{metadata.st_ino}" != expected_model_devino or metadata.st_size != MODEL_BYTES:
        os.close(model_fd)
        raise CampaignError("model descriptor identity mismatch")
    if not stat.S_ISREG(os.stat(lock_path).st_mode):
        os.close(model_fd)
        raise CampaignError("engine lock unavailable")
    kv = out / "kv"
    kv.mkdir(mode=0o700)
    command = [str(BIN), "--cuda", "-m", f"/proc/self/fd/{model_fd}", "-c", "32768",
               "--host", "127.0.0.1", "--port", str(PORT), "--ssd-streaming",
               "--ssd-streaming-cache-experts", "40GB", "--kv-disk-dir", str(kv),
               "--kv-disk-space-mb", "4096", "--kv-cache-boundary-align-tokens", "4",
               "--kv-cache-boundary-trim-tokens", "8"]
    process: subprocess.Popen[bytes] | None = None
    try:
        with (out / "server.log").open("xb", buffering=0) as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=(model_fd,))
            BASE.wait_ready(process)
            body = request_path.read_bytes()
            request = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            started = time.monotonic_ns()
            with urllib.request.urlopen(request, timeout=2700) as response:
                if response.status != 200:
                    raise CampaignError(f"request returned HTTP {response.status}")
                raw = response.read()
            completed = time.monotonic_ns()
            semantic, semantic_sha = semantic_response(raw)
            BASE.write_new(out / "response.json", raw)
            BASE.write_json_new(out / "observation.json", {
                "request_start_ns": started, "response_complete_ns": completed,
                "semantic": semantic, "response_semantic_sha256": semantic_sha})
            exit_status = BASE.stop_server(process)
            process = None
            if exit_status != 0:
                raise CampaignError(f"server exited {exit_status}")
            BASE.write_json_new(out / "child-exit.json", {
                "shutdown_requested": True, "forced_kill": False, "exit_status": 0})
            BASE.remove_kv_scratch(kv)
    finally:
        if process is not None:
            try:
                BASE.stop_server(process)
            except Exception:
                pass
        os.close(model_fd)
    os.sync()
    return 0


def parse_arm(arm: str, block: int, position: int, out: Path, containment_rc: int,
              stdout: str, request_sha256: str, config_sha256: str,
              expected_prompt_tokens: int, expected_environment_sha256: str | None = None,
              replay: bool = False,
              snapshot: dict[str, tuple[bytes, tuple[int, int, int, int]]] | None = None,
              ) -> dict[str, Any]:
    if containment_rc != 0:
        raise CampaignError(f"containment failed rc={containment_rc}")
    done = DONE_RE.fullmatch(stdout)
    if done is None:
        raise CampaignError("safe-run receipt missing")
    crash = Path(done.group(1))
    safety_dir = out / "safety"
    safety: dict[str, bytes] = {}
    if replay:
        if not safety_dir.is_dir() or safety_dir.is_symlink():
            raise CampaignError("copied safety evidence is unavailable")
    else:
        safety_dir.mkdir(mode=0o700)
    def evidence(name: str) -> tuple[bytes, Any]:
        if snapshot is None:
            return BASE.read_stable(out / name)
        if name not in snapshot:
            raise CampaignError(f"snapshotted artifact missing: {name}")
        payload, identity = snapshot[name]
        return payload, types.SimpleNamespace(
            st_dev=identity[0], st_ino=identity[1], st_size=identity[2],
            st_mtime_ns=identity[3])
    for offset, name in enumerate(("main.log", "samples.log", "kernel.log"), start=2):
        source = f"safety/{name}" if replay else None
        payload, _ = evidence(source) if source is not None else BASE.read_stable(crash / name)
        if hashlib.sha256(payload).hexdigest() != done.group(offset):
            raise CampaignError(f"safe-run digest mismatch: {name}")
        if not replay:
            BASE.write_new(safety_dir / name, payload)
        safety[name] = payload
    artifacts = {name: evidence(name) for name in
                 ("server.log", "response.json", "observation.json", "environment.json",
                  "child-exit.json")}
    main = safety["main.log"].decode("utf-8", errors="strict")
    samples = safety["samples.log"].decode("utf-8", errors="strict")
    kernel = safety["kernel.log"].decode("utf-8", errors="strict")
    server = artifacts["server.log"][0].decode("utf-8", errors="strict")
    observation = json.loads(artifacts["observation.json"][0])
    environment_artifact = json.loads(artifacts["environment.json"][0])
    response = artifacts["response.json"][0]
    _, semantic_sha = semantic_response(response)
    environment_lines = re.findall(
        r"executed_environment_allowlist=[A-Z0-9_,]+ "
        r"executed_environment_sha256=([0-9a-f]{64})", main)
    artifact_environment_sha = validate_environment_artifact(arm, out, environment_artifact)
    if len(environment_lines) != 1 or environment_lines[0] != artifact_environment_sha or (
            expected_environment_sha256 is not None
            and environment_lines[0] != expected_environment_sha256):
        raise CampaignError("executed environment binding mismatch")
    for name, (payload, metadata) in artifacts.items():
        marker = (f"final_artifact_verified path={out / name} "
                  f"sha256={hashlib.sha256(payload).hexdigest()} "
                  f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}")
        if main.count(marker) != 1:
            raise CampaignError(f"final artifact binding mismatch: {name}")
    logit_names = (sorted(name for name in snapshot if name.startswith("logits.sync"))
                   if snapshot is not None else
                   [path.name for path in sorted(out.glob("logits.sync*.start*.prompt*.suffix*"))])
    if len(logit_names) != 1 or ".sync1." not in logit_names[0]:
        raise CampaignError("expected exactly one synchronized logit tensor")
    logit_name = logit_names[0]
    logits, metadata = evidence(logit_name)
    if metadata.st_size != LOGIT_BYTES:
        raise CampaignError("wrong logit tensor size")
    logit_sha = hashlib.sha256(logits).hexdigest()
    sequence_sha = hashlib.sha256(json.dumps(
        [(logit_name, logit_sha, metadata.st_size)],
        separators=(",", ":")).encode()).hexdigest()
    memory = [int(value) for value in re.findall(r"\bmem_avail_kb=([0-9]+)\b", samples)]
    swaps = [int(value) for value in re.findall(r"\bcgroup_swap_current_bytes=([0-9]+)\b", samples)]
    final = re.findall(
        r"cgroup_final .* swap_current_bytes=([0-9]+) events=low [0-9]+,high [0-9]+,"
        r"max ([0-9]+),oom ([0-9]+),oom_kill ([0-9]+),oom_group_kill ([0-9]+),", main)
    if not memory or not swaps or len(final) != 1:
        raise CampaignError("incomplete safety counters")
    clean_descendant_marker = (
        "wrapper and descendant checks clean" in main
        and main.count("wrapper and descendant checks clean") == 1
    )
    if not clean_descendant_marker:
        raise CampaignError("safe-run descendant-clean attestation missing")
    max_delta, oom_delta, oom_kill_delta, oom_group = map(int, final[0][1:])
    if oom_group != 0:
        raise CampaignError("cgroup OOM group kill")
    usage = observation["semantic"]["usage"]
    validate_novel_sync_trace(server, expected_prompt_tokens)
    if usage["prompt_tokens"] != expected_prompt_tokens:
        raise CampaignError("API and independent token accounting differ")
    return {
        "block": block, "position": position, "arm": arm, "run_id": out.name,
        "binary_sha256": BINARY_SHA256, "model_sha256": MODEL_SHA256,
        "common_config_sha256": config_sha256, "request_sha256": request_sha256,
        "topk_cub": int(arm == "on"),
        "request_start_ns": observation["request_start_ns"],
        "response_complete_ns": observation["response_complete_ns"],
        "prompt_tokens": usage["prompt_tokens"], "cached_tokens": 0,
        "cache_write_tokens": usage["prompt_tokens"],
        "response_semantic_sha256": semantic_sha, "final_logits_sha256": logit_sha,
        "logit_sequence_sha256": sequence_sha,
        "executed_environment_sha256": environment_lines[0],
        "topk_marker_count": server.count(TOPK_MARKER),
        "server_fresh": server.count(LISTENER) == 1 and server.count(SHUTDOWN) == 1,
        "safety": {"containment_rc": 0, "minimum_mem_available_kib": min(memory),
                   "swap_growth_bytes": max(swaps + [int(final[0][0])]),
                   "cgroup_max_delta": max_delta, "cgroup_oom_delta": oom_delta,
                   "cgroup_oom_kill_delta": oom_kill_delta,
                   "xid_count": len(re.findall(r"\bXid\b", kernel, re.IGNORECASE)),
                   "surviving_descendants": 0 if replay else len(BASE.server_pids())},
    }


def load_scorer(payload: bytes) -> Any:
    if hashlib.sha256(payload).hexdigest() != SCORER_SHA256:
        raise CampaignError("scorer digest mismatch")
    module = types.ModuleType("w4_serving_scorer")
    exec(compile(payload, str(SCORER), "exec"), module.__dict__)
    return module


def finalize_failure(error: BaseException) -> None:
    """Preserve an escaping W4 failure without borrowing W7's manifest identity."""
    attempt = BASE._ACTIVE_ATTEMPT
    candidate = BASE._ACTIVE_CANDIDATE
    if attempt is None or candidate is None:
        return
    failure = f"{type(error).__name__}: {error}"
    rejected_symlinks = sorted(
        str(path.relative_to(attempt)) for path in attempt.rglob("*") if path.is_symlink()
    )
    manifest_path = attempt / "manifest.json"
    if manifest_path.is_symlink():
        manifest_path.unlink()
    elif manifest_path.exists():
        os.rename(manifest_path, attempt / "manifest.pre-failure.json")
    raw_path = attempt / "raw.jsonl"
    if raw_path.is_symlink():
        raw_path.unlink()
        raw = b""
    else:
        raw = BASE.read_stable(raw_path)[0] if raw_path.is_file() else b""
    if not raw_path.exists():
        BASE.write_new(raw_path, raw)
    summary_path = attempt / "summary.json"
    displaced: tuple[str, str] | None = None
    if summary_path.is_symlink():
        summary_path.unlink()
    elif summary_path.exists():
        prior, _ = BASE.read_stable(summary_path)
        prior_path = attempt / "summary.pre-finalization.json"
        os.rename(summary_path, prior_path)
        displaced = (prior_path.name, hashlib.sha256(prior).hexdigest())
    summary = (json.dumps({"failure": failure, "verdict": "FAIL"},
                          sort_keys=True, separators=(",", ":")) + "\n").encode()
    BASE.write_new(summary_path, summary)
    artifacts = {}
    rejected_unstable = []
    for path in sorted(attempt.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            relative = str(path.relative_to(attempt))
            try:
                payload, _ = BASE.read_stable(path)
            except (OSError, CampaignError):
                rejected_unstable.append(relative)
                continue
            artifacts[relative] = hashlib.sha256(payload).hexdigest()
    BASE.write_json_new(manifest_path, {
        "schema": "glm52-w4-serving-campaign-failure-v1",
        "candidate_hash": candidate, "failure": failure,
        "runner_sha256": sha256_file(Path(__file__)),
        "scorer_sha256": SCORER_SHA256, "base_runner_sha256": BASE_SHA256,
        "cgroup_sha256": CGROUP_SHA256, "safe_run_sha256": SAFE_SHA256,
        "memory_guard_sha256": MEMORY_GUARD_SHA256,
        "binary_sha256": BINARY_SHA256,
        "model_sha256": MODEL_SHA256, "fixture_sha256": FIXTURE_SHA256,
        "microgate_sha256": MICROGATE_SHA256,
        "drand_verifier_sha256": DRAND_VERIFIER_SHA256,
        "drand_node_sha256": DRAND_NODE_SHA256,
        "drand_bundle_sha256": DRAND_BUNDLE_SHA256,
        "system_ca_bundle_sha256": SYSTEM_CA_BUNDLE_SHA256,
        "rejected_symlinks": rejected_symlinks,
        "rejected_unstable_paths": rejected_unstable,
        "artifacts": artifacts, "verdict": "FAIL",
    })


def _campaign(candidate: str, receipt: Path) -> int:
    if COMMIT_RE.fullmatch(candidate) is None:
        raise CampaignError("invalid candidate commit")
    OUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = BASE.create_and_activate_attempt(OUT_ROOT, candidate, uuid.uuid4().hex)
    if not user_systemd_available():
        raise CampaignError("user-systemd containment is unavailable")
    verify_dependencies()
    verify_candidate(candidate)
    if BASE.server_pids() or subprocess.run(
            ["/usr/bin/pgrep", "-x", "fio"], capture_output=True).returncode == 0:
        raise CampaignError("engine or fio already active")
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:")))
    if available < 110 * 1024 * 1024:
        raise CampaignError("less than 110 GiB available")
    lock_fd = os.open(CAMPAIGN_LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scorer_bytes = git_bytes(candidate, "scripts/101_score_w4_serving_campaign.py")
    if scorer_bytes != BASE.read_stable(SCORER)[0] or hashlib.sha256(scorer_bytes).hexdigest() != SCORER_SHA256:
        raise CampaignError("frozen scorer mismatch")
    guard_fd = os.open(MEMORY_GUARD, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    if BASE.sha256_descriptor(guard_fd) != MEMORY_GUARD_SHA256:
        raise CampaignError("memory guard mismatch")
    model_fd = os.open(MODEL, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    before = os.fstat(model_fd)
    if BASE.sha256_descriptor(model_fd) != MODEL_SHA256:
        raise CampaignError("model content mismatch")
    after = os.fstat(model_fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise CampaignError("model changed during scan")
    BASE.write_json_new(attempt / "model-identity.json", {
        "sha256": MODEL_SHA256, "bytes": after.st_size,
        "device": after.st_dev, "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns, "descriptor_held_for_all_arms": True,
    })
    publication = fetch_publication_receipt(candidate)
    seed, receipt_sha, receipt_bytes = verify_randomness(receipt, candidate, publication)
    schedules = derive_schedules(seed)
    engine_fd = os.open(attempt / "engine.lock", os.O_RDWR | os.O_CREAT | os.O_EXCL |
                        os.O_NOFOLLOW, 0o600)
    os.unlink(attempt / "engine.lock")
    engine_stat = os.fstat(engine_fd)
    engine_path = f"/proc/{os.getpid()}/fd/{engine_fd}"
    engine_identity = f"{engine_stat.st_dev}:{engine_stat.st_ino}"
    BASE.write_new(attempt / "randomness-receipt.json", receipt_bytes)
    BASE.write_json_new(attempt / "publication-receipt.json", publication)
    request_path = attempt / "request.json"
    request_sha = make_request(request_path)
    microgate_bytes, _ = BASE.read_stable(MICROGATE)
    microgate = json.loads(microgate_bytes)
    selected_micro = {name: microgate[name] for name in (
        "block_a_ms", "block_b_ms", "selected_ids_sha256", "speedup_lower_95",
        "required_speedup_lower_95", "verdict")}
    BASE.write_json_new(attempt / "microgate-summary.json", selected_micro)
    BASE.write_json_new(attempt / "schedules.json", schedules)
    tokenization = load_scorer(scorer_bytes).independent_tokenization(request_path)
    BASE.write_json_new(attempt / "request-tokenization.json", tokenization)
    config = load_scorer(scorer_bytes).EXPECTED_CONFIGURATION
    config_sha = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    BASE.write_json_new(attempt / "configuration.json", config)
    rows: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        for block, schedule in enumerate(schedules):
            for position, letter in enumerate(schedule):
                verify_candidate(candidate)
                arm = "off" if letter == "A" else "on"
                out = attempt / f"b{block}-p{position}-{arm}-{uuid.uuid4().hex[:12]}"
                out.mkdir(mode=0o700)
                env, env_sha = environment_for_arm(
                    arm, out, engine_path, engine_identity, lock_fd,
                    f"/proc/{os.getpid()}/fd/{guard_fd}")
                tag = f"w4s-b{block}p{position}-{uuid.uuid4().hex[:10]}"
                command = [str(CGROUP), "--tag", tag, "--", "/usr/bin/python3",
                           str(Path(__file__)), "--driver", arm, str(out), str(request_path),
                           request_sha, candidate, f"{after.st_dev}:{after.st_ino}",
                           f"/proc/{os.getpid()}/fd/{model_fd}", engine_path]
                completed = BASE.run_contained_command(command, env, tag)
                BASE.write_new(out / "containment.stdout", completed.stdout.encode())
                BASE.write_new(out / "containment.stderr", completed.stderr.encode())
                BASE.write_new(out / "containment.rc", f"{completed.returncode}\n".encode())
                row = parse_arm(arm, block, position, out, completed.returncode,
                                completed.stdout, request_sha, config_sha,
                                tokenization["prompt_tokens"], env_sha)
                rows.append(row)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    summary = load_scorer(scorer_bytes).score_campaign_rows(
        rows, schedules, selected_micro)
    if failure:
        summary = {**summary, "runtime_failure": failure, "verdict": "FAIL"}
    raw = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                   for row in rows)
    summary_bytes = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    artifacts = {str(path.relative_to(attempt)): sha256_file(path)
                 for path in sorted(attempt.rglob("*")) if path.is_file()}
    manifest = {
        "schema": "glm52-w4-serving-campaign-v1", "candidate_hash": candidate,
        "runner_sha256": sha256_file(Path(__file__)), "scorer_sha256": SCORER_SHA256,
        "base_runner_sha256": BASE_SHA256, "cgroup_sha256": CGROUP_SHA256,
        "safe_run_sha256": SAFE_SHA256, "memory_guard_sha256": MEMORY_GUARD_SHA256,
        "binary_sha256": BINARY_SHA256, "engine_source_commit": ENGINE_SOURCE_COMMIT,
        "model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES,
        "fixture_sha256": FIXTURE_SHA256, "executed_request_sha256": request_sha,
        "microgate_sha256": MICROGATE_SHA256, "configuration": config,
        "configuration_sha256": config_sha, "tokenization": tokenization,
        "public_randomness_sha256": seed,
        "public_randomness_receipt_sha256": receipt_sha, "schedules": schedules,
        "publication_receipt": publication,
        "drand_bundle_sha256": DRAND_BUNDLE_SHA256,
        "system_ca_bundle_sha256": SYSTEM_CA_BUNDLE_SHA256,
        "completed_rows": len(rows), "artifacts": artifacts,
        "verdict": summary["verdict"],
    }
    manifest["artifacts"]["raw.jsonl"] = hashlib.sha256(raw).hexdigest()
    manifest["artifacts"]["summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
    BASE.write_new(attempt / "raw.jsonl", raw)
    BASE.write_new(attempt / "summary.json", summary_bytes)
    BASE.write_json_new(attempt / "manifest.json", manifest)
    if summary["verdict"] == "PASS":
        scorer_bootstrap = (
            "import sys;__file__=" + repr(str(SCORER)) +
            ";exec(compile(sys.stdin.buffer.read(),__file__,'exec'))"
        )
        replay = subprocess.run(
            ["/usr/bin/python3", "-I", "-B", "-c", scorer_bootstrap, str(attempt)],
            input=scorer_bytes, capture_output=True, check=False, timeout=300,
            env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin",
                 "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
        try:
            replay_result = json.loads(replay.stdout)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CampaignError("authoritative scorer replay emitted invalid JSON") from error
        if replay.returncode != 0 or replay.stderr or replay_result != summary:
            raise CampaignError("authoritative scorer did not reproduce PASS")
    print(f"W4_SERVING_CAMPAIGN_ATTEMPT={attempt}")
    for descriptor in (engine_fd, model_fd, guard_fd):
        os.close(descriptor)
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)
    return 0 if summary["verdict"] == "PASS" else 1


def campaign(candidate: str, receipt: Path) -> int:
    handlers = BASE.install_campaign_signal_handlers()
    try:
        return _campaign(candidate, receipt)
    finally:
        BASE.restore_campaign_signal_handlers(handlers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument("--driver", nargs=8, metavar=("ARM", "OUT", "REQUEST", "REQUEST_SHA",
                                                      "CANDIDATE", "MODEL_DEVINO",
                                                      "MODEL_DESCRIPTOR", "LOCK"))
    args = parser.parse_args()
    if args.driver:
        arm, out, request, request_sha, candidate, devino, descriptor, lock = args.driver
        return driver(arm, Path(out), Path(request), request_sha, candidate,
                      devino, descriptor, lock)
    if args.candidate is None or args.receipt is None:
        parser.error("candidate and receipt are required")
    return campaign(args.candidate, args.receipt)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        finalize_failure(error)
        print(f"W4_SERVING_CAMPAIGN_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
