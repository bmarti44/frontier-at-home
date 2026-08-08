#!/usr/bin/env python3
"""Score the W7 stable-model expert-cache generation invariant.

This is deliberately a log scorer, not an engine self-reported verdict.  A
single server process that has begun accepting requests must not announce a
model-generation change while serving a fixed model.  The later paired gate
still owns byte identity and performance acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile


LISTEN = "ds4-server: listening on "
SHUTDOWN = "ds4-server: shutdown requested"
CACHE_ENABLED = "CUDA persistent expert cache enabled:"
INDEXED_RESUME = "GLM sync branch=indexed_resume"
PROMPT_DONE = " prompt done "
FALSE_FLUSH = "CUDA persistent expert cache flushed (model load generation changed)"
STABLE_REMAP = "CUDA stable model remap enabled generation="
STABLE_REMAP_RE = re.compile(r"CUDA stable model remap enabled generation=([1-9][0-9]*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DONE_RE = re.compile(
    r"SAFE_RUN_DONE rc=0 killed=no dir=(/home/bmarti44/\.local/state/glm52-crashlog/[A-Za-z0-9._-]+) "
    r"main_sha256=([0-9a-f]{64}) samples_sha256=([0-9a-f]{64}) kernel_sha256=([0-9a-f]{64})\n?"
)
PROMPT_START_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) prompt start$"
)
PROMPT_DONE_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) prompt done [0-9]+(?:\.[0-9]+)?s$"
)
FATAL_PATTERNS = (
    re.compile(r"CUDA GLM prefill failed", re.IGNORECASE),
    re.compile(
        r"(?:CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|"
        r"CUDA.{0,160}(?:allocation failed|out of memory))",
        re.IGNORECASE,
    ),
    re.compile(r"NV_ERR_NO_MEMORY", re.IGNORECASE),
    re.compile(r"forward_token failed", re.IGNORECASE),
    re.compile(r"(?:out of memory|oom-kill|killed process)", re.IGNORECASE),
    re.compile(r"\bFATAL\b", re.IGNORECASE),
    re.compile(r"(?:NVRM.*)?\bXid\b", re.IGNORECASE),
    re.compile(r"request timeout", re.IGNORECASE),
)


def score_text(
    text: str,
    *,
    http_status: str,
    response_text: str,
    containment_rc: str,
    containment_stdout: str,
    mode: str = "on",
    child_exit_text: str = '{"shutdown_requested":true,"forced_kill":false,"exit_status":0}',
    safety_main_text: str = "",
    expected_binary_sha256: str = "",
    expected_environment_sha256: str = "",
    expected_memory_guard_sha256: str = "",
    model_identity_text: str = "",
    expected_model_sha256: str = "",
    expected_model_bytes: int = 0,
) -> dict[str, object]:
    lines = text.splitlines()
    listen = next((i for i, line in enumerate(lines) if LISTEN in line), None)
    shutdown_indexes = [
        i for i, line in enumerate(lines) if SHUTDOWN in line and (listen is None or i > listen)
    ]
    shutdown = shutdown_indexes[0] if len(shutdown_indexes) == 1 else None
    window = lines[listen + 1 : shutdown] if listen is not None and shutdown is not None else []
    cache_enabled_count = sum(CACHE_ENABLED in line for line in window)
    request_windows: list[dict[str, object]] = []
    starts = [i for i, line in enumerate(window) if PROMPT_START_RE.search(line)]
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(window)
        lines_for_request = window[start:end]
        start_match = PROMPT_START_RE.search(lines_for_request[0])
        assert start_match is not None
        done_matches = [
            match
            for line in lines_for_request
            if (match := PROMPT_DONE_RE.search(line)) is not None
        ]
        request_windows.append(
            {
                "context_start": int(start_match.group(1)),
                "prompt_tokens": int(start_match.group(2)),
                "suffix_tokens": int(start_match.group(3)),
                "indexed_resume_count": sum(INDEXED_RESUME in line for line in lines_for_request),
                "matching_prompt_done_count": sum(
                    match.groups() == start_match.groups() for match in done_matches
                ),
            }
        )
    indexed_completed = [
        item
        for item in request_windows
        if item["indexed_resume_count"] == 1
        and item["matching_prompt_done_count"] == 1
    ]
    response_ok = False
    response_tuple = None
    try:
        response = json.loads(response_text)
        usage = response["usage"]
        choices = response["choices"]
        prompt_details = usage["prompt_tokens_details"]
        response_tuple = (
            prompt_details["cached_tokens"],
            usage["prompt_tokens"],
            prompt_details["cache_write_tokens"],
        )
        response_ok = (
            isinstance(response, dict)
            and "error" not in response
            and isinstance(usage, dict)
            and isinstance(prompt_details, dict)
            and isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            and type(choices[0].get("text")) is str
            and choices[0]["text"] == ""
            and choices[0].get("finish_reason") == "length"
            and all(
                type(value) is int
                for value in (
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    prompt_details.get("cached_tokens"),
                    prompt_details.get("cache_write_tokens"),
                )
            )
            and usage["completion_tokens"] == 0
            and usage["total_tokens"] == usage["prompt_tokens"]
            and prompt_details["cached_tokens"] + prompt_details["cache_write_tokens"]
            == usage["prompt_tokens"]
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        response_ok = False
    matching_response_windows = [
        item
        for item in request_windows
        if response_ok
        and (
            item["context_start"],
            item["prompt_tokens"],
            item["suffix_tokens"],
        )
        == response_tuple
    ]
    request_bound = (
        len(indexed_completed) == 1
        and len(matching_response_windows) == 1
        and matching_response_windows[0] is indexed_completed[0]
    )
    false_flush_count = sum(FALSE_FLUSH in line for line in window)
    activation_generations = [
        int(match.group(1))
        for line in lines[: listen + 1 if listen is not None else 0]
        if (match := STABLE_REMAP_RE.search(line)) is not None
    ]
    activation_ok = (
        mode == "on" and len(activation_generations) == 1
    ) or (
        mode == "off" and not activation_generations and STABLE_REMAP not in text
    )
    try:
        child_exit = json.loads(child_exit_text)
    except (json.JSONDecodeError, TypeError):
        child_exit = None
    child_exit_ok = (
        type(child_exit) is dict
        and set(child_exit) == {"shutdown_requested", "forced_kill", "exit_status"}
        and child_exit.get("shutdown_requested") is True
        and child_exit.get("forced_kill") is False
        and type(child_exit.get("exit_status")) is int
        and child_exit.get("exit_status") == 0
    )
    try:
        model_identity = json.loads(model_identity_text)
    except (json.JSONDecodeError, TypeError):
        model_identity = None
    model_identity_ok = (
        type(model_identity) is dict
        and set(model_identity) == {"bytes", "device", "inode", "sha256", "executed_path"}
        and type(model_identity.get("bytes")) is int
        and model_identity.get("bytes") == expected_model_bytes
        and type(model_identity.get("device")) is int
        and type(model_identity.get("inode")) is int
        and model_identity.get("device", -1) >= 0
        and model_identity.get("inode", 0) > 0
        and model_identity.get("sha256") == expected_model_sha256
        and type(model_identity.get("executed_path")) is str
        and re.fullmatch(r"/proc/[1-9][0-9]*/fd/[0-9]+", model_identity["executed_path"])
        is not None
    )
    fatal_markers = sorted(
        {
            match.group(0)
            for pattern in FATAL_PATTERNS
            for line in window
            if (match := pattern.search(line)) is not None
        }
    )
    clean_containment = (
        containment_rc.strip() == "0"
        and DONE_RE.fullmatch(containment_stdout) is not None
    )
    safety_lines = safety_main_text.splitlines()
    safe_start_lines = [line for line in safety_lines if "SAFE_RUN start " in line]
    cgroup_lines = [line for line in safety_lines if "cgroup_verified " in line]
    binary_lines = [line for line in safety_lines if "executed_candidate_verified " in line]
    environment_lines = [line for line in safety_lines if "executed_environment_allowlist=" in line]
    memory_guard_lines = [line for line in safety_lines if "memory_guard_descriptor_path=" in line]
    clean_exit_lines = [line for line in safety_lines if "executed candidate was verified alive at least once;" in line]
    safe_end_lines = [line for line in safety_lines if "SAFE_RUN end " in line]
    safety_checks = {
        "safe_start_settings_bound": len(safe_start_lines) == 1
        and all(marker in safe_start_lines[0] for marker in (
            "vlimit_kb=419430400", "kill_floor_gib=24", "min_start_gib=110", "timeout_s=2400"
        )),
        "cgroup_settings_bound": len(cgroup_lines) == 1
        and all(marker in cgroup_lines[0] for marker in (
            "memory_high=83751862272", "memory_max=85899345920",
            "memory_swap_max=0", "memory_oom_group=1",
        )),
        "executed_binary_bound": bool(expected_binary_sha256)
        and len(binary_lines) == 1
        and f"executed_binary_sha256={expected_binary_sha256}" in binary_lines[0],
        "executed_environment_bound": bool(expected_environment_sha256)
        and len(environment_lines) == 1
        and "executed_environment_allowlist=DS4_CUDA_STABLE_MODEL_REMAP" in environment_lines[0]
        and f"executed_environment_sha256={expected_environment_sha256}" in environment_lines[0],
        "safe_wrapper_clean_exit": len(clean_exit_lines) == 1
        and len(safe_end_lines) == 1
        and "SAFE_RUN end rc=0 killed=no" in safe_end_lines[0]
        and not any("FATAL" in line or "KILL_FLOOR breached" in line for line in safety_lines),
    }
    if expected_memory_guard_sha256:
        safety_checks["memory_guard_identity_bound"] = (
            SHA256_RE.fullmatch(expected_memory_guard_sha256) is not None
            and len(memory_guard_lines) == 1
            and f"memory_guard_sha256={expected_memory_guard_sha256}" in memory_guard_lines[0]
        )
    checks = {
        "listener_observed": listen is not None,
        "shutdown_observed_once": len(shutdown_indexes) == 1,
        "stable_remap_activation_matches_arm": activation_ok,
        "child_exit_clean": child_exit_ok,
        "executed_model_identity_bound": model_identity_ok,
        "persistent_cache_enabled_once": cache_enabled_count == 1,
        "exactly_one_bound_indexed_resume_completed": request_bound,
        "http_status_200": http_status.strip() == "200",
        "response_schema_valid": response_ok,
        "clean_containment": clean_containment,
        **safety_checks,
        "fatal_markers_absent": not fatal_markers,
        "stable_model_generation_flushes_zero": false_flush_count == 0,
    }
    return {
        "formula": "PASS iff all checks are true; generation-change flushes are counted strictly after listener readiness and before shutdown",
        "observed": {
            "cache_enabled_count": cache_enabled_count,
            "request_window_count": len(request_windows),
            "bound_indexed_resume_count": len(indexed_completed),
            "response_request_tuple": response_tuple,
            "matching_response_window_count": len(matching_response_windows),
            "fatal_markers": fatal_markers,
            "false_generation_flush_count": false_flush_count,
            "stable_remap_activation_generations": activation_generations,
            "shutdown_marker_count": len(shutdown_indexes),
            "child_exit": child_exit,
            "model_identity": model_identity,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def write_exclusive(path: Path, rendered: str) -> None:
    """Publish a result once; never follow or replace an existing path."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        payload = rendered.encode("utf-8")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def publish_triplet_atomic(
    destination: Path,
    manifest: dict[str, object],
    raw_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Publish the complete evidence triplet with one no-replace directory rename."""
    parent = destination.parent
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise ValueError("evidence parent must be a real directory")
    raw_bytes = b"".join(_json_bytes(row) for row in raw_rows)
    summary_bytes = _json_bytes(summary)
    bound_manifest = json.loads(json.dumps(manifest, allow_nan=False))
    artifacts = bound_manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be an object")
    artifacts["raw.jsonl"] = hashlib.sha256(raw_bytes).hexdigest()
    artifacts["summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
    manifest_bytes = _json_bytes(bound_manifest)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    published = False
    try:
        _write_exclusive_bytes(temporary / "raw.jsonl", raw_bytes)
        _write_exclusive_bytes(temporary / "summary.json", summary_bytes)
        _write_exclusive_bytes(temporary / "manifest.json", manifest_bytes)
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(temporary, destination)
        published = True
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def copy_bound_artifact(source: Path, destination: Path, expected_sha256: str | None) -> dict[str, object]:
    """Copy one regular file through stable descriptors and bind its reviewed digest."""
    if expected_sha256 is not None and SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("invalid expected SHA-256")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    source_fd = os.open(source, flags)
    reserved_fd, reserved_path = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(reserved_fd)
    temporary = Path(reserved_path)
    os.unlink(temporary)
    destination_fd = -1
    published = False
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source is not a regular file")
        destination_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        identity_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        actual = digest.hexdigest()
        if identity_before != identity_after or (expected_sha256 is not None and actual != expected_sha256):
            raise ValueError("source identity or digest changed")
        os.close(destination_fd)
        destination_fd = -1
        _rename_noreplace(temporary, destination)
        published = True
        return {"sha256": actual, "device": before.st_dev, "inode": before.st_ino, "bytes": before.st_size}
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if not published:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def bind_runtime_artifacts(attempt: Path, out: Path, crash_dir: Path) -> dict[str, object]:
    """Snapshot runtime files and require every engine file's safe-run digest."""
    bound = out / "bound"
    safety = bound / "safety"
    bound.mkdir(mode=0o700)
    safety.mkdir(mode=0o700)
    bindings: dict[str, object] = {}
    for name in ("containment.stdout", "containment.stderr", "containment.rc"):
        bindings[name] = copy_bound_artifact(attempt / name, bound / name, None)
    containment_text = (bound / "containment.stdout").read_text(encoding="utf-8", errors="strict")
    done_match = DONE_RE.fullmatch(containment_text)
    if done_match is None or Path(done_match.group(1)) != crash_dir:
        raise ValueError("containment completion digest record mismatch")
    main_sha256, samples_sha256, kernel_sha256 = done_match.groups()[1:]
    bindings["safety/main.log"] = copy_bound_artifact(
        crash_dir / "main.log", safety / "main.log", main_sha256
    )
    main_text = (safety / "main.log").read_text(encoding="utf-8", errors="strict")
    final_re = re.compile(
        r" final_artifact_verified path=([^ ]+) sha256=([0-9a-f]{64}) device_inode=([0-9]+:[0-9]+:[0-9]+)$"
    )
    recorded: dict[str, tuple[str, int, int, int]] = {}
    for line in main_text.splitlines():
        match = final_re.search(line)
        if match:
            if match.group(1) in recorded:
                raise ValueError("duplicate final artifact binding")
            device, inode, size = (int(value) for value in match.group(3).split(":"))
            recorded[match.group(1)] = (match.group(2), device, inode, size)
    runtime_names = (
        "server.log",
        "live-response.json",
        "live-http-status",
        "primary-response.json",
        "primary-http-status",
        "child-exit.json",
        "model.identity.json",
    )
    expected_paths = {str(out / name) for name in runtime_names}
    if set(recorded) != expected_paths:
        raise ValueError("safe-run final artifact set mismatch")
    for name in runtime_names:
        expected_digest, expected_device, expected_inode, expected_bytes = recorded[str(out / name)]
        binding = copy_bound_artifact(out / name, bound / name, expected_digest)
        if (
            binding["device"], binding["inode"], binding["bytes"]
        ) != (expected_device, expected_inode, expected_bytes):
            raise ValueError("safe-run final artifact identity mismatch")
        bindings[name] = binding
    safety_re = re.compile(
        r" safety_artifact_verified name=(samples|kernel)\.log sha256=([0-9a-f]{64}) size=([0-9]+)$"
    )
    safety_hashes: dict[str, str] = {}
    for line in main_text.splitlines():
        match = safety_re.search(line)
        if match:
            name = f"{match.group(1)}.log"
            if name in safety_hashes:
                raise ValueError("duplicate safety artifact binding")
            safety_hashes[name] = match.group(2)
    if set(safety_hashes) != {"samples.log", "kernel.log"}:
        raise ValueError("safe-run safety artifact set mismatch")
    if safety_hashes != {"samples.log": samples_sha256, "kernel.log": kernel_sha256}:
        raise ValueError("stdout and main safety digests disagree")
    for name, digest in safety_hashes.items():
        bindings[f"safety/{name}"] = copy_bound_artifact(crash_dir / name, safety / name, digest)
    _write_exclusive_bytes(bound / "bindings.json", _json_bytes(bindings))
    return bindings


def _read_snapshot(
    path: Path,
    expected_sha256: str | None = None,
    expected_identity: tuple[int, int, int] | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Read and validate bytes through the one descriptor later used by scoring."""
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("snapshot source is not a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size)
    identity_after = (after.st_dev, after.st_ino, after.st_size)
    if identity_before != identity_after or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns:
        raise ValueError("snapshot identity changed while reading")
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("snapshot digest mismatch")
    if expected_identity is not None and identity_before != expected_identity:
        raise ValueError("snapshot device/inode/size mismatch")
    return b"".join(chunks), {
        "sha256": actual,
        "device": before.st_dev,
        "inode": before.st_ino,
        "bytes": before.st_size,
    }


def score_and_publish_bound_attempt(
    *,
    attempt: Path,
    out: Path,
    crash_dir: Path,
    evidence_dir: Path,
    identities: dict[str, object],
    containment_stdout: str,
    containment_rc: int,
) -> dict[str, object]:
    """Bind, score, and publish from one immutable in-memory snapshot."""
    if type(containment_rc) is not int:
        raise ValueError("invalid private containment status")
    payloads: dict[str, bytes] = {}
    bindings: dict[str, dict[str, object]] = {}
    payloads["containment.stdout"] = containment_stdout.encode("utf-8", errors="strict")
    payloads["containment.rc"] = f"{containment_rc}\n".encode("ascii")
    bindings["containment.stdout"] = {
        "channel": "private-command-pipe",
        "sha256": hashlib.sha256(payloads["containment.stdout"]).hexdigest(),
        "bytes": len(payloads["containment.stdout"]),
    }
    bindings["containment.rc"] = {
        "channel": "parent-shell-exit-status",
        "sha256": hashlib.sha256(payloads["containment.rc"]).hexdigest(),
        "bytes": len(payloads["containment.rc"]),
    }
    payloads["containment.stderr"], bindings["containment.stderr"] = _read_snapshot(
        attempt / "containment.stderr"
    )
    containment_text = containment_stdout
    done_match = DONE_RE.fullmatch(containment_text)
    if done_match is None or Path(done_match.group(1)) != crash_dir:
        raise ValueError("containment completion digest record mismatch")
    main_sha256, samples_sha256, kernel_sha256 = done_match.groups()[1:]
    payloads["safety/main.log"], bindings["safety/main.log"] = _read_snapshot(
        crash_dir / "main.log", main_sha256
    )
    main_text = payloads["safety/main.log"].decode("utf-8", errors="strict")
    final_re = re.compile(
        r" final_artifact_verified path=([^ ]+) sha256=([0-9a-f]{64}) device_inode=([0-9]+:[0-9]+:[0-9]+)$"
    )
    recorded: dict[str, tuple[str, tuple[int, int, int]]] = {}
    for line in main_text.splitlines():
        match = final_re.search(line)
        if match:
            if match.group(1) in recorded:
                raise ValueError("duplicate final artifact binding")
            recorded[match.group(1)] = (
                match.group(2), tuple(int(value) for value in match.group(3).split(":"))
            )
    runtime_names = (
        "server.log", "live-response.json", "live-http-status", "primary-response.json",
        "primary-http-status", "child-exit.json", "model.identity.json",
    )
    if set(recorded) != {str(out / name) for name in runtime_names}:
        raise ValueError("safe-run final artifact set mismatch")
    for name in runtime_names:
        digest, identity = recorded[str(out / name)]
        payloads[name], bindings[name] = _read_snapshot(out / name, digest, identity)
    safety_re = re.compile(
        r" safety_artifact_verified name=(samples|kernel)\.log sha256=([0-9a-f]{64}) size=([0-9]+)$"
    )
    safety_hashes: dict[str, str] = {}
    for line in main_text.splitlines():
        match = safety_re.search(line)
        if match:
            name = f"{match.group(1)}.log"
            if name in safety_hashes:
                raise ValueError("duplicate safety artifact binding")
            safety_hashes[name] = match.group(2)
    if safety_hashes != {"samples.log": samples_sha256, "kernel.log": kernel_sha256}:
        raise ValueError("safe-run safety artifact digest mismatch")
    for name, digest in safety_hashes.items():
        key = f"safety/{name}"
        payloads[key], bindings[key] = _read_snapshot(crash_dir / name, digest)

    expected_binary = str(identities["binary_sha256"])
    expected_environment = str(identities["executed_environment_sha256"])
    expected_memory_guard = str(identities["memory_guard_sha256"])
    expected_model = str(identities["model_sha256"])
    expected_model_bytes = identities["model_bytes"]
    if type(expected_model_bytes) is not int:
        raise ValueError("invalid expected model bytes")
    result = score_text(
        payloads["server.log"].decode("utf-8", errors="strict"),
        http_status=payloads["primary-http-status"].decode("utf-8", errors="strict"),
        response_text=payloads["primary-response.json"].decode("utf-8", errors="strict"),
        containment_rc=str(containment_rc),
        containment_stdout=containment_text,
        mode="on",
        child_exit_text=payloads["child-exit.json"].decode("utf-8", errors="strict"),
        safety_main_text=main_text,
        expected_binary_sha256=expected_binary,
        expected_environment_sha256=expected_environment,
        expected_memory_guard_sha256=expected_memory_guard,
        model_identity_text=payloads["model.identity.json"].decode("utf-8", errors="strict"),
        expected_model_sha256=expected_model,
        expected_model_bytes=expected_model_bytes,
    )
    final_head_matches_candidate = (
        identities.get("execution_head") == identities.get("candidate_hash")
    )
    result["checks"]["final_head_matches_candidate"] = final_head_matches_candidate
    if not final_head_matches_candidate:
        result["verdict"] = "FAIL"
    manifest = {
        "schema": "glm52-w7-runtime-v3",
        **identities,
        "arm": "on",
        "purpose": "single-arm production-path diagnostic, not performance or context capability",
        "public_randomness": None,
        "artifact_bindings": bindings,
        "artifacts": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    }
    rows: list[dict[str, object]] = []
    for name in ("server.log", "safety/main.log", "safety/samples.log", "safety/kernel.log", "containment.stdout", "containment.stderr", "containment.rc"):
        for number, line in enumerate(payloads[name].decode("utf-8", errors="strict").splitlines(), 1):
            rows.append({"source": name, "line_number": number, "text": line})
    publish_triplet_atomic(evidence_dir, manifest, rows, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--http-status", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--containment-rc", type=Path, required=True)
    parser.add_argument("--containment-stdout", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--child-exit", type=Path, required=True)
    parser.add_argument("--safety-main", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--expected-environment-sha256", required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-model-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_text(
        args.server_log.read_text(encoding="utf-8", errors="strict"),
        http_status=args.http_status.read_text(encoding="utf-8", errors="strict"),
        response_text=args.response.read_text(encoding="utf-8", errors="strict"),
        containment_rc=args.containment_rc.read_text(encoding="utf-8", errors="strict"),
        containment_stdout=args.containment_stdout.read_text(encoding="utf-8", errors="strict"),
        mode=args.mode,
        child_exit_text=args.child_exit.read_text(encoding="utf-8", errors="strict"),
        safety_main_text=args.safety_main.read_text(encoding="utf-8", errors="strict"),
        expected_binary_sha256=args.expected_binary_sha256,
        expected_environment_sha256=args.expected_environment_sha256,
        model_identity_text=args.model_identity.read_text(encoding="utf-8", errors="strict"),
        expected_model_sha256=args.expected_model_sha256,
        expected_model_bytes=args.expected_model_bytes,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        write_exclusive(args.output, rendered)
    else:
        print(rendered, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
