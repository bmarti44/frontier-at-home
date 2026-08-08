#!/usr/bin/env python3
"""Run the frozen five-block DSV4 cold-load A/B campaign.

The runner deliberately owns only the experiment lifecycle.  It does not alter
the installed profile or service and refuses to run without a functioning
per-arm user-systemd cgroup.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import NamedTuple


ROOT = Path("/home/bmarti44/spark-deepseek-v4-flash")
SCORER = ROOT / "scripts/94_score_dsv4_cold_load.py"
MODEL_DIR = ROOT / "weights/unsloth-ud-q2_k_xl"
MODEL_MANIFEST = MODEL_DIR / "manifest.json"
CAMPAIGN_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
PORT = 8021
MIN_START_KB = 115 * 1024 * 1024
KILL_FLOOR_KB = 10 * 1024 * 1024
MAX_INITIAL_SWAP_BYTES = 1024**3
MAX_CACHE_BYTES = 1024**3
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
UNIT_RE = re.compile(r"cold-b[0-4]-p[0-3]-[0-9a-f]{12}\Z")
LOAD_START_MARKER = "llm_load_tensors:"
LOAD_END_MARKER = "llama_server: model loaded"
_LIBC = ctypes.CDLL(None, use_errno=True)
_ACTIVE_OUTPUT: Path | None = None


class CampaignError(RuntimeError):
    pass


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise CampaignError("short evidence write")
        view = view[written:]


class ArmPlan(NamedTuple):
    block: int
    position: int
    arm: str
    run_id: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_bundle_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.iterdir() if path.is_file() and not path.is_symlink())
    if not files:
        raise CampaignError("empty runtime bundle")
    for path in files:
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_json_new(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def arm_schedule(randomness_hex: str) -> list[str]:
    if SHA256_RE.fullmatch(randomness_hex) is None:
        raise ValueError("randomness must be one lowercase SHA-256 value")
    seed = bytes.fromhex(randomness_hex)
    domain = b"frontier-at-home/dsv4-cold-load/v1\0"
    return [
        "ABBA" if hashlib.sha256(domain + seed + bytes([block])).digest()[0] & 1 == 0 else "BAAB"
        for block in range(5)
    ]


def campaign_plan(randomness_hex: str) -> list[ArmPlan]:
    result: list[ArmPlan] = []
    nonce = hashlib.sha256(bytes.fromhex(randomness_hex) + b"run-id").hexdigest()
    for block, schedule in enumerate(arm_schedule(randomness_hex)):
        for position, letter in enumerate(schedule):
            suffix = hashlib.sha256(f"{nonce}:{block}:{position}".encode()).hexdigest()[:12]
            result.append(ArmPlan(block, position, "off" if letter == "A" else "on", f"b{block}-p{position}-{suffix}"))
    return result


def direct_io_arguments(arm: str) -> list[str]:
    if arm == "on":
        return ["--direct-io-required"]
    if arm == "off":
        return ["--no-direct-io"]
    raise CampaignError("invalid arm")


def require_fresh_output(path: Path) -> None:
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise CampaignError("arm output must be a fresh empty directory")


def _run(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)


def preflight_host() -> None:
    manager = _run(["/usr/bin/systemctl", "--user", "is-system-running"])
    if manager.returncode != 0 or manager.stdout.strip() not in {"running", "degraded"}:
        raise CampaignError("user-systemd containment is unavailable")
    for name in ("ds4-server", "llama-server", "fio"):
        probe = _run(["/usr/bin/pgrep", "-x", name])
        if probe.returncode == 0:
            raise CampaignError("engine or fio is already active")
        if probe.returncode != 1:
            raise CampaignError("cannot establish idle host")
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and fields[0] in {"MemAvailable:", "SwapTotal:", "SwapFree:"}:
            meminfo[fields[0]] = int(fields[1])
    if meminfo.get("MemAvailable:", 0) < MIN_START_KB:
        raise CampaignError("less than 115 GiB available")
    used_swap = (meminfo.get("SwapTotal:", 0) - meminfo.get("SwapFree:", 0)) * 1024
    if used_swap > MAX_INITIAL_SWAP_BYTES:
        raise CampaignError("more than 1 GiB swap is already in use")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", PORT))
    except OSError as error:
        raise CampaignError("campaign port is already occupied") from error
    finally:
        listener.close()


def containment_command(unit: str, server_command: list[str], log_path: Path) -> list[str]:
    if UNIT_RE.fullmatch(unit) is None or not server_command or not log_path.is_absolute():
        raise CampaignError("invalid containment input")
    return [
        "/usr/bin/systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}",
        "--property=Type=exec", "--property=MemoryHigh=100G", "--property=MemoryMax=104G",
        "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=45s",
        "--property=RuntimeMaxSec=300s", "--property=WorkingDirectory=/",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}", "--", *server_command,
    ]


def _read_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    raw = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    fields = raw[close + 2:].split() if close >= 0 else []
    if len(fields) <= 19:
        raise CampaignError("malformed process stat")
    return int(fields[19])


def direct_shard_count(pid: int, shards: list[Path], *, proc_root: Path = Path("/proc")) -> int:
    expected = {str(path.resolve()) for path in shards}
    observed: set[str] = set()
    fd_root = proc_root / str(pid) / "fd"
    for entry in fd_root.iterdir():
        try:
            target = os.readlink(entry)
            if target not in expected:
                continue
            lines = (proc_root / str(pid) / "fdinfo" / entry.name).read_text(encoding="ascii").splitlines()
            flags = [line.split()[1] for line in lines if line.startswith("flags:")]
            if len(flags) != 1:
                raise CampaignError("ambiguous descriptor flags")
            if int(flags[0], 8) & os.O_DIRECT:
                observed.add(target)
        except FileNotFoundError:
            continue
    return len(observed)


def _resident_bytes(path: Path) -> int:
    with path.open("rb", buffering=0) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size == 0:
            return 0
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            page = mmap.PAGESIZE
            count = (size + page - 1) // page
            vector = (ctypes.c_ubyte * count)()
            # Python exposes read-only mappings without a writable buffer.  A
            # zero-copy read-only address is obtained through PyObject_GetBuffer.
            class Buffer(ctypes.Structure):
                _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p), ("len", ctypes.c_ssize_t),
                            ("itemsize", ctypes.c_ssize_t), ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
                            ("format", ctypes.c_char_p), ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
                            ("strides", ctypes.POINTER(ctypes.c_ssize_t)), ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
                            ("internal", ctypes.c_void_p)]
            view = Buffer()
            get_buffer = ctypes.pythonapi.PyObject_GetBuffer
            get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(Buffer), ctypes.c_int]
            get_buffer.restype = ctypes.c_int
            release = ctypes.pythonapi.PyBuffer_Release
            release.argtypes = [ctypes.POINTER(Buffer)]
            if get_buffer(mapping, ctypes.byref(view), 0) != 0:
                raise CampaignError("cannot inspect file residency")
            try:
                if _LIBC.mincore(ctypes.c_void_p(view.buf), ctypes.c_size_t(size), vector) != 0:
                    raise OSError(ctypes.get_errno(), "mincore")
            finally:
                release(ctypes.byref(view))
            return sum(1 for value in vector if value & 1) * page
        finally:
            mapping.close()


def evict_and_measure(shards: list[Path]) -> int:
    for path in shards:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    os.sync()
    resident = sum(_resident_bytes(path) for path in shards)
    if resident > MAX_CACHE_BYTES:
        raise CampaignError("cold-cache precondition failed")
    return resident


def _unit_pid(unit: str) -> int:
    result = _run(["/usr/bin/systemctl", "--user", "show", f"{unit}.service", "-p", "MainPID", "--value"])
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        raise CampaignError("cannot bind contained server PID")
    return int(result.stdout.strip())


def _unit_stop(unit: str) -> None:
    _run(["/usr/bin/systemctl", "--user", "stop", f"{unit}.service"], timeout=60)
    for _ in range(120):
        state = _run(["/usr/bin/systemctl", "--user", "show", f"{unit}.service", "-p", "ActiveState", "--value"])
        if state.returncode != 0 or state.stdout.strip() in {"", "inactive", "failed"}:
            return
        time.sleep(0.25)
    _run(["/usr/bin/systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", f"{unit}.service"])
    raise CampaignError("contained server required SIGKILL")


def _cgroup_survivors(path: Path) -> int:
    try:
        values = [line for line in (path / "cgroup.procs").read_text(encoding="ascii").splitlines() if line]
    except FileNotFoundError:
        return 0
    if any(not value.isdigit() for value in values):
        raise CampaignError("malformed contained cgroup membership")
    return len(values)


def _post(url: str, payload: dict[str, object] | None, key: str | None) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _cgroup_counters(pid: int) -> tuple[Path, dict[str, int], int]:
    entries = [line for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines() if line.startswith("0::/")]
    if len(entries) != 1:
        raise CampaignError("cannot bind contained cgroup")
    path = Path("/sys/fs/cgroup") / entries[0][3:]
    events: dict[str, int] = {}
    for line in (path / "memory.events").read_text(encoding="ascii").splitlines():
        name, value = line.split()
        events[name] = int(value)
    swap = int((path / "memory.swap.current").read_text(encoding="ascii"))
    return path, events, swap


def _kernel_cursor() -> str:
    result = _run(["/usr/bin/journalctl", "-k", "-n", "0", "--show-cursor", "--no-pager"])
    matches = re.findall(r"^-- cursor: (\S+)$", result.stdout, flags=re.MULTILINE)
    if result.returncode != 0 or len(matches) != 1:
        raise CampaignError("cannot bind kernel fault cursor")
    return matches[0]


def _xids_after(cursor: str) -> int:
    result = _run(["/usr/bin/journalctl", "-k", f"--after-cursor={cursor}", "--no-pager"])
    if result.returncode != 0:
        raise CampaignError("cannot inspect post-arm kernel faults")
    return len(re.findall(r"\b(?:NVRM.*Xid|oom-kill|Out of memory: Killed process)\b", result.stdout, re.IGNORECASE))


def _global_swap_used() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and fields[0] in {"SwapTotal:", "SwapFree:"}:
            values[fields[0]] = int(fields[1]) * 1024
    return values["SwapTotal:"] - values["SwapFree:"]


def _wait_for_markers(pid: int, log_path: Path, shards: list[Path]) -> tuple[int, int, int, int, int, int]:
    deadline = time.monotonic() + 240
    load_start = load_end = 0
    max_read = max_direct = 0
    minimum_mem = 1 << 62
    initial_swap = _global_swap_used()
    maximum_swap_growth = 0
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            raise CampaignError("server died during load")
        now = time.monotonic_ns()
        try:
            text = log_path.read_text(encoding="utf-8", errors="strict")
            if not load_start and LOAD_START_MARKER in text:
                load_start = now
            if LOAD_END_MARKER in text:
                load_end = now
        except FileNotFoundError:
            pass
        try:
            io_fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
            max_read = max(max_read, int(io_fields["read_bytes"]))
            max_direct = max(max_direct, direct_shard_count(pid, shards))
        except FileNotFoundError:
            pass
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
        minimum_mem = min(minimum_mem, available)
        maximum_swap_growth = max(maximum_swap_growth, _global_swap_used() - initial_swap)
        try:
            _, _, current_cgroup_swap = _cgroup_counters(pid)
            maximum_swap_growth = max(maximum_swap_growth, current_cgroup_swap)
        except FileNotFoundError:
            pass
        if available < KILL_FLOOR_KB:
            raise CampaignError("whole-system memory kill floor crossed")
        if load_start and load_end:
            return load_start, load_end, max_read, max_direct, minimum_mem, maximum_swap_growth
        time.sleep(0.05)
    raise CampaignError("server load timed out")


def _server_command(binary: Path, library_dir: Path, first_shard: Path, arm: str, key: str) -> list[str]:
    return [
        "/usr/bin/env", "-i", "HOME=/home/bmarti44", "LANG=C.UTF-8",
        f"LD_LIBRARY_PATH={library_dir}", str(binary), "--model", str(first_shard),
        "--alias", "deepseek-v4-flash", "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "8192", "-np", "1", "-ngl", "999", "-b", "2048", "-ub", "512",
        "--no-warmup", "--cache-ram", "0", "--no-mmap", "--api-key", key,
        *direct_io_arguments(arm),
    ]


def _configuration_sha256() -> str:
    configuration = {
        "alias": "deepseek-v4-flash", "host": "127.0.0.1", "port": PORT,
        "context": 8192, "parallel": 1, "gpu_layers": 999, "batch": 2048,
        "ubatch": 512, "warmup": False, "cache_ram_mib": 0, "mmap": False,
        "completion": {"prompt": "Reply with exactly OK.", "n_predict": 1, "temperature": 0, "seed": 1, "n_probs": 10},
    }
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _run_arm(plan: ArmPlan, attempt: Path, binary: Path, library_dir: Path, shards: list[Path], bindings: dict[str, object]) -> dict[str, object]:
    arm_dir = attempt / plan.run_id
    arm_dir.mkdir(mode=0o700)
    require_fresh_output(arm_dir)
    cache_before = evict_and_measure(shards)
    log_path = arm_dir / "server.log"
    key = hashlib.sha256(f"{bindings['candidate_hash']}:{plan.run_id}".encode()).hexdigest()
    unit = f"cold-{plan.run_id}"
    cursor = _kernel_cursor()
    launch = time.monotonic_ns()
    command = containment_command(unit, _server_command(binary, library_dir, shards[0], plan.arm, key), log_path)
    started = _run(command, timeout=30)
    if started.returncode != 0:
        raise CampaignError(f"cannot start contained arm: {started.stderr.strip()}")
    pid = 0
    try:
        for _ in range(100):
            pid = _unit_pid(unit)
            if pid > 1:
                break
            time.sleep(0.05)
        if pid <= 1:
            raise CampaignError("contained server PID never appeared")
        start_ticks = _read_start_ticks(pid)
        cgroup_path, events_before, cgroup_swap_before = _cgroup_counters(pid)
        tensor_start, tensor_end, physical_read, direct_count, minimum_mem, swap_growth = _wait_for_markers(pid, log_path, shards)
        health_status = 0
        for _ in range(200):
            try:
                health_status, _ = _post(f"http://127.0.0.1:{PORT}/health", None, key)
            except urllib.error.URLError:
                time.sleep(0.05)
                continue
            if health_status == 200:
                break
            available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
            minimum_mem = min(minimum_mem, available)
            if available < KILL_FLOOR_KB:
                raise CampaignError("whole-system memory kill floor crossed")
            time.sleep(0.05)
        health_ready = time.monotonic_ns()
        unauth_status, _ = _post(f"http://127.0.0.1:{PORT}/health", None, None)
        request = {"prompt": "Reply with exactly OK.", "n_predict": 1, "temperature": 0, "seed": 1, "n_probs": 10}
        completion_status, completion = _post(f"http://127.0.0.1:{PORT}/completion", request, key)
        available_after_completion = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
        minimum_mem = min(minimum_mem, available_after_completion)
        if minimum_mem < KILL_FLOOR_KB:
            raise CampaignError("whole-system memory kill floor crossed")
        doc = json.loads(completion)
        semantic = json.dumps({"content": doc.get("content"), "tokens_predicted": doc.get("tokens_predicted")}, sort_keys=True, separators=(",", ":")).encode()
        probabilities = doc.get("completion_probabilities")
        logit = json.dumps(probabilities, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if not probabilities:
            raise CampaignError("completion omitted first-token probabilities")
        _, events_after, cgroup_swap_after = _cgroup_counters(pid)
        oom_delta = events_after.get("oom", 0) - events_before.get("oom", 0)
        oom_kill_delta = events_after.get("oom_kill", 0) - events_before.get("oom_kill", 0)
        max_delta = events_after.get("max", 0) - events_before.get("max", 0)
        swap_growth = max(swap_growth, cgroup_swap_after - cgroup_swap_before)
        xid_count = _xids_after(cursor)
        _unit_stop(unit)
        pid_survives = Path(f"/proc/{pid}").exists()
        survivors = max(int(pid_survives), _cgroup_survivors(cgroup_path))
        row = {
            "schema_version": 1, "block": plan.block, "position": plan.position, "arm": plan.arm,
            "run_id": plan.run_id, "candidate_hash": bindings["candidate_hash"],
            "model_sha256": bindings["model_sha256"], "configuration_sha256": bindings["configuration_sha256"],
            "runtime_bundle_sha256": bindings["runtime_bundle_sha256"], "process_launch_monotonic_ns": launch,
            "health_ready_monotonic_ns": health_ready, "tensor_load_start_monotonic_ns": tensor_start,
            "tensor_load_end_monotonic_ns": tensor_end, "server_pid": pid, "server_start_ticks": start_ticks,
            "server_fresh": True, "physical_read_bytes": physical_read, "cache_resident_bytes_before": cache_before,
            "direct_shard_count": direct_count, "direct_required": plan.arm == "on",
            "semantic_sha256": hashlib.sha256(semantic).hexdigest(), "first_token_logit_sha256": hashlib.sha256(logit).hexdigest(),
            "authenticated_health": health_status == 200, "authenticated_completion": completion_status == 200,
            "unauthenticated_rejected": unauth_status in {401, 403}, "minimum_mem_available_kb": minimum_mem,
            "swap_growth_bytes": max(0, swap_growth), "cgroup_oom_delta": oom_delta,
            "cgroup_oom_kill_delta": oom_kill_delta, "cgroup_max_delta": max_delta, "xid_count": xid_count,
            "surviving_descendants": survivors, "containment_rc": 0,
        }
        _write_json_new(arm_dir / "row.json", row)
        return row
    finally:
        _unit_stop(unit)


def _load_randomness(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or SHA256_RE.fullmatch(value.get("randomness", "")) is None:
        raise CampaignError("invalid public-randomness receipt")
    return value["randomness"], hashlib.sha256(raw).hexdigest()


def campaign(args: argparse.Namespace) -> int:
    global _ACTIVE_OUTPUT
    preflight_host()
    binary = args.binary.resolve(strict=True)
    library_dir = args.library_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _ACTIVE_OUTPUT = output
    randomness, receipt_sha = _load_randomness(args.randomness_receipt.resolve(strict=True))
    runner_sha = _sha256(Path(__file__))
    scorer_bytes = SCORER.read_bytes()
    scorer_sha = hashlib.sha256(scorer_bytes).hexdigest()
    if runner_sha != args.runner_sha256 or scorer_sha != args.scorer_sha256 or _sha256(binary) != args.binary_sha256:
        raise CampaignError("frozen runtime or harness digest mismatch")
    if _runtime_bundle_sha256(library_dir) != args.runtime_bundle_sha256:
        raise CampaignError("runtime bundle digest mismatch")
    if _configuration_sha256() != args.configuration_sha256:
        raise CampaignError("configuration digest mismatch")
    model_doc = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if _sha256(MODEL_MANIFEST) != args.model_sha256:
        raise CampaignError("model manifest digest mismatch")
    shards = [MODEL_DIR / item["name"] for item in model_doc["files"]]
    if sum(item["bytes"] for item in model_doc["files"]) != args.model_bytes:
        raise CampaignError("model byte count mismatch")
    for path, item in zip(shards, model_doc["files"], strict=True):
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise CampaignError("model shard identity mismatch")
    bindings = {
        "candidate_hash": args.candidate_hash, "model_sha256": args.model_sha256,
        "configuration_sha256": args.configuration_sha256, "runtime_bundle_sha256": args.runtime_bundle_sha256,
    }
    manifest = {
        "schema_version": 1, **bindings, "runner_sha256": runner_sha, "scorer_sha256": scorer_sha,
        "model_bytes": args.model_bytes, "randomness": {"value": randomness, "receipt_sha256": receipt_sha},
        "schedules": arm_schedule(randomness),
    }
    frozen_scorer = output / "frozen-scorer.py"
    _write_bytes_new(frozen_scorer, scorer_bytes)
    _write_json_new(output / "manifest.json", manifest)
    lock_fd = os.open(CAMPAIGN_LOCK, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for plan in campaign_plan(randomness):
            preflight_host()
            rows.append(_run_arm(plan, output, binary, library_dir, shards, bindings))
        raw_path = output / "raw.jsonl"
        descriptor = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            for row in rows:
                _write_all(descriptor, json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        scored = _run([
            "/usr/bin/python3", "-I", str(frozen_scorer), str(output / "manifest.json"),
            str(raw_path), str(output / "summary.json"),
        ], timeout=60)
        if scored.returncode != 0:
            raise CampaignError("fixed scorer rejected campaign")
    finally:
        os.close(lock_fd)
    return 0


def _preserve_failure(message: str) -> None:
    if _ACTIVE_OUTPUT is None or not _ACTIVE_OUTPUT.is_dir():
        return
    completed_rows: list[bytes] = []
    for path in sorted(_ACTIVE_OUTPUT.glob("b[0-4]-p[0-3]-*/row.json")):
        try:
            value = json.loads(path.read_bytes())
            completed_rows.append(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        except (OSError, ValueError, TypeError):
            completed_rows.append(json.dumps({"artifact": str(path), "failure": "malformed completed row"}, sort_keys=True).encode() + b"\n")
    for name, payload in (
        ("raw.jsonl", b"".join(completed_rows)),
        ("summary.json", (json.dumps({"failure": message, "verdict": "FAIL"}, sort_keys=True) + "\n").encode()),
    ):
        path = _ACTIVE_OUTPUT / name
        if path.exists():
            continue
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-hash", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--scorer-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-bytes", type=int, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--runtime-bundle-sha256", required=True)
    parser.add_argument("--randomness-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("candidate_hash", "binary_sha256", "runner_sha256", "scorer_sha256", "model_sha256", "configuration_sha256", "runtime_bundle_sha256"):
        if SHA256_RE.fullmatch(getattr(args, name)) is None:
            parser.error(f"{name} must be one lowercase SHA-256 value")
    try:
        return campaign(args)
    except (CampaignError, OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        _preserve_failure(f"{type(error).__name__}: {error}")
        print(f"dsv4 cold-load campaign: FAIL: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
