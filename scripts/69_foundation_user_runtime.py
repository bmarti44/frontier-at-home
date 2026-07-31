#!/usr/bin/env python3
"""Fixed runtime primitives for sudo-free matched foundation measurements.

This module deliberately contains no authority to start a model by itself.
The registered foundation runner invokes it inside an already-created user
systemd cgroup after freezing candidate and artifact identities.
"""

from __future__ import annotations

import hashlib
import math
import os
import signal
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


GIB = 1 << 30
ROOT = Path(__file__).resolve().parents[1]
MEMWATCH = ROOT / "scripts" / "01_memwatch.sh"
CGROUP_LIMITS = {
    "dsv4": (105 * GIB, 110 * GIB, 0),
    "glm52": (68 * GIB, 72 * GIB, 0),
}


def _bounded_port(port: int) -> str:
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("port must be an integer from 1024 through 65535")
    return str(port)


def server_invocation(
    profile: str, binary: Path, model: Path, port: int
) -> tuple[list[str], dict[str, str]]:
    """Return the only approved server command/environment for each profile."""
    port_text = _bounded_port(port)
    binary_text = str(binary)
    model_text = str(model)
    if profile == "dsv4":
        return (
            [
                binary_text,
                "--model",
                model_text,
                "--alias",
                "deepseek-v4-flash",
                "--host",
                "127.0.0.1",
                "--port",
                port_text,
                "-c",
                "1048576",
                "-np",
                "1",
                "-ngl",
                "999",
                "-b",
                "512",
                "-ub",
                "256",
                "--no-warmup",
                "--cache-ram",
                "0",
                "--no-mmap",
            ],
            {},
        )
    if profile == "glm52":
        return (
            [
                binary_text,
                "--cuda",
                "-m",
                model_text,
                "-c",
                "8192",
                "--host",
                "127.0.0.1",
                "--port",
                port_text,
                "--ssd-streaming",
                "--ssd-streaming-cache-experts",
                "40GB",
            ],
            {
                "DS4_CUDA_EXPERT_CACHE_GB": "40",
                "DS4_CUDA_EXPERT_CACHE_PIN": "1",
                "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
                "DS4_CUDA_FETCH_THREADS": "6",
                "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
                "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
                "DS4_GLM_COMPACT_CACHE_AFFINE_INT8": "1",
                "DS4_TOKEN_TIMING_LOG": "1",
            },
        )
    raise ValueError("profile must be dsv4 or glm52")


def benchmark_invocation(
    profile: str,
    result_path: Path,
    port: int,
    seed: int,
    tokenizer_path: Path | None,
    tokenizer_sha256: str | None,
) -> list[str]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("benchmark seed must be an unsigned 32-bit integer")
    port_text = _bounded_port(port)
    if profile not in {"dsv4", "glm52"}:
        raise ValueError("unknown benchmark profile")
    command = [
        str(ROOT / ".venv-harness" / "bin" / "python"),
        str(ROOT / "scripts" / "30_bench_speed.py"),
        "--base-url",
        f"http://127.0.0.1:{port_text}",
        "--out",
        str(result_path),
        "--stack-label",
        f"foundation-{profile}",
        "--model-id",
        "deepseek-v4-flash" if profile == "dsv4" else "glm-5.2",
        "--reps",
        "2",
        "--context-levels",
        "0",
        "--max-tokens",
        "160",
        "--min-completion-tokens",
        "128",
        "--seed",
        str(seed),
    ]
    if profile == "dsv4":
        command.append("--ignore-eos-supported")
    else:
        if tokenizer_path is None or tokenizer_sha256 is None:
            raise ValueError("GLM benchmark tokenizer identity is required")
        command.extend(
            [
                "--output-tokenizer-path",
                str(tokenizer_path),
                "--output-tokenizer-sha256",
                tokenizer_sha256,
                "--token-timing-log",
                str(result_path.parent / "server.log"),
            ]
        )
    return command


def validate_cgroup(
    profile: str, memory_high: int, memory_max: int, swap_max: int
) -> None:
    expected = CGROUP_LIMITS.get(profile)
    if expected is None:
        raise ValueError("unknown cgroup profile")
    values = (memory_high, memory_max, swap_max)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("cgroup values must be integers")
    if values != expected:
        raise ValueError(
            f"unsafe {profile} cgroup limits: observed={values} expected={expected}"
        )


def read_cgroup_limits(cgroup: Path) -> tuple[int, int, int]:
    values = []
    for name in ("memory.high", "memory.max", "memory.swap.max"):
        try:
            raw = (cgroup / name).read_text(encoding="ascii").strip()
            value = int(raw)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cgroup {name} is absent or non-finite") from exc
        if value < 0:
            raise ValueError(f"cgroup {name} is negative")
        values.append(value)
    return values[0], values[1], values[2]


def current_cgroup_path() -> Path:
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ValueError("cannot inspect current cgroup") from exc
    matches = [row.split(":", 2)[2] for row in rows if row.startswith("0::")]
    if len(matches) != 1 or not matches[0].startswith("/"):
        raise ValueError("current unified cgroup is invalid")
    path = Path("/sys/fs/cgroup") / matches[0].lstrip("/")
    if not path.is_dir() or path.is_symlink():
        raise ValueError("current cgroup directory is unsafe")
    return path


def verify_artifact(path: Path, expected_sha256: str) -> None:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("artifact digest is invalid")
    try:
        int(expected_sha256, 16)
        details = path.lstat()
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError("artifact identity is invalid") from exc
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise ValueError("artifact must be a plain file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"artifact hash mismatch: {path}")


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not finite and positive")
    return result


def baseline_from_result(
    result: dict[str, Any],
    *,
    profile: str,
    server_instance_id: str,
    fixture_sha256: str,
    binary_sha256: str,
    configuration_sha256: str,
    available_memory_gib: float,
) -> dict[str, Any]:
    """Convert one fixed two-repetition speed result into foundation raw data."""
    if profile not in {"dsv4", "glm52"} or result.get("suite_valid") is not True:
        raise ValueError("benchmark identity or suite validity is invalid")
    metadata = result.get("metadata")
    cells = result.get("cells")
    if (
        not isinstance(metadata, dict)
        or metadata.get("reps") != 2
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], dict)
        or cells[0].get("ctx_tokens") != 0
        or cells[0].get("valid") is not True
    ):
        raise ValueError("benchmark shape is invalid")
    reps = cells[0].get("reps")
    if (
        not isinstance(reps, list)
        or len(reps) != 2
        or any(not isinstance(rep, dict) or rep.get("valid") is not True for rep in reps)
    ):
        raise ValueError("cold/warm repetitions are invalid")
    cold, warm = reps
    timestamps = warm.get("token_timestamps_ns")
    if (
        not isinstance(timestamps, list)
        or len(timestamps) < 128
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in timestamps)
        or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError("raw token timestamps are invalid")
    prompt_tokens = warm.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError("evaluated prompt token count is invalid")
    prefill_rate = _positive_number(warm.get("prefill_tok_s"), "prefill rate")
    memory = _positive_number(available_memory_gib, "available memory")
    if memory < 10.0:
        raise ValueError("foundation memory floor was violated")
    identities = (server_instance_id, fixture_sha256, binary_sha256, configuration_sha256)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in identities
    ):
        raise ValueError("foundation identity hash is invalid")
    return {
        "profile": profile,
        "server_instance_id": server_instance_id,
        "fixture_sha256": fixture_sha256,
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "token_timestamps": [value / 1_000_000_000 for value in timestamps],
        "evaluated_tokens": prompt_tokens,
        "prefill_seconds": prompt_tokens / prefill_rate,
        "warm_ttft_seconds": _positive_number(warm.get("ttft_s"), "warm TTFT"),
        "cold_ttft_seconds": _positive_number(cold.get("ttft_s"), "cold TTFT"),
        "available_memory_gib": memory,
        "truncated": False,
        "oom": False,
        "xid": False,
        "failures": [],
    }


def _proc_start_ticks(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split(") ", 1)[1].split()
        value = int(fields[19])
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("cannot read server process identity") from exc
    if value <= 0:
        raise RuntimeError("server process identity is invalid")
    return value


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1_048_576
    raise RuntimeError("MemAvailable is absent")


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait(timeout=10)


def supervise_process(
    command: list[str],
    environment: dict[str, str],
    out: Path,
    *,
    port: int,
    watchdog_floor_gib: int,
    startup_timeout_seconds: int,
    probe_command: list[str],
) -> dict[str, Any]:
    """Run one server and probe under the independent, identity-bound watchdog."""
    _bounded_port(port)
    if (
        not command
        or not probe_command
        or isinstance(watchdog_floor_gib, bool)
        or not isinstance(watchdog_floor_gib, int)
        or not 1 <= watchdog_floor_gib <= 64
        or isinstance(startup_timeout_seconds, bool)
        or not isinstance(startup_timeout_seconds, int)
        or not 1 <= startup_timeout_seconds <= 1800
    ):
        raise ValueError("foundation supervision arguments are invalid")
    if out.exists() or out.is_symlink():
        raise ValueError("foundation arm output already exists")
    out.mkdir(mode=0o700, parents=True)
    target = out / "memwatch.target"
    ready = out / "memwatch.ready"
    memwatch_log = out / "memwatch.log"
    memwatch_stderr_path = out / "memwatch.stderr.log"
    server_log_path = out / "server.log"
    samples = [_mem_available_gib()]
    server: subprocess.Popen[Any] | None = None
    memwatch: subprocess.Popen[Any] | None = None
    armed = False
    success = False
    server_pid = 0
    server_instance_id = ""
    with memwatch_stderr_path.open("xb") as memwatch_stderr, server_log_path.open("xb") as server_log:
        try:
            memwatch = subprocess.Popen(
                [
                    str(MEMWATCH),
                    "--target-file",
                    str(target),
                    "--ready-file",
                    str(ready),
                    "--threshold-gib",
                    str(watchdog_floor_gib),
                    "--interval-sec",
                    "0.25",
                    "--log",
                    str(memwatch_log),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=memwatch_stderr,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not ready.is_file():
                if memwatch.poll() is not None:
                    raise RuntimeError("memory watchdog exited before readiness")
                time.sleep(0.05)
            if not ready.is_file() or ready.read_text(encoding="ascii").strip() != "READY":
                raise RuntimeError("memory watchdog did not become ready")

            child_env = {
                "HOME": os.environ.get("HOME", "/home/bmarti44"),
                "LANG": "C.UTF-8",
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                **environment,
            }
            server = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )
            server_pid = server.pid
            pgid = os.getpgid(server.pid)
            ticks = _proc_start_ticks(server.pid)
            if pgid != server.pid:
                raise RuntimeError("server process group is not isolated")
            target.write_text(f"{server.pid} {pgid} {ticks} engine\n", encoding="ascii")
            armed = True
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            server_instance_id = hashlib.sha256(
                f"{boot_id}:{server.pid}:{ticks}".encode()
            ).hexdigest()

            deadline = time.monotonic() + startup_timeout_seconds
            health_url = f"http://127.0.0.1:{port}/v1/models"
            while time.monotonic() < deadline:
                samples.append(_mem_available_gib())
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup with status {server.returncode}")
                if memwatch.poll() is not None:
                    raise RuntimeError("memory watchdog exited while server was armed")
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("server startup timed out")

            completed = subprocess.run(
                probe_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=3600,
                env=child_env,
            )
            samples.append(_mem_available_gib())
            (out / "probe.stdout.log").write_bytes(completed.stdout)
            (out / "probe.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"foundation probe failed with status {completed.returncode}")

            target.write_text(
                f"DISARM {server.pid} {pgid} {ticks}\n", encoding="ascii"
            )
            memwatch.wait(timeout=10)
            if memwatch.returncode != 0:
                raise RuntimeError("memory watchdog did not disarm cleanly")
            armed = False
            _terminate_group(server)
            success = True
        finally:
            if not success:
                if memwatch is not None and memwatch.poll() is None:
                    memwatch.send_signal(signal.SIGTERM)
                    try:
                        memwatch.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        memwatch.kill()
                        memwatch.wait()
                if server is not None:
                    _terminate_group(server)
            elif server is not None and server.poll() is None:
                _terminate_group(server)
            if armed and memwatch is not None and memwatch.poll() is None:
                memwatch.kill()
                memwatch.wait()
    return {
        "server_instance_id": server_instance_id,
        "server_pid": server_pid,
        "available_memory_gib": min(samples),
    }


if __name__ == "__main__":
    raise SystemExit("69_foundation_user_runtime.py is a library; use its registered runner")
