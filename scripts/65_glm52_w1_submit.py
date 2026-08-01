#!/usr/bin/env python3
"""Root-owned, narrowly delegated authority for the fixed GLM W1 campaign."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import re
import socket
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path("/home/bmarti44/spark-deepseek-v4-flash")
ENGINE_REPOSITORY = Path("/home/bmarti44/.cache/glm52-w1-real-capture-a37")
MODEL = Path(
    "/home/dsv4/ds4-project/gguf-glm/"
    "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
)
STATE_ROOT = Path("/var/lib/glm52-w1")
CONTROLLER_ATTEMPTS = STATE_ROOT / "controller-attempts"
INSTALLED_HARNESS = Path("/usr/local/libexec/glm52-w1/harness")
LOCK = Path("/run/lock/glm52-w1-submit.lock")
INFERENCE_LOCK = Path("/run/lock/frontier-at-home/inference.lock")
LEGACY_INFERENCE_LOCK = Path("/run/dsv4/inference.lock")
RUNG0_MAINTENANCE_STATE = Path("/run/glm52-rung0-maintenance.json")
RUNG0_AUTO_RESTORE_UNIT = "glm52-rung0-auto-restore"
RUNG0_LOCK_HOLDER_UNIT = "glm52-rung0-lock-holder"
DSV4_SERVICE = "deepseek-v4-flash-llamacpp.service"
DSV4_GUARD_TIMER = "dsv4-guard.timer"
RUNNER = Path("scripts/glm52-runners/W1")
PROFILE = Path("configs/glm52-profile.json")
ROOT_EXECUTION_SURFACE = (
    "scripts/65_glm52_w1_submit.py",
    "scripts/glm52_w1_affine_campaign.py",
    "scripts/glm52_goal.py",
    "scripts/03_memory_guard.py",
    "results/glm52-gates/harness/glm_cgroup_run.sh",
    "results/glm52-gates/harness/glm_safe_run.sh",
)
OWNER = "bmarti44"
OWNER_UID = 1000
DSV4 = "dsv4"
HASH40 = re.compile(r"^[0-9a-f]{40}$")
HASH64 = re.compile(r"^[0-9a-f]{64}$")
MAX_RECEIPT_FILES = 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024 * 1024
MAX_DIAGNOSTIC_JSON_BYTES = 64 * 1024
ROOT_UID = 0
ROOT_GID = 0
ACTIVE_REQUEST: dict[str, Any] | None = None


def parse_request(argv: list[str]) -> tuple[str, ...]:
    if (
        len(argv) == 2
        and argv[0] in {"rung0-maintenance-begin", "rung0-maintenance-end"}
        and HASH40.fullmatch(argv[1])
    ):
        return tuple(argv)
    if (
        len(argv) == 4
        and argv[0] == "run"
        and HASH40.fullmatch(argv[1])
        and HASH40.fullmatch(argv[2])
        and HASH64.fullmatch(argv[3])
    ):
        return tuple(argv)
    if (
        len(argv) == 2
        and argv[0] in {"status", "diagnose"}
        and HASH64.fullmatch(argv[1])
    ):
        return tuple(argv)
    raise ValueError(
        "usage: glm52-w1-submit run HARNESS_COMMIT ENGINE_COMMIT MODEL_SHA256\n"
        "       glm52-w1-submit status COMPOSITE_SHA256\n"
        "       glm52-w1-submit diagnose COMPOSITE_SHA256\n"
        "       glm52-w1-submit rung0-maintenance-{begin,end} CANDIDATE_HASH"
    )


def receipt_exit_code(receipt: dict[str, Any]) -> int:
    return (
        0
        if receipt.get("terminal_state") == "PASS"
        and receipt.get("service_returncode") == 0
        else 1
    )


def _run(
    argv: list[str],
    *,
    check: bool = True,
    timeout: int = 60,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
        env=environment
        or {
            "HOME": "/nonexistent",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {argv[0]}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed


def _git_as_owner(repository: Path, *arguments: str) -> str:
    completed = _run(
        [
            "/usr/sbin/runuser",
            "-u",
            OWNER,
            "--",
            "/usr/bin/env",
            "-i",
            "HOME=/nonexistent",
            "PATH=/usr/bin:/bin",
            "LANG=C.UTF-8",
            "GIT_CONFIG_NOSYSTEM=1",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ]
    )
    return completed.stdout.strip()


def _git_root(repository: Path, *arguments: str) -> str:
    return _run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ]
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(path: Path, value: object, mode: int = 0o400) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _assert_root() -> None:
    if os.geteuid() != 0:
        raise PermissionError("must run as root")
    if os.environ.get("SUDO_USER") != OWNER or os.environ.get("SUDO_UID") != "1000":
        raise PermissionError("must be delegated by bmarti44 through sudo")
    if pwd.getpwnam(OWNER).pw_uid != OWNER_UID:
        raise PermissionError("benchmark owner identity changed")


def _unit_active(name: str) -> bool:
    return (
        _run(
            ["/usr/bin/systemctl", "is-active", "--quiet", name],
            check=False,
        ).returncode
        == 0
    )


def _engine_processes() -> list[dict[str, object]]:
    processes: list[dict[str, object]] = []
    dsv4_uid = pwd.getpwnam(DSV4).pw_uid
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            name = (entry / "comm").read_text(encoding="utf-8").strip()
            if name not in {"llama-server", "ds4-server"}:
                continue
            status = (entry / "status").read_text(encoding="utf-8")
            uid = int(status.split("Uid:", 1)[1].split()[0])
            command = (entry / "cmdline").read_bytes().split(b"\0")
            arguments = [value.decode("utf-8", "strict") for value in command if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                UnicodeError, ValueError):
            continue
        processes.append(
            {"pid": int(entry.name), "name": name, "uid": uid,
             "is_dsv4": uid == dsv4_uid, "arguments": arguments}
        )
    return processes


def _wait_for_engine_exit(timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _engine_processes():
            return
        time.sleep(0.5)
    raise RuntimeError("inference engine survived maintenance stop")


def _dsv4_command_is_1m(arguments: list[str]) -> bool:
    try:
        context_index = arguments.index("-c")
        context_ok = arguments[context_index + 1] == "1048576"
    except (ValueError, IndexError):
        context_ok = False
    try:
        alias_index = arguments.index("--alias")
        alias_ok = arguments[alias_index + 1] == "deepseek-v4-flash"
    except (ValueError, IndexError):
        alias_ok = False
    return context_ok and alias_ok


def _verify_dsv4_restored(timeout: float = 240.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_health = "not attempted"
    while time.monotonic() < deadline:
        engines = _engine_processes()
        if (
            len(engines) == 1
            and engines[0]["name"] == "llama-server"
            and engines[0]["is_dsv4"] is True
            and _dsv4_command_is_1m(engines[0]["arguments"])
        ):
            health = _run(
                ["/usr/bin/curl", "-fsS", "--max-time", "5",
                 "http://127.0.0.1:8013/health"],
                check=False,
                timeout=10,
            )
            last_health = health.stdout[-1000:]
            if health.returncode == 0:
                return engines[0]
        time.sleep(1)
    raise RuntimeError(f"DSV4 1M restoration failed: {last_health}")


def _cancel_rung0_auto_restore() -> None:
    _run(
        ["/usr/bin/systemctl", "stop",
         f"{RUNG0_AUTO_RESTORE_UNIT}.timer",
         f"{RUNG0_AUTO_RESTORE_UNIT}.service"],
        check=False,
    )
    _run(
        ["/usr/bin/systemctl", "reset-failed",
         f"{RUNG0_AUTO_RESTORE_UNIT}.service"],
        check=False,
    )


def _stop_rung0_lock_holder() -> None:
    _run(
        ["/usr/bin/systemctl", "stop", f"{RUNG0_LOCK_HOLDER_UNIT}.service"],
        check=False,
    )


def _start_rung0_lock_holder() -> None:
    _stop_rung0_lock_holder()
    _run(
        [
            "/usr/bin/systemd-run", f"--unit={RUNG0_LOCK_HOLDER_UNIT}",
            "--service-type=exec", "--property=KillMode=control-group",
            "/usr/bin/flock", "-n", "-E", "75", str(INFERENCE_LOCK),
            "/usr/bin/flock", "-n", "-E", "75", str(LEGACY_INFERENCE_LOCK),
            "/usr/bin/sleep", "43200",
        ],
        timeout=30,
    )
    if not _unit_active(f"{RUNG0_LOCK_HOLDER_UNIT}.service"):
        raise RuntimeError("Rung 0 inference lock holder failed to start")


def _schedule_rung0_auto_restore() -> None:
    _cancel_rung0_auto_restore()
    _run(
        [
            "/usr/bin/systemd-run",
            f"--unit={RUNG0_AUTO_RESTORE_UNIT}",
            "--on-active=12h",
            "--property=Type=oneshot",
            "--property=ExecStartPre=-/usr/bin/pkill -KILL -x ds4-server",
            f"--property=ExecStartPre=/usr/bin/systemctl stop {RUNG0_LOCK_HOLDER_UNIT}.service",
            "/usr/bin/systemctl", "start", DSV4_SERVICE,
        ],
        timeout=30,
    )


def _rung0_maintenance_begin(candidate: str) -> int:
    _assert_docker_closed()
    _assert_repository(REPOSITORY, candidate, reject_untracked=True)
    if RUNG0_MAINTENANCE_STATE.exists():
        state = json.loads(RUNG0_MAINTENANCE_STATE.read_text(encoding="utf-8"))
        if (state.get("candidate_hash") == candidate and
                not _engine_processes() and
                _unit_active(f"{RUNG0_LOCK_HOLDER_UNIT}.service")):
            print(json.dumps(state, sort_keys=True, separators=(",", ":")))
            return 0
        raise RuntimeError("another GLM maintenance window is already recorded")
    if not _unit_active(DSV4_SERVICE):
        raise RuntimeError("DSV4 service is not active before maintenance")
    _verify_dsv4_restored(timeout=30)
    stopped = False
    try:
        _run(["/usr/bin/systemctl", "stop", DSV4_GUARD_TIMER])
        stopped = True
        _run(["/usr/bin/systemctl", "stop", DSV4_SERVICE], timeout=180)
        _wait_for_engine_exit()
        with _hold_inference_locks():
            pass
        _start_rung0_lock_holder()
        _schedule_rung0_auto_restore()
        state = {
            "schema_version": 1,
            "state": "MAINTENANCE",
            "candidate_hash": candidate,
            "previous_profile": "dsv4-1m-fast",
            "auto_restore_unit": f"{RUNG0_AUTO_RESTORE_UNIT}.timer",
            "opened_at_epoch": int(time.time()),
        }
        _canonical_json(RUNG0_MAINTENANCE_STATE, state, mode=0o600)
        print(json.dumps(state, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception:
        _cancel_rung0_auto_restore()
        _stop_rung0_lock_holder()
        if stopped:
            _run(["/usr/bin/systemctl", "start", DSV4_SERVICE], check=False,
                 timeout=180)
        _run(["/usr/bin/systemctl", "start", DSV4_GUARD_TIMER], check=False)
        raise


def _rung0_maintenance_end(candidate: str) -> int:
    _assert_repository(REPOSITORY, candidate, reject_untracked=True)
    if not RUNG0_MAINTENANCE_STATE.is_file():
        restored = _verify_dsv4_restored(timeout=30)
        print(json.dumps({"state": "RESTORED", "engine": restored},
                         sort_keys=True, separators=(",", ":")))
        return 0
    state = json.loads(RUNG0_MAINTENANCE_STATE.read_text(encoding="utf-8"))
    if state.get("candidate_hash") != candidate:
        raise ValueError("maintenance candidate differs from recorded window")
    engines = _engine_processes()
    already_restored = False
    if engines:
        _verify_dsv4_restored(timeout=30)
        already_restored = True
    _cancel_rung0_auto_restore()
    _stop_rung0_lock_holder()
    if not already_restored:
        _run(["/usr/bin/systemctl", "start", DSV4_SERVICE], timeout=180)
    restored = _verify_dsv4_restored()
    _run(["/usr/bin/systemctl", "start", DSV4_GUARD_TIMER])
    RUNG0_MAINTENANCE_STATE.unlink()
    print(json.dumps({"state": "RESTORED", "engine": restored},
                     sort_keys=True, separators=(",", ":")))
    return 0


def _assert_docker_closed() -> None:
    group_line = Path("/etc/group").read_text(encoding="utf-8")
    docker = next(
        (line for line in group_line.splitlines() if line.startswith("docker:")),
        "",
    )
    members = docker.rsplit(":", 1)[-1].split(",") if docker else []
    if OWNER in members:
        raise PermissionError("bmarti44 remains in the root-equivalent docker group")
    if _unit_active("docker.socket") or _unit_active("docker.service"):
        raise PermissionError("root-equivalent Docker service remains active")
    for process in Path("/proc").glob("[0-9]*"):
        try:
            command = (process / "comm").read_text(encoding="utf-8").strip()
            uid = (
                (process / "status")
                .read_text(encoding="utf-8")
                .split("Uid:", 1)[1]
                .split()[0]
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if command in {"dockerd", "containerd"} and uid == "0":
            raise PermissionError("a root container runtime remains active")
    docker_socket = Path("/var/run/docker.sock")
    if docker_socket.exists():
        mode = docker_socket.stat().st_mode
        if stat.S_ISSOCK(mode):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.2)
                if probe.connect_ex(str(docker_socket)) == 0:
                    raise PermissionError("a Docker-compatible socket remains live")
            finally:
                probe.close()
    docker_gid = next(
        (
            int(line.split(":")[2])
            for line in group_line.splitlines()
            if line.startswith("docker:")
        ),
        -1,
    )
    caller_pid = int(os.environ.get("SUDO_PID", "0"))
    if caller_pid > 1:
        status_path = Path(f"/proc/{caller_pid}/status")
        if status_path.is_file():
            groups_line = next(
                (
                    line
                    for line in status_path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("Groups:")
                ),
                "",
            )
            groups = {int(value) for value in groups_line.split()[1:]}
            if docker_gid in groups:
                raise PermissionError(
                    "the invoking session retains the root-equivalent docker group"
                )


def _assert_repository(
    repository: Path,
    expected_commit: str,
    *,
    reject_untracked: bool,
    required_path: Path | None = None,
) -> None:
    if not repository.is_dir() or repository.is_symlink():
        raise ValueError(f"repository is absent or unsafe: {repository}")
    actual = _git_as_owner(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if actual != expected_commit:
        raise ValueError(f"repository HEAD differs: {repository}")
    untracked = "all" if reject_untracked else "no"
    if _git_as_owner(
        repository, "status", "--porcelain", f"--untracked-files={untracked}"
    ):
        raise ValueError(f"repository tracked content is not clean: {repository}")
    if required_path is not None:
        tracked = _git_as_owner(
            repository, "ls-files", "--error-unmatch", str(required_path)
        )
        if tracked != str(required_path):
            raise ValueError(f"required runner is not tracked: {required_path}")


def _model_hash_from_profile(harness: str) -> str:
    raw = _git_as_owner(REPOSITORY, "show", f"{harness}:{PROFILE}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"profile contains duplicate key {key!r}")
            result[key] = value
        return result

    profile = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    value = profile.get("model_sha256")
    if not isinstance(value, str) or not HASH64.fullmatch(value):
        raise ValueError("profile model hash is invalid")
    return value


def _assert_root_execution_surface(candidate: Path) -> None:
    for relative in ROOT_EXECUTION_SURFACE:
        installed = INSTALLED_HARNESS / relative
        requested = candidate / relative
        try:
            installed_details = installed.lstat()
            requested_details = requested.lstat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"root execution surface is absent: {relative}"
            ) from exc
        if (
            not stat.S_ISREG(installed_details.st_mode)
            or not stat.S_ISREG(requested_details.st_mode)
            or installed.is_symlink()
            or requested.is_symlink()
        ):
            raise ValueError(f"root execution surface is unsafe: {relative}")
        if _sha256(installed) != _sha256(requested):
            raise ValueError(
                f"root execution surface differs: {relative}"
            )


def _request_id(harness: str, engine: str, model: str) -> str:
    return hashlib.sha256(f"{harness}:{engine}:{model}:W1-root-v3".encode()).hexdigest()


def first_drand_round_after(frozen_at: str) -> int:
    """Return the first default-chain round published strictly after freeze."""
    try:
        frozen = datetime.fromisoformat(frozen_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("freeze timestamp is invalid") from exc
    if frozen.tzinfo is None:
        raise ValueError("freeze timestamp lacks a timezone")
    frozen = frozen.astimezone(timezone.utc)
    genesis = datetime.fromtimestamp(1_595_431_050, timezone.utc)
    if frozen < genesis:
        return 1
    return (frozen - genesis) // timedelta(seconds=30) + 2


def _fetch_first_post_freeze_drand(frozen_at: str) -> str:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate drand response key: {key}")
            value[key] = item
        return value

    round_number = first_drand_round_after(frozen_at)
    deadline = time.monotonic() + 45
    while True:
        result = _run(
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
                f"https://api.drand.sh/public/{round_number}",
            ],
            check=False,
            timeout=15,
        )
        if result.returncode == 0:
            try:
                record = json.loads(
                    result.stdout,
                    object_pairs_hook=unique_pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite drand value: {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError("post-freeze drand response is invalid") from exc
            if isinstance(record, dict) and record.get("round") == round_number:
                return result.stdout
            raise RuntimeError("post-freeze drand round differs from request")
        if time.monotonic() >= deadline:
            raise RuntimeError("post-freeze public randomness fetch failed")
        time.sleep(2)


def _tree_manifest(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"evidence contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError(f"evidence file type or link count is unsafe: {relative}")
        total += details.st_size
        if len(rows) >= MAX_RECEIPT_FILES or total > MAX_RECEIPT_BYTES:
            raise ValueError("evidence file-count or byte limit exceeded")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size": details.st_size,
            }
        )
    if not rows:
        raise ValueError("evidence tree is empty")
    return rows


def _manifest_sha256(rows: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _open_noatime(path: Path, flags: int, *, dir_fd: int | None = None) -> int:
    return os.open(
        path if dir_fd is None else path.name,
        flags | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NOATIME,
        dir_fd=dir_fd,
    )


def _read_regular_noatime(path: Path, maximum: int) -> bytes:
    descriptor = _open_noatime(path, os.O_RDONLY)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_uid != ROOT_UID
            or details.st_gid != ROOT_GID
            or stat.S_IMODE(details.st_mode) != 0o400
            or details.st_size > maximum
        ):
            raise ValueError("diagnostic input is unsafe")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        value = b"".join(chunks)
        if len(value) > maximum:
            raise ValueError("diagnostic input is unexpectedly large")
        return value
    finally:
        os.close(descriptor)


def _listdir_noatime(path: Path) -> list[str]:
    descriptor = _open_noatime(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return sorted(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def _diagnostic_input_manifest(root: Path) -> list[dict[str, object]]:
    """Snapshot only the fixed identity and error inputs, without atime writes."""
    rows: list[dict[str, object]] = []
    inputs = (
        (Path("."), "directory", 0),
        (Path("freeze"), "directory", 0),
        (Path("receipt.json"), "file", MAX_DIAGNOSTIC_JSON_BYTES),
        (Path("request.json"), "file", MAX_DIAGNOSTIC_JSON_BYTES),
        (Path("freeze/freeze.json"), "file", MAX_DIAGNOSTIC_JSON_BYTES),
        (Path("campaign.log"), "file", 16 * 1024 * 1024),
    )
    for relative, kind, maximum in inputs:
        path = root if relative == Path(".") else root / relative
        flags = os.O_RDONLY | (os.O_DIRECTORY if kind == "directory" else 0)
        descriptor = _open_noatime(path, flags)
        try:
            details = os.fstat(descriptor)
            expected_mode = 0o500 if kind == "directory" else 0o400
            expected_type = (
                stat.S_ISDIR(details.st_mode)
                if kind == "directory"
                else stat.S_ISREG(details.st_mode) and details.st_nlink == 1
            )
            if (
                not expected_type
                or details.st_uid != ROOT_UID
                or details.st_gid != ROOT_GID
                or stat.S_IMODE(details.st_mode) != expected_mode
                or (maximum and details.st_size > maximum)
            ):
                raise ValueError("failed campaign diagnostic input is unsafe")
            row: dict[str, object] = {
                "path": relative.as_posix(),
                "type": kind,
                "device": details.st_dev,
                "inode": details.st_ino,
                "links": details.st_nlink,
                "mode": stat.S_IMODE(details.st_mode),
                "uid": details.st_uid,
                "gid": details.st_gid,
                "atime_ns": details.st_atime_ns,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
                "size": details.st_size,
            }
            if kind == "file":
                digest = hashlib.sha256()
                while block := os.read(descriptor, 1024 * 1024):
                    digest.update(block)
                row["sha256"] = digest.hexdigest()
            rows.append(row)
        finally:
            os.close(descriptor)
    return rows


def _seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("refusing to seal a symlink")
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o500 if path.is_dir() else 0o400, follow_symlinks=False)
    os.chown(root, 0, 0)
    os.chmod(root, 0o500)


def _quarantine_seal(root: Path) -> None:
    """Revoke all non-root writes without following attacker-created links."""
    for path in sorted(root.rglob("*"), reverse=True):
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            continue
        if stat.S_ISDIR(details.st_mode):
            os.chown(path, 0, 0, follow_symlinks=False)
            os.chmod(path, 0o500, follow_symlinks=False)
        elif stat.S_ISREG(details.st_mode) and details.st_nlink == 1:
            os.chown(path, 0, 0, follow_symlinks=False)
            os.chmod(path, 0o400, follow_symlinks=False)
    os.chown(root, 0, 0)
    os.chmod(root, 0o500)


def _record_failed_active_request(exc: Exception) -> None:
    active = ACTIVE_REQUEST
    if active is None:
        return
    root = Path(active["root"])
    receipt_path = root / "receipt.json"
    if not receipt_path.exists():
        receipt = {
            "schema_version": 2,
            "terminal_state": "FAIL",
            "service_returncode": 1,
            "failure_phase": str(active["phase"]),
            "failure_type": type(exc).__name__,
            "request_id": str(active["request_id"]),
        }
        _canonical_json(receipt_path, receipt)
    _quarantine_seal(root)


def _validate_current_lock_parent(lock_root: Path, gid: int) -> None:
    parent = lock_root.lstat()
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_gid != gid
        or stat.S_IMODE(parent.st_mode) != 0o750
    ):
        raise PermissionError("current inference lock directory is unsafe")


def _open_one_inference_lock(path: Path, *, stable_parent: bool):
    identity = pwd.getpwnam(DSV4)
    if stable_parent:
        _validate_current_lock_parent(path.parent, identity.pw_gid)
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o660,
    )
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        os.close(descriptor)
        raise PermissionError("inference lock is not a private regular file")
    os.fchown(descriptor, 0, identity.pw_gid)
    os.fchmod(descriptor, 0o660)
    return os.fdopen(descriptor, "a+b")


def _validate_legacy_lock_namespace() -> None:
    identity = pwd.getpwnam(DSV4)
    parent = LEGACY_INFERENCE_LOCK.parent.lstat()
    details = LEGACY_INFERENCE_LOCK.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_gid != identity.pw_gid
        or stat.S_IMODE(parent.st_mode) != 0o1770
        or stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_uid != 0
        or details.st_gid != identity.pw_gid
        or details.st_nlink != 1
    ):
        raise PermissionError("legacy inference lock namespace is unsafe")


@contextmanager
def _hold_inference_locks():
    _validate_legacy_lock_namespace()
    with _open_one_inference_lock(
        LEGACY_INFERENCE_LOCK, stable_parent=False
    ) as legacy:
        try:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError(
                "a pre-migration inference server still holds the legacy lock"
            ) from exc
        with _open_one_inference_lock(
            INFERENCE_LOCK, stable_parent=True
        ) as current:
            legacy_identity = os.fstat(legacy.fileno())
            current_identity = os.fstat(current.fileno())
            same_inode = (
                legacy_identity.st_dev == current_identity.st_dev
                and legacy_identity.st_ino == current_identity.st_ino
            )
            if not same_inode:
                fcntl.flock(current, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if not same_inode:
                    fcntl.flock(current, fcntl.LOCK_UN)
                fcntl.flock(legacy, fcntl.LOCK_UN)


def _bundle_clone(
    repository: Path,
    commit: str,
    bundle: Path,
    destination: Path,
) -> None:
    owner_git = [
        "/usr/sbin/runuser",
        "-u",
        OWNER,
        "--",
        "/usr/bin/env",
        "-i",
        "HOME=/nonexistent",
        "PATH=/usr/bin:/bin",
        "LANG=C.UTF-8",
        "GIT_CONFIG_NOSYSTEM=1",
        "GIT_CONFIG_GLOBAL=/dev/null",
        "/usr/bin/git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(repository),
        "bundle",
        "create",
        "-",
        "HEAD",
    ]
    try:
        with bundle.open("xb") as output:
            completed = subprocess.run(
                owner_git,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
                check=False,
            )
            output.flush()
            os.fsync(output.fileno())
        if completed.returncode:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {owner_git[0]}\n"
                f"{completed.stderr[-4000:]}"
            )
    except Exception:
        bundle.unlink(missing_ok=True)
        raise
    os.chmod(bundle, 0o400)
    heads = _run(
        ["/usr/bin/git", "bundle", "list-heads", str(bundle)]
    ).stdout.splitlines()
    if not any(line.split()[0] == commit for line in heads if line.split()):
        raise ValueError("candidate bundle does not contain the exact commit")
    _run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--no-checkout",
            str(bundle),
            str(destination),
        ],
        timeout=300,
    )
    _run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            commit,
        ],
        timeout=300,
    )
    if _run(
        ["/usr/bin/git", "-C", str(destination), "status", "--porcelain"]
    ).stdout.strip():
        raise ValueError("root candidate clone is not clean")


def _bundle_root_repository(
    repository: Path,
    commit: str,
    bundle: Path,
) -> None:
    _run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            "bundle",
            "create",
            str(bundle),
            "HEAD",
        ],
        timeout=300,
    )
    os.chmod(bundle, 0o400)
    heads = _run(
        ["/usr/bin/git", "bundle", "list-heads", str(bundle)]
    ).stdout.splitlines()
    if not any(line.split()[0] == commit for line in heads if line.split()):
        raise ValueError("installed harness bundle lacks the exact commit")


def _root_environment(request_root: Path) -> dict[str, str]:
    return {
        "HOME": "/root",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "GLM_W1_ROOT_AUTHORITY": "1",
        "GLM_W1_AUTHORITY_REQUEST_ROOT": str(request_root),
    }


def _next_attempt_root(request_base: Path) -> Path:
    existing = [
        int(path.name.removeprefix("attempt-"))
        for path in request_base.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    ]
    return request_base / f"attempt-{max(existing, default=0) + 1:03d}"


def _copy_regular_tree(source: Path, destination: Path) -> None:
    rows = _tree_manifest(source)
    destination.mkdir(mode=0o700, parents=False)
    for row in rows:
        relative = Path(str(row["path"]))
        target = destination / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_path = source / relative
        with source_path.open("rb") as reader, target.open("xb") as writer:
            while block := reader.read(1024 * 1024):
                writer.write(block)
        if _sha256(target) != row["sha256"]:
            raise ValueError("exported attempt differs from root evidence")
    for path in destination.rglob("*"):
        os.chown(path, 0, 0)
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chown(destination, 0, 0)
    os.chmod(destination, 0o555)


def _publish_controller_attempt(source: Path) -> Path:
    gate = CONTROLLER_ATTEMPTS
    gate.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chown(gate, 0, 0)
    os.chmod(gate, 0o755)
    existing = [
        int(path.name.removeprefix("attempt-"))
        for path in gate.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    ]
    destination = gate / f"attempt-{max(existing, default=0) + 1:03d}"
    if destination.exists():
        raise ValueError("controller attempt destination already exists")
    _copy_regular_tree(source, destination)
    return destination


def _select_campaign_controller_attempt(output: Path) -> Path:
    attempt = output / "controller-attempt-final"
    try:
        details = attempt.lstat()
    except FileNotFoundError as exc:
        raise ValueError("exact campaign attempt is absent") from exc
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != 0
        or details.st_gid != 0
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise ValueError("exact campaign attempt is unsafe")
    return attempt


def run_campaign(harness: str, engine: str, model_hash: str) -> int:
    global ACTIVE_REQUEST
    _assert_docker_closed()
    _assert_repository(
        REPOSITORY, harness, reject_untracked=True, required_path=RUNNER
    )
    _assert_repository(
        ENGINE_REPOSITORY, engine, reject_untracked=False
    )
    if _model_hash_from_profile(harness) != model_hash:
        raise ValueError("requested model hash differs from the reviewed profile")
    if not MODEL.is_file() or MODEL.is_symlink() or _sha256(MODEL) != model_hash:
        raise ValueError("model content differs from the reviewed profile")

    request_id = _request_id(harness, engine, model_hash)
    request_base = STATE_ROOT / "requests" / request_id
    request_base.mkdir(parents=True, mode=0o711, exist_ok=True)
    os.chown(request_base, 0, 0)
    os.chmod(request_base, 0o711)
    completed_receipts = sorted(request_base.glob("attempt-*/receipt.json"))
    if completed_receipts:
        latest = json.loads(completed_receipts[-1].read_text(encoding="utf-8"))
        composite = latest.get("composite_candidate_sha256")
        authority_link = (
            STATE_ROOT / "by-composite" / composite
            if isinstance(composite, str) and HASH64.fullmatch(composite)
            else None
        )
        if (
            receipt_exit_code(latest) == 0
            and authority_link is not None
            and authority_link.is_file()
            and not authority_link.is_symlink()
            and authority_link.stat().st_ino
            == completed_receipts[-1].stat().st_ino
        ):
            print(json.dumps(latest, sort_keys=True, separators=(",", ":")))
            return 0

    request_root = _next_attempt_root(request_base)
    request_root.mkdir(mode=0o711)
    os.chown(request_root, 0, 0)
    os.chmod(request_root, 0o711)
    ACTIVE_REQUEST = {
        "root": str(request_root),
        "request_id": request_id,
        "phase": "request-initialization",
    }
    _canonical_json(
        request_root / "request.json",
        {
            "schema_version": 2,
            "request_id": request_id,
            "harness_commit": harness,
            "engine_commit": engine,
            "model_sha256": model_hash,
        },
    )
    harness_bundle = request_root / "harness.bundle"
    engine_bundle = request_root / "engine.bundle"
    harness_repository = request_root / "harness-repository"
    engine_repository = request_root / "engine-repository"
    _bundle_clone(REPOSITORY, harness, harness_bundle, harness_repository)
    _assert_root_execution_surface(harness_repository)
    _bundle_clone(ENGINE_REPOSITORY, engine, engine_bundle, engine_repository)

    ACTIVE_REQUEST["phase"] = "freeze"
    freeze = request_root / "freeze"
    campaign = request_root / "campaign"
    crashlog = request_root / "crashlog"
    campaign.mkdir(mode=0o700)
    os.chown(campaign, 0, 0)
    os.chmod(campaign, 0o700)
    crashlog.mkdir(mode=0o700)
    dsv4_identity = pwd.getpwnam(DSV4)
    os.chown(crashlog, dsv4_identity.pw_uid, dsv4_identity.pw_gid)
    environment = _root_environment(request_root)
    campaign_program = INSTALLED_HARNESS / "scripts/glm52_w1_affine_campaign.py"
    freeze_result = _run(
        [
            "/usr/bin/python3",
            str(campaign_program),
            "freeze",
            "--harness-source",
            str(harness_repository),
            "--engine-source",
            str(engine_repository),
            "--engine-candidate-hash",
            engine,
            "--model",
            str(MODEL),
            "--model-sha256",
            model_hash,
            "--freeze-dir",
            str(freeze),
        ],
        check=False,
        timeout=3600,
        cwd=INSTALLED_HARNESS,
        environment=environment,
    )
    (request_root / "freeze.log").write_text(
        freeze_result.stdout, encoding="utf-8"
    )
    if freeze_result.returncode:
        receipt = {
            "schema_version": 2,
            "terminal_state": "FAIL",
            "service_returncode": freeze_result.returncode,
            "failure_phase": "freeze",
            "request_id": request_id,
        }
        _canonical_json(request_root / "receipt.json", receipt)
        _seal(request_root)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        ACTIVE_REQUEST = None
        return 1

    descriptor = json.loads((freeze / "freeze.json").read_text(encoding="utf-8"))
    composite = descriptor.get("composite_candidate_sha256")
    if not isinstance(composite, str) or not HASH64.fullmatch(composite):
        raise ValueError("campaign composite hash is absent")
    frozen_harness = Path(descriptor.get("harness_source", "")).resolve()
    if (
        frozen_harness != harness_repository.resolve()
        or _git_root(frozen_harness, "rev-parse", "HEAD^{commit}") != harness
    ):
        raise ValueError("frozen root harness path or identity is invalid")
    ACTIVE_REQUEST["phase"] = "public-randomness"
    drand = request_root / "drand.json"
    frozen_at = descriptor.get("frozen_at")
    if not isinstance(frozen_at, str):
        raise ValueError("campaign freeze timestamp is absent")
    drand.write_text(
        _fetch_first_post_freeze_drand(frozen_at),
        encoding="utf-8",
    )

    ACTIVE_REQUEST["phase"] = "campaign"
    run_result = _run(
        [
            "/usr/bin/python3",
            str(campaign_program),
            "run",
            "--freeze-dir",
            str(freeze),
            "--drand-json",
            str(drand),
            "--output",
            str(campaign),
        ],
        check=False,
        timeout=12 * 60 * 60,
        cwd=INSTALLED_HARNESS,
        environment=environment,
    )
    (request_root / "campaign.log").write_text(
        run_result.stdout, encoding="utf-8"
    )
    try:
        controller_attempt = _select_campaign_controller_attempt(campaign)
    except ValueError:
        receipt = {
            "schema_version": 2,
            "terminal_state": "FAIL",
            "service_returncode": run_result.returncode or 1,
            "failure_phase": "campaign",
            "request_id": request_id,
            "composite_candidate_sha256": composite,
        }
        _canonical_json(request_root / "receipt.json", receipt)
        _seal(request_root)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        ACTIVE_REQUEST = None
        return 1

    ACTIVE_REQUEST["phase"] = "publication"
    attempt_manifest = _tree_manifest(controller_attempt)
    authority_pass = run_result.returncode == 0
    receipt = {
        "schema_version": 2,
        "terminal_state": "PASS" if authority_pass else "FAIL",
        "service_returncode": run_result.returncode,
        "campaign_returncode": run_result.returncode,
        "request_id": request_id,
        "harness_commit": harness,
        "engine_commit": engine,
        "model_sha256": model_hash,
        "composite_candidate_sha256": composite,
        "controller_attempt_manifest_sha256": _manifest_sha256(attempt_manifest),
        "controller_attempt_files": attempt_manifest,
    }
    by_composite = STATE_ROOT / "by-composite" / composite
    if authority_pass and by_composite.exists():
        raise ValueError("composite receipt already exists")
    destination = _publish_controller_attempt(controller_attempt)
    _canonical_json(request_root / "receipt.json", receipt)
    _seal(request_root)
    if authority_pass:
        os.link(request_root / "receipt.json", by_composite)
        os.chmod(by_composite, 0o444)
    receipt["controller_attempt"] = str(destination)
    # The external receipt is immutable; the exported path is informational
    # and is deliberately not part of its acceptance digest.
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    ACTIVE_REQUEST = None
    return receipt_exit_code(receipt)


def show_status(composite: str) -> int:
    receipt = STATE_ROOT / "by-composite" / composite
    if not receipt.is_file() or receipt.is_symlink():
        print(json.dumps({"composite_candidate_sha256": composite, "status": "PENDING"}))
        return 3
    value = json.loads(receipt.read_text(encoding="utf-8"))
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return receipt_exit_code(value)


def diagnose_campaign(composite: str) -> dict[str, object]:
    """Expose the exact error from one immutable failed campaign, read-only."""
    matches: list[tuple[str, Path]] = []
    requests = STATE_ROOT / "requests"
    for request_id in _listdir_noatime(requests):
        if not HASH64.fullmatch(request_id):
            continue
        request_base = requests / request_id
        for attempt_name in _listdir_noatime(request_base):
            if not re.fullmatch(r"attempt-[0-9]{3}", attempt_name):
                continue
            request_root = request_base / attempt_name
            details = request_root.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISDIR(details.st_mode)
                or details.st_uid != ROOT_UID
                or details.st_gid != ROOT_GID
                or stat.S_IMODE(details.st_mode) != 0o500
            ):
                continue
            try:
                receipt = json.loads(
                    _read_regular_noatime(
                        request_root / "receipt.json",
                        MAX_DIAGNOSTIC_JSON_BYTES,
                    ).decode("utf-8")
                )
            except (FileNotFoundError, OSError, ValueError, TypeError):
                continue
            if (
                receipt.get("composite_candidate_sha256") == composite
                and receipt.get("request_id") == request_id
                and receipt.get("terminal_state") == "FAIL"
                and receipt.get("failure_phase") == "campaign"
            ):
                matches.append((request_id, request_root))
    if len(matches) != 1:
        raise ValueError("exactly one sealed failed campaign was not found")
    request_id, request_root = matches[0]
    before = _diagnostic_input_manifest(request_root)
    request = json.loads(
        _read_regular_noatime(
            request_root / "request.json",
            MAX_DIAGNOSTIC_JSON_BYTES,
        ).decode("utf-8")
    )
    freeze = json.loads(
        _read_regular_noatime(
            request_root / "freeze" / "freeze.json",
            MAX_DIAGNOSTIC_JSON_BYTES,
        ).decode("utf-8")
    )
    if (
        request.get("request_id") != request_id
        or freeze.get("composite_candidate_sha256") != composite
    ):
        raise ValueError("failed campaign identity binding is invalid")
    lines = [
        line
        for line in _read_regular_noatime(
            request_root / "campaign.log",
            16 * 1024 * 1024,
        )
        .decode("utf-8")
        .splitlines()
        if line
    ]
    if not lines:
        raise ValueError("failed campaign log has no exact error")
    exact_error = lines[-1]
    if len(exact_error.encode("utf-8")) > 4096:
        raise ValueError("failed campaign error is unexpectedly large")
    after = _diagnostic_input_manifest(request_root)
    if before != after:
        raise ValueError("sealed failed campaign changed during diagnosis")
    return {
        "schema_version": 1,
        "terminal_state": "NO_RESULT",
        "composite_candidate_sha256": composite,
        "request_id": request_id,
        "exact_error": exact_error,
        "diagnostic_inputs_manifest_sha256": _manifest_sha256(before),
    }


def main(argv: list[str]) -> int:
    global ACTIVE_REQUEST
    request = parse_request(argv)
    _assert_root()
    os.umask(0o077)
    STATE_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
    (STATE_ROOT / "requests").mkdir(mode=0o711, exist_ok=True)
    (STATE_ROOT / "by-composite").mkdir(mode=0o755, exist_ok=True)
    CONTROLLER_ATTEMPTS.mkdir(mode=0o755, exist_ok=True)
    with LOCK.open("a+b") as lock:
        os.chmod(LOCK, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if request[0] == "rung0-maintenance-begin":
            return _rung0_maintenance_begin(request[1])
        if request[0] == "rung0-maintenance-end":
            return _rung0_maintenance_end(request[1])
        if request[0] == "run":
            with _hold_inference_locks():
                try:
                    return run_campaign(request[1], request[2], request[3])
                except Exception as exc:
                    _record_failed_active_request(exc)
                    ACTIVE_REQUEST = None
                    raise
        if request[0] == "diagnose":
            print(
                json.dumps(
                    diagnose_campaign(request[1]),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        return show_status(request[1])


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, RuntimeError, PermissionError) as exc:
        print(f"glm52-w1-submit: {exc}", file=sys.stderr)
        raise SystemExit(1)
