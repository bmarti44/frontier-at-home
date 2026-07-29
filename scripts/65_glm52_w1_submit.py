#!/usr/bin/env python3
"""Root-owned, narrowly delegated authority for the fixed GLM W1 campaign."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


REPOSITORY = Path("/home/bmarti44/spark-deepseek-v4-flash")
ENGINE_REPOSITORY = Path("/home/bmarti44/.cache/glm52-w1-real-capture-a37")
MODEL = Path(
    "/home/dsv4/ds4-project/gguf-glm/"
    "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
)
STATE_ROOT = Path("/var/lib/glm52-w1")
LOCK = Path("/run/lock/glm52-w1-submit.lock")
RUNNER = Path("scripts/glm52-runners/W1")
PROFILE = Path("configs/glm52-profile.json")
OWNER = "bmarti44"
OWNER_UID = 1000
HASH40 = re.compile(r"^[0-9a-f]{40}$")
HASH64 = re.compile(r"^[0-9a-f]{64}$")


def parse_request(argv: list[str]) -> tuple[str, ...]:
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
        and argv[0] == "status"
        and HASH64.fullmatch(argv[1])
    ):
        return tuple(argv)
    raise ValueError(
        "usage: glm52-w1-submit run HARNESS_COMMIT ENGINE_COMMIT MODEL_SHA256\n"
        "       glm52-w1-submit status COMPOSITE_SHA256"
    )


def _run(
    argv: list[str],
    *,
    check: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
        },
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {argv[0]}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed


def _git(repository: Path, *arguments: str) -> str:
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


def _assert_docker_closed() -> None:
    group_line = Path("/etc/group").read_text(encoding="utf-8")
    docker = next(
        (line for line in group_line.splitlines() if line.startswith("docker:")),
        "",
    )
    members = docker.rsplit(":", 1)[-1].split(",") if docker else []
    if OWNER in members:
        raise PermissionError("bmarti44 remains in the root-equivalent docker group")
    active = _run(
        ["/usr/bin/systemctl", "is-active", "--quiet", "docker.socket"],
        check=False,
    )
    if active.returncode == 0:
        raise PermissionError("root-equivalent docker socket remains active")


def _assert_repository(
    repository: Path,
    expected_commit: str,
    required_path: Path | None = None,
) -> None:
    if not repository.is_dir() or repository.is_symlink():
        raise ValueError(f"repository is absent or unsafe: {repository}")
    actual = _git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if actual != expected_commit:
        raise ValueError(f"repository HEAD differs: {repository}")
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"repository is not clean: {repository}")
    if required_path is not None:
        tracked = _git(repository, "ls-files", "--error-unmatch", str(required_path))
        if tracked != str(required_path):
            raise ValueError(f"required runner is not tracked: {required_path}")


def _model_hash_from_profile() -> str:
    profile = json.loads((REPOSITORY / PROFILE).read_text(encoding="utf-8"))
    value = profile.get("model_sha256")
    if not isinstance(value, str) or not HASH64.fullmatch(value):
        raise ValueError("profile model hash is invalid")
    return value


def _request_id(harness: str, engine: str, model: str) -> str:
    return hashlib.sha256(f"{harness}:{engine}:{model}:W1-root-v1".encode()).hexdigest()


def _tree_manifest(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"evidence contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file() or details.st_nlink != 1:
            raise ValueError(f"evidence file type or link count is unsafe: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size": details.st_size,
                "device": details.st_dev,
                "inode": details.st_ino,
            }
        )
    return rows


def _seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o500 if path.is_dir() else 0o400, follow_symlinks=False)
    os.chown(root, 0, 0)
    os.chmod(root, 0o500)


def _journal_identity(unit: str) -> dict[str, str]:
    completed = _run(
        [
            "/usr/bin/journalctl",
            "--no-pager",
            "-o",
            "json",
            "-u",
            f"{unit}.service",
        ],
        check=False,
    )
    invocation_ids: set[str] = set()
    cursors: list[str] = []
    boot_ids: set[str] = set()
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        invocation = row.get("_SYSTEMD_INVOCATION_ID")
        cursor = row.get("__CURSOR")
        boot_id = row.get("_BOOT_ID")
        if isinstance(invocation, str) and invocation:
            invocation_ids.add(invocation)
        if isinstance(cursor, str) and cursor:
            cursors.append(cursor)
        if isinstance(boot_id, str) and boot_id:
            boot_ids.add(boot_id)
    if len(invocation_ids) != 1 or len(boot_ids) != 1 or not cursors:
        raise ValueError("root systemd invocation identity is incomplete")
    return {
        "unit": f"{unit}.service",
        "invocation_id": next(iter(invocation_ids)),
        "boot_id": next(iter(boot_ids)),
        "first_cursor": cursors[0],
        "last_cursor": cursors[-1],
    }


def _copy_attempt(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("campaign state is absent or unsafe")
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        copy_function=shutil.copy2,
    )


def run_campaign(harness: str, engine: str, model_hash: str) -> int:
    _assert_docker_closed()
    _assert_repository(REPOSITORY, harness, RUNNER)
    _assert_repository(ENGINE_REPOSITORY, engine)
    if _model_hash_from_profile() != model_hash:
        raise ValueError("requested model hash differs from the reviewed profile")
    if not MODEL.is_file() or MODEL.is_symlink() or _sha256(MODEL) != model_hash:
        raise ValueError("model content differs from the reviewed profile")

    request_id = _request_id(harness, engine, model_hash)
    request_root = STATE_ROOT / "requests" / request_id
    if (request_root / "receipt.json").is_file():
        print((request_root / "receipt.json").read_text(encoding="utf-8"), end="")
        return 0
    if request_root.exists():
        raise ValueError("incomplete prior root request requires inspection")
    request_root.mkdir(parents=True, mode=0o700)
    _canonical_json(
        request_root / "request.json",
        {
            "schema_version": 1,
            "request_id": request_id,
            "harness_commit": harness,
            "engine_commit": engine,
            "model_sha256": model_hash,
        },
    )

    unit = f"glm52-w1-root-{request_id[:16]}"
    command = [
        "/usr/bin/systemd-run",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
        f"--unit={unit}",
        "--service-type=exec",
        "--uid=bmarti44",
        "--gid=bmarti44",
        f"--working-directory={REPOSITORY}",
        "--setenv=HOME=/home/bmarti44",
        "--setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--setenv=LANG=C.UTF-8",
        "--setenv=XDG_RUNTIME_DIR=/run/user/1000",
        "--setenv=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
        "-p",
        "KillMode=control-group",
        "-p",
        "SendSIGKILL=yes",
        "-p",
        "TimeoutStopSec=45s",
        "-p",
        "RuntimeMaxSec=12h",
        "-p",
        "MemoryAccounting=yes",
        "-p",
        "MemoryHigh=72G",
        "-p",
        "MemoryMax=76G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
        "-p",
        "TasksMax=4096",
        "--",
        str(REPOSITORY / RUNNER),
        str(REPOSITORY / "results/glm52-goal"),
        "W1",
    ]
    completed = _run(command, check=False, timeout=12 * 60 * 60 + 120)
    (request_root / "service.log").write_text(completed.stdout, encoding="utf-8")
    journal = _journal_identity(unit)

    base = (
        Path("/home/bmarti44/.local/state")
        / f"glm52-controller-W1-{harness[:12]}-{engine[:12]}"
    )
    snapshot = request_root / "snapshot"
    _copy_attempt(base, snapshot)
    freeze = json.loads((snapshot / "freeze/freeze.json").read_text(encoding="utf-8"))
    composite = freeze.get("composite_candidate_sha256")
    if not isinstance(composite, str) or not HASH64.fullmatch(composite):
        raise ValueError("campaign composite hash is absent")
    manifest = _tree_manifest(snapshot)
    receipt = {
        "schema_version": 1,
        "authority": "root-owned-glm52-w1-v1",
        "request_id": request_id,
        "harness_commit": harness,
        "engine_commit": engine,
        "model_sha256": model_hash,
        "composite_candidate_sha256": composite,
        "service_returncode": completed.returncode,
        "systemd": journal,
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "snapshot_files": manifest,
    }
    _canonical_json(request_root / "receipt.json", receipt)
    _seal(request_root)
    composite_link = STATE_ROOT / "by-composite" / composite
    composite_link.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if composite_link.exists():
        raise ValueError("composite receipt already exists")
    os.link(request_root / "receipt.json", composite_link)
    os.chmod(composite_link, 0o400)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if completed.returncode == 0 else completed.returncode


def show_status(composite: str) -> int:
    receipt = STATE_ROOT / "by-composite" / composite
    if not receipt.is_file() or receipt.is_symlink():
        print(json.dumps({"composite_candidate_sha256": composite, "status": "PENDING"}))
        return 3
    print(receipt.read_text(encoding="utf-8"), end="")
    return 0


def main(argv: list[str]) -> int:
    request = parse_request(argv)
    _assert_root()
    STATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock:
        os.chmod(LOCK, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if request[0] == "run":
            return run_campaign(request[1], request[2], request[3])
        return show_status(request[1])


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, RuntimeError, PermissionError) as exc:
        print(f"glm52-w1-submit: {exc}", file=sys.stderr)
        raise SystemExit(1)
