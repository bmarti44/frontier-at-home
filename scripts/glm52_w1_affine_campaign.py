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
import secrets
import shutil
import subprocess
import sys
import tempfile
import pwd
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SCORER = ROOT / "scripts/glm52_goal.py"
ROOT_AUTHORITY = os.environ.get("GLM_W1_ROOT_AUTHORITY") == "1"
AUTHORITY_REQUEST_ROOT = Path(
    os.environ.get("GLM_W1_AUTHORITY_REQUEST_ROOT", "/nonexistent")
)
MASTER_MANIFEST = Path(
    "gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv"
)
COMMON_ENGINE_ENVIRONMENT = {
    "DS4_LOCK_FILE": (
        "/run/dsv4/ds4-engine.lock"
        if ROOT_AUTHORITY
        else "/run/user/1000/ds4-engine.lock"
    ),
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
    "GLM_SAFE_RUN_AS_CURRENT_USER": "0" if ROOT_AUTHORITY else "1",
    "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
    "GLM_SAFE_KILL_FLOOR_GIB": "40",
    "GLM_SAFE_MIN_START_GIB": "110",
    "GLM_SAFE_TIMEOUT_S": "1800",
}
if ROOT_AUTHORITY:
    if (
        os.geteuid() != 0
        or not re.fullmatch(
            r"/var/lib/glm52-w1/requests/[0-9a-f]{64}/attempt-[0-9]{3}",
            str(AUTHORITY_REQUEST_ROOT),
        )
    ):
        raise RuntimeError("invalid root W1 authority environment")
    SAFE_ENVIRONMENT.update(
        {
            "GLM_W1_ROOT_AUTHORITY": "1",
            "GLM_SAFE_CRASH_ROOT": str(AUTHORITY_REQUEST_ROOT / "crashlog"),
        }
    )
START_RE = re.compile(
    r"^ds4: GLM compact cache fidelity resolved_mode=(\d+)$", re.MULTILINE
)
EXIT_RE = re.compile(
    r"^ds4: GLM compact cache fidelity attestation resolved_mode=(\d+) "
    r"affine_store_rows=(\d+) affine_changed_values=(\d+)$",
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
    composite_candidate_sha256: str,
) -> str:
    for value, label in (
        (drand_randomness, "drand randomness"),
        (composite_candidate_sha256, "composite candidate"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{label} is invalid")
    return hashlib.sha256(
        f"{composite_candidate_sha256}:{drand_randomness}:W1".encode()
    ).hexdigest()


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


def parse_attestation(log: str) -> tuple[int, int, int]:
    starts = START_RE.findall(log)
    exits = EXIT_RE.findall(log)
    if len(starts) != 1 or len(exits) != 1:
        raise ValueError("runtime mode attestation is missing or duplicated")
    start_mode = int(starts[0])
    exit_mode, rows, changed = map(int, exits[0])
    if start_mode != exit_mode:
        raise ValueError("runtime mode changed within one process")
    return start_mode, rows, changed


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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_canonical_json(path: Path, value: Any) -> str:
    data = _canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"existing immutable artifact differs: {path}")
    if not path.exists():
        path.write_bytes(data)
    return sha256_bytes(data)


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _git_environment() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _trusted_git(
    source: Path,
    *arguments: str,
    git_dir: Path | None = None,
) -> list[str]:
    command = [
        "/usr/bin/git",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
    ]
    if git_dir is None:
        command.extend(
            [
                "-c",
                f"safe.directory={source.resolve()}",
                "-C",
                str(source.resolve()),
            ]
        )
    else:
        command.extend(
            [
                f"--git-dir={git_dir.resolve()}",
                f"--work-tree={source.resolve()}",
            ]
        )
    command.extend(
        [
        *arguments,
        ]
    )
    return command


def _validate_trusted_git_dir(git_dir: Path, allowed_root: Path) -> None:
    git_dir = git_dir.resolve()
    allowed_root = allowed_root.resolve()
    if not git_dir.is_relative_to(allowed_root):
        raise ValueError("worktree Git directory escapes the root-owned clone")
    cursor = git_dir
    while True:
        details = cursor.lstat()
        if (
            cursor.is_symlink()
            or not cursor.is_dir()
            or details.st_uid != 0
            or details.st_gid != 0
            or details.st_mode & 0o022
        ):
            raise ValueError("worktree Git metadata is not root controlled")
        if cursor == allowed_root:
            break
        cursor = cursor.parent


def _source_commit(source: Path, *, git_dir: Path | None = None) -> str:
    status = subprocess.run(
        _trusted_git(
            source,
            "status",
            "--porcelain",
            "--untracked-files=no",
            git_dir=git_dir,
        ),
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=_git_environment(),
    )
    if status.stdout:
        raise ValueError("engine source has tracked modifications")
    commit = subprocess.run(
        _trusted_git(source, "rev-parse", "HEAD", git_dir=git_dir),
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        env=_git_environment(),
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("engine candidate commit is invalid")
    return commit


def _commit_time(source: Path, commit: str) -> str:
    raw = subprocess.run(
        _trusted_git(source, "show", "-s", "--format=%cI", commit),
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        env=_git_environment(),
    ).stdout.strip()
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        raise ValueError("candidate commit time lacks a timezone")
    return value.astimezone(timezone.utc).isoformat()


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
        or not re.fullmatch(r"[0-9a-f]{192}", signature)
        or hashlib.sha256(bytes.fromhex(signature)).hexdigest() != randomness
    ):
        raise ValueError("drand record is malformed")
    return {
        "round": round_number,
        "randomness": randomness,
        "signature": signature,
    }


def _reject_duplicate_drand_keys(
    pairs: list[tuple[str, Any]], host: str
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate drand relay key from {host}: {key}")
        value[key] = item
    return value


def _authenticate_drand(record: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "round": record["round"],
        "randomness": record["randomness"],
        "signature": record["signature"],
    }
    for host in ("api.drand.sh", "api2.drand.sh", "api3.drand.sh"):
        response = subprocess.run(
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
                f"https://{host}/public/{record['round']}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={
                "HOME": "/nonexistent",
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
            },
        )
        if response.returncode:
            raise ValueError(f"drand relay unavailable: {host}")
        published = json.loads(
            response.stdout,
            object_pairs_hook=lambda pairs: _reject_duplicate_drand_keys(
                pairs, host
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite drand relay value: {value}")
            ),
        )
        if any(published.get(field) != value for field, value in expected.items()):
            raise ValueError(f"drand relay disagreement: {host}")
    return {
        **record,
        "obtained_at": datetime.now(timezone.utc).isoformat(),
    }


def verify_model_content(model: Path, expected_sha256: str) -> str:
    _validate_sha256(expected_sha256, "expected model")
    observed = sha256_file(model)
    if observed != expected_sha256:
        raise ValueError("model content hash changed")
    return observed


def model_identity(model: Path) -> str:
    stat = model.stat()
    return (
        f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
        f"{stat.st_uid}:{stat.st_gid}:{stat.st_mode & 0o777}"
    )


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


def _seal_root_fixture_inputs(manifests: list[Path]) -> None:
    fixture_directories = {manifest.parent.resolve() for manifest in manifests}
    if len(fixture_directories) != 1:
        raise ValueError("fixture manifests do not share one directory")
    for manifest in manifests:
        details = manifest.lstat()
        if manifest.is_symlink() or not manifest.is_file() or details.st_nlink != 1:
            raise ValueError("fixture manifest is not a private regular file")
        os.chown(manifest, 0, 0)
        os.chmod(manifest, 0o444)
    fixture_directory = fixture_directories.pop()
    os.chown(fixture_directory, 0, 0)
    os.chmod(manifest.parent, 0o555)


def _fixture_descriptor(
    source: Path, manifests: list[Path], content_sha256: str
) -> dict[str, Any]:
    blocks = []
    for block, manifest in enumerate(manifests):
        rows = _manifest_rows(manifest)
        blocks.append(
            {
                "block": block,
                "manifest_sha256": sha256_file(manifest),
                "ordered_case_ids": [row[0] for row in rows],
                "referenced_files": [
                    {
                        "path": relative,
                        "sha256": sha256_file((source / relative).resolve()),
                    }
                    for row in rows
                    for relative in row[1:]
                ],
            }
        )
    return {
        "schema_version": 1,
        "content_sha256": content_sha256,
        "blocks": blocks,
    }


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        env=_git_environment(),
    ).stdout.strip()


def _engine_build_descriptor(
    source: Path,
    git_dir: Path,
    engine_commit: str,
    server_binary: Path,
    quality_binary: Path,
) -> dict[str, Any]:
    objects = {
        str(path.relative_to(source)): sha256_file(path)
        for path in sorted(source.rglob("*.o"))
        if path.is_file()
    }
    return {
        "schema_version": 1,
        "repository": _command_output(
            _trusted_git(
                source, "remote", "get-url", "origin", git_dir=git_dir
            ),
            source,
        ),
        "commit": engine_commit,
        "tree": _command_output(
            _trusted_git(
                source,
                "rev-parse",
                f"{engine_commit}^{{tree}}",
                git_dir=git_dir,
            ),
            source,
        ),
        "status_porcelain": _command_output(
            _trusted_git(
                source,
                "status",
                "--porcelain",
                "--untracked-files=no",
                git_dir=git_dir,
            ),
            source,
        ),
        "build_commands": [
            "make clean",
            "make -j2 CUDA_ARCH=native ds4-server "
            "gguf-tools/quality-testing/score_official "
            "tests/test_glm_affine_int8_cuda",
            "./tests/test_glm_affine_int8_cuda",
        ],
        "compiler": _command_output(["cc", "--version"]).splitlines()[0],
        "cuda_compiler": _command_output(
            ["/usr/local/cuda/bin/nvcc", "--version"]
        ),
        "server_binary_sha256": sha256_file(server_binary),
        "quality_binary_sha256": sha256_file(quality_binary),
        "object_sha256": objects,
    }


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 900,
    untrusted: bool = False,
) -> str:
    run_home = pwd.getpwuid(os.geteuid()).pw_dir
    actual_command = command
    transient_unit: str | None = None
    if ROOT_AUTHORITY and untrusted:
        transient_unit = f"glm52-w1-build-{os.getpid()}-{secrets.token_hex(4)}"
        actual_command = [
            "/usr/bin/systemd-run",
            "--wait",
            "--collect",
            "--pipe",
            "--quiet",
            f"--unit={transient_unit}",
            "--service-type=exec",
            "--uid=dsv4",
            "--gid=dsv4",
            f"--working-directory={cwd}",
            "-p",
            "KillMode=control-group",
            "-p",
            "SendSIGKILL=yes",
            "-p",
            "TimeoutStopSec=15s",
            "-p",
            f"RuntimeMaxSec={timeout}s",
            "-p",
            "MemoryAccounting=yes",
            "-p",
            "MemoryHigh=36G",
            "-p",
            "MemoryMax=40G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "OOMPolicy=kill",
            "-p",
            "TasksMax=4096",
            "-p",
            "ProtectHome=read-only",
            "-p",
            "NoNewPrivileges=yes",
            "--",
            "/usr/bin/env",
            "-i",
            "HOME=/home/dsv4",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG=C.UTF-8",
            *command,
        ]
    try:
        completed = subprocess.run(
            actual_command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout + 30 if transient_unit else timeout,
            check=False,
            env={
            "HOME": run_home,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            },
        )
    finally:
        if transient_unit is not None:
            subprocess.run(
                ["/usr/bin/systemctl", "stop", f"{transient_unit}.service"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
    if completed.returncode:
        raise ValueError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed.stdout


def _fresh_worktree(repository: Path, destination: Path, commit: str) -> None:
    if destination.exists():
        raise ValueError(f"fresh worktree destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        [
            "/usr/bin/git",
            "worktree",
            "add",
            "--detach",
            str(destination),
            commit,
        ],
        cwd=repository,
    )


def _chown_tree(path: Path, user: str) -> None:
    identity = pwd.getpwnam(user)
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError("candidate source contains a symlink")
        os.chown(child, identity.pw_uid, identity.pw_gid)
    os.chown(path, identity.pw_uid, identity.pw_gid)


def _seal_candidate_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        details = child.lstat()
        if child.is_symlink():
            raise ValueError("candidate source contains a symlink")
        executable = bool(details.st_mode & 0o111)
        os.chown(child, 0, 0)
        os.chmod(
            child,
            0o555 if child.is_dir() or executable else 0o444,
        )
    os.chown(path, 0, 0)
    os.chmod(path, 0o555)


def freeze_candidate(args: argparse.Namespace) -> int:
    """Clean-build and freeze every identity before public randomness."""
    requested_harness = (
        args.harness_source.resolve()
        if args.harness_source is not None
        else ROOT
    )
    engine_repository = args.engine_source.resolve()
    model = args.model.resolve()
    freeze_dir = args.freeze_dir.resolve()
    if freeze_dir.exists():
        raise ValueError("freeze directory already exists")
    if not model.is_file() or not engine_repository.is_dir():
        raise ValueError("engine source or model is absent")
    if ROOT_AUTHORITY and (
        args.harness_source is None
        or requested_harness
        != (AUTHORITY_REQUEST_ROOT / "harness-repository").resolve()
    ):
        raise ValueError("root authority harness source is invalid")
    harness_commit = _source_commit(requested_harness)
    engine_commit = _source_commit(engine_repository)
    if args.engine_candidate_hash and args.engine_candidate_hash != engine_commit:
        raise ValueError("engine candidate hash changed")

    tag = f"{harness_commit[:12]}-{engine_commit[:12]}"
    worktree_root = (
        AUTHORITY_REQUEST_ROOT / "worktrees"
        if ROOT_AUTHORITY
        else Path("/home/bmarti44/.cache")
    )
    if ROOT_AUTHORITY:
        worktree_root.mkdir(mode=0o711, parents=True, exist_ok=True)
        os.chown(worktree_root, 0, 0)
        os.chmod(worktree_root, 0o711)
    harness_source = (
        requested_harness
        if ROOT_AUTHORITY
        else worktree_root / f"glm52-w1-harness-{tag}"
    )
    engine_source = worktree_root / f"glm52-w1-build-{tag}"
    if not ROOT_AUTHORITY:
        _fresh_worktree(ROOT, harness_source, harness_commit)
    _fresh_worktree(engine_repository, engine_source, engine_commit)
    engine_git_dir = Path(
        _command_output(
            _trusted_git(
                engine_source, "rev-parse", "--absolute-git-dir"
            ),
            engine_source,
        )
    ).resolve()
    if ROOT_AUTHORITY:
        _validate_trusted_git_dir(
            engine_git_dir, engine_repository / ".git"
        )
    if ROOT_AUTHORITY:
        _seal_candidate_tree(harness_source)
        _chown_tree(engine_source, "dsv4")

    transcript_parts = []
    transcript_parts.append(
        _run_checked(
            ["/usr/bin/make", "clean"],
            cwd=engine_source,
            untrusted=ROOT_AUTHORITY,
        )
    )
    transcript_parts.append(
        _run_checked(
            [
                "/usr/bin/make",
                "-j2",
                "CUDA_ARCH=native",
                "ds4-server",
                "gguf-tools/quality-testing/score_official",
                "tests/test_glm_affine_int8_cuda",
            ],
            cwd=engine_source,
            untrusted=ROOT_AUTHORITY,
        )
    )
    transcript_parts.append(
        _run_checked(
            ["./tests/test_glm_affine_int8_cuda"],
            cwd=engine_source,
            untrusted=ROOT_AUTHORITY,
        )
    )
    build_transcript = "".join(transcript_parts)
    server = engine_source / "ds4-server"
    quality = engine_source / "gguf-tools/quality-testing/score_official"
    cuda_test = engine_source / "tests/test_glm_affine_int8_cuda"
    for binary in (server, quality, cuda_test):
        if not binary.is_file() or binary.read_bytes()[:4] != b"\x7fELF":
            raise ValueError(f"clean build did not produce an ELF binary: {binary}")
    if ROOT_AUTHORITY:
        _seal_candidate_tree(engine_source)

    freeze_dir.mkdir(mode=0o700, parents=True)
    transcript_path = freeze_dir / "clean-build.log"
    transcript_path.write_text(build_transcript, encoding="utf-8")
    bundle_path = freeze_dir / "engine.bundle"
    _run_checked(
        _trusted_git(
            engine_source,
            "bundle",
            "create",
            str(bundle_path),
            "HEAD",
            git_dir=engine_git_dir,
        ),
        cwd=engine_source,
    )
    bundle_heads = _run_checked(
        _trusted_git(
            engine_source,
            "bundle",
            "list-heads",
            str(bundle_path),
            git_dir=engine_git_dir,
        ),
        cwd=engine_source,
    )
    if engine_commit not in bundle_heads:
        raise ValueError("engine bundle does not contain the frozen commit")

    if not args.model_sha256:
        raise ValueError("expected model content hash is required")
    model_sha256 = verify_model_content(model, args.model_sha256)
    if not ROOT_AUTHORITY and os.access(model, os.W_OK):
        raise ValueError("campaign model is writable by the benchmark owner")
    master = engine_source / MASTER_MANIFEST
    fixture_master_sha256 = content_complete_fixture_sha256(
        engine_source, [master]
    )
    runner_path = harness_source / "scripts/glm52_w1_affine_campaign.py"
    scorer_path = harness_source / "scripts/glm52_goal.py"
    engine_build = _engine_build_descriptor(
        engine_source, engine_git_dir, engine_commit, server, quality
    )
    engine_build["clean_build_transcript_sha256"] = sha256_file(
        transcript_path
    )
    engine_build["cuda_test_binary_sha256"] = sha256_file(cuda_test)
    engine_build["cuda_test_passed"] = True
    engine_build_path = freeze_dir / "engine-build.json"
    engine_build_sha256 = _write_canonical_json(
        engine_build_path, engine_build
    )
    base = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "harness_candidate_hash": harness_commit,
        "harness_tree": _command_output(
            _trusted_git(
                harness_source,
                "rev-parse",
                f"{harness_commit}^{{tree}}",
            ),
            harness_source,
        ),
        "harness_source": str(harness_source),
        "runner_sha256": sha256_file(runner_path),
        "scorer_sha256": sha256_file(scorer_path),
        "engine_candidate_hash": engine_commit,
        "engine_git_dir": str(engine_git_dir),
        "engine_tree": _command_output(
            _trusted_git(
                engine_source,
                "rev-parse",
                f"{engine_commit}^{{tree}}",
                git_dir=engine_git_dir,
            ),
            engine_source,
        ),
        "engine_source": str(engine_source),
        "engine_source_sha256": sha256_file(bundle_path),
        "engine_build_sha256": engine_build_sha256,
        "server_binary_sha256": sha256_file(server),
        "quality_binary_sha256": sha256_file(quality),
        "cuda_test_binary_sha256": sha256_file(cuda_test),
        "model_path": str(model),
        "model_content_sha256": model_sha256,
        "tokenizer_content_sha256": model_sha256,
        "model_identity": model_identity(model),
        "fixture_master_sha256": fixture_master_sha256,
    }
    descriptor = {
        **base,
        "composite_candidate_sha256": sha256_bytes(
            _canonical_json_bytes(base)
        ),
    }
    _write_canonical_json(freeze_dir / "freeze.json", descriptor)
    if ROOT_AUTHORITY:
        _seal_candidate_tree(engine_source)
    print(
        json.dumps(
            {
                "freeze_dir": str(freeze_dir),
                "composite_candidate_sha256": descriptor[
                    "composite_candidate_sha256"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def verify_frozen_candidate(
    freeze_dir: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    freeze_dir = freeze_dir.resolve()
    descriptor = _strict_json(freeze_dir / "freeze.json")
    if not isinstance(descriptor, dict):
        raise ValueError("freeze descriptor is not an object")
    composite = descriptor.get("composite_candidate_sha256")
    base = {
        key: value
        for key, value in descriptor.items()
        if key != "composite_candidate_sha256"
    }
    if (
        not isinstance(composite, str)
        or sha256_bytes(_canonical_json_bytes(base)) != composite
    ):
        raise ValueError("composite candidate digest changed")
    harness_source = Path(descriptor["harness_source"]).resolve()
    engine_source = Path(descriptor["engine_source"]).resolve()
    engine_git_dir = Path(descriptor["engine_git_dir"]).resolve()
    model = Path(descriptor["model_path"]).resolve()
    if ROOT_AUTHORITY:
        _validate_trusted_git_dir(
            engine_git_dir,
            AUTHORITY_REQUEST_ROOT / "engine-repository" / ".git",
        )
    if (
        _source_commit(harness_source) != descriptor["harness_candidate_hash"]
        or _source_commit(engine_source, git_dir=engine_git_dir)
        != descriptor["engine_candidate_hash"]
        or sha256_file(
            harness_source / "scripts/glm52_w1_affine_campaign.py"
        )
        != descriptor["runner_sha256"]
        or sha256_file(harness_source / "scripts/glm52_goal.py")
        != descriptor["scorer_sha256"]
        or sha256_file(engine_source / "ds4-server")
        != descriptor["server_binary_sha256"]
        or sha256_file(engine_source / "gguf-tools/quality-testing/score_official")
        != descriptor["quality_binary_sha256"]
        or sha256_file(freeze_dir / "engine.bundle")
        != descriptor["engine_source_sha256"]
        or sha256_file(freeze_dir / "engine-build.json")
        != descriptor["engine_build_sha256"]
        or model_identity(model) != descriptor["model_identity"]
    ):
        raise ValueError("frozen candidate identity changed")
    if sha256_file(Path(__file__).resolve()) != descriptor["runner_sha256"]:
        raise ValueError("live runner differs from frozen candidate")
    master_sha256 = content_complete_fixture_sha256(
        engine_source, [engine_source / MASTER_MANIFEST]
    )
    if master_sha256 != descriptor["fixture_master_sha256"]:
        raise ValueError("frozen fixture master changed")
    return descriptor, harness_source, engine_source, model


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


def _journal_witness(message: str, expected_nonce: str) -> dict[str, str]:
    if not message.startswith(f"W1_WITNESS nonce={expected_nonce} "):
        raise ValueError("journal witness nonce is absent or wrong")
    completed = subprocess.run(
        [
            "/usr/bin/journalctl",
            "--no-pager",
            "-o",
            "json",
            "--since",
            "5 minutes ago",
            "-t",
            "glm52-w1-witness",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
        },
    )
    if completed.returncode:
        raise ValueError("system journal witness is unavailable")
    rows = []
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        expected_uid = "995" if ROOT_AUTHORITY else "1000"
        if row.get("MESSAGE") == message and row.get("_UID") == expected_uid:
            rows.append(row)
    if len(rows) != 1:
        raise ValueError("system journal witness is missing or duplicated")
    row = rows[0]
    unit_match = re.search(r"(?:^| )unit=([A-Za-z0-9_.@-]+)(?: |$)", message)
    if unit_match is None:
        raise ValueError("journal witness unit is absent")
    unit = unit_match.group(1) + ".service"
    receipt = {
        "cursor": str(row.get("__CURSOR", "")),
        "realtime_timestamp": str(row.get("__REALTIME_TIMESTAMP", "")),
        "boot_id": str(row.get("_BOOT_ID", "")),
        "invocation_id": str(row.get("_SYSTEMD_INVOCATION_ID", "")),
        "pid": str(row.get("_PID", "")),
        "uid": str(row.get("_UID", "")),
        "cgroup": str(row.get("_SYSTEMD_CGROUP", "")),
        "user_unit": str(
            row.get("_SYSTEMD_UNIT" if ROOT_AUTHORITY else "_SYSTEMD_USER_UNIT", "")
        ),
        "message": message,
    }
    if (
        not all(receipt.values())
        or receipt["user_unit"] != unit
        or not receipt["cgroup"].endswith(f"/{unit}")
    ):
        raise ValueError("system journal witness has wrong trusted metadata")
    return receipt


def _score(campaign: dict[str, Any]) -> dict[str, Any]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "glm52_goal_fixed", SCORER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load fixed scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._score_w1_affine_raw([campaign])


def _goal_module(frozen_scorer_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "glm52_goal_authority", frozen_scorer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load controller authority")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finalize_controller_attempt(
    output: Path,
    campaign: dict[str, Any],
    summary: dict[str, Any],
    frozen_binary: Path,
    final_model_identity: str,
    harness_source: Path,
    freeze_dir: Path,
) -> Path:
    goal = _goal_module(SCORER)
    staging = output / "controller-attempt"
    if staging.exists():
        raise ValueError("controller attempt staging already exists")
    staging.mkdir(mode=0o700)

    source_descriptor = {
        "schema_version": 1,
        "candidate_hash": campaign["harness_candidate_hash"],
        "git_tree": _command_output(
            _trusted_git(
                harness_source,
                "rev-parse",
                f"{campaign['harness_candidate_hash']}^{{tree}}",
            ),
            harness_source,
        ),
    }
    model_descriptor = {
        "schema_version": 1,
        "content_sha256": campaign["model_content_sha256"],
        "identity": final_model_identity,
    }
    tokenizer_descriptor = {
        "schema_version": 1,
        "lineage": "embedded-in-model-container",
        "content_sha256": campaign["tokenizer_content_sha256"],
    }
    scorer_descriptor = {
        "schema_version": 1,
        "scorer_id": "w1.affine-quality.v2",
        "implementation_sha256": goal.registered_scorer_digest(
            "w1.affine-quality.v2"
        ),
    }
    artifact_values = {
        "source.json": source_descriptor,
        "model.json": model_descriptor,
        "tokenizer.json": tokenizer_descriptor,
        "scorer.json": scorer_descriptor,
        "evidence.json": campaign,
    }
    for name, value in artifact_values.items():
        _write_canonical_json(staging / name, value)
    shutil.copyfile(output / "engine-build.json", staging / "engine-build.json")
    shutil.copyfile(freeze_dir / "engine.bundle", staging / "engine.bundle")
    shutil.copyfile(freeze_dir / "clean-build.log", staging / "clean-build.log")
    shutil.copyfile(output / "configuration.json", staging / "configuration.json")
    shutil.copyfile(output / "fixture.json", staging / "fixture.json")
    shutil.copyfile(frozen_binary, staging / "quality-binary")
    os.chmod(staging / "quality-binary", 0o500)
    (staging / "raw.jsonl").write_bytes(_canonical_json_bytes(campaign))
    _write_canonical_json(staging / "summary.json", summary)

    manifest = {
        "schema_version": 1,
        "gate": "W1",
        "candidate_hash": campaign["harness_candidate_hash"],
        "lineage": campaign["lineage"],
        "artifacts": {
            "source": "source.json",
            "diff": "engine-build.json",
            "binary": "quality-binary",
            "scorer": "scorer.json",
            "model": "model.json",
            "tokenizer": "tokenizer.json",
            "fixture": "fixture.json",
            "configuration": "configuration.json",
            "evidence": "evidence.json",
            "engine_source": "engine.bundle",
            "build_log": "clean-build.log",
        },
        "source_sha256": sha256_file(staging / "source.json"),
        "diff_sha256": sha256_file(staging / "engine-build.json"),
        "binary_sha256": sha256_file(staging / "quality-binary"),
        "scorer_sha256": sha256_file(staging / "scorer.json"),
        "model_sha256": sha256_file(staging / "model.json"),
        "tokenizer_sha256": sha256_file(staging / "tokenizer.json"),
        "fixture_sha256": sha256_file(staging / "fixture.json"),
        "configuration_sha256": sha256_file(staging / "configuration.json"),
        "evidence_sha256": sha256_file(staging / "evidence.json"),
        "engine_source_sha256": sha256_file(staging / "engine.bundle"),
        "build_log_sha256": sha256_file(staging / "clean-build.log"),
    }
    _write_canonical_json(staging / "manifest.json", manifest)
    goal.validate_attempt(
        staging,
        root_authority_pending=ROOT_AUTHORITY,
        source_repository=harness_source if ROOT_AUTHORITY else None,
    )

    if ROOT_AUTHORITY:
        destination = output / "controller-attempt-final"
    else:
        gate_dir = ROOT / "results/glm52-goal/W1"
        gate_dir.mkdir(parents=True, exist_ok=True)
        existing_numbers = [
            int(path.name.removeprefix("attempt-"))
            for path in gate_dir.glob("attempt-*")
            if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
        ]
        destination = (
            gate_dir
            / f"attempt-{max(existing_numbers, default=0) + 1:03d}"
        )
    if destination.exists():
        raise ValueError("controller attempt destination already exists")
    os.replace(staging, destination)
    goal.validate_attempt(
        destination,
        root_authority_pending=ROOT_AUTHORITY,
        source_repository=harness_source if ROOT_AUTHORITY else None,
    )
    return destination


def _freeze_scorer(source: Path, engine_commit: str, binary_sha256: str) -> Path:
    frozen = (
        AUTHORITY_REQUEST_ROOT / "frozen-scorer"
        if ROOT_AUTHORITY
        else Path(
            f"/home/bmarti44/.cache/glm52-w1-affine-score-{engine_commit[:12]}"
        )
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
    if ROOT_AUTHORITY:
        os.chown(target, 0, 0)
        os.chmod(target, 0o555)
        os.chown(frozen, 0, 0)
        os.chmod(frozen, 0o555)
    return frozen


def _campaign_paths(seed: str, engine_commit: str, output: Path | None) -> Path:
    return output or Path(
        f"/home/bmarti44/.local/state/glm52-confirm-w1-affine-"
        f"{engine_commit[:12]}-{seed[:12]}"
    )


def run(args: argparse.Namespace) -> int:
    freeze_dir = args.freeze_dir.resolve()
    (
        frozen_candidate,
        harness_source,
        source,
        model,
    ) = verify_frozen_candidate(freeze_dir)
    engine_commit = frozen_candidate["engine_candidate_hash"]
    scorer_binary = source / "gguf-tools/quality-testing/score_official"
    server_binary = source / "ds4-server"
    if not scorer_binary.is_file() or not server_binary.is_file():
        raise ValueError("clean-built engine binaries are absent")
    binary_sha256 = sha256_file(scorer_binary)
    harness_commit = frozen_candidate["harness_candidate_hash"]
    composite_candidate_sha256 = frozen_candidate[
        "composite_candidate_sha256"
    ]
    drand = _authenticate_drand(_drand_record(args.drand_json.resolve()))
    seed = confirmation_seed(drand["randomness"], composite_candidate_sha256)
    frozen_at = frozen_candidate["frozen_at"]
    lineage = {
        "freeze": {
            "candidate_hash": harness_commit,
            "frozen_at": frozen_at,
            "composite_candidate_sha256": composite_candidate_sha256,
        },
        "randomness": {
            "source": "drand-default",
            **drand,
            "seed_sha256": seed,
        },
    }
    goal = _goal_module(SCORER)
    goal.validate_manifest_lineage(
        lineage,
        "W1",
        harness_commit,
        commit_time_fetcher=lambda candidate: _commit_time(
            harness_source, candidate
        ),
    )
    frozen = _freeze_scorer(source, engine_commit, binary_sha256)
    output = _campaign_paths(seed, engine_commit, args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ROOT_AUTHORITY:
        os.chown(output, 0, 0)
        os.chmod(output, 0o711)
        artifact_root = AUTHORITY_REQUEST_ROOT / "artifacts"
        artifact_root.mkdir(mode=0o700, exist_ok=True)
        dsv4 = pwd.getpwnam("dsv4")
        os.chown(artifact_root, dsv4.pw_uid, dsv4.pw_gid)
    else:
        artifact_root = output
    manifests = _write_manifests(source, output, seed)
    if ROOT_AUTHORITY:
        _seal_root_fixture_inputs(manifests)
    fixture_content_sha256 = content_complete_fixture_sha256(source, manifests)
    fixture_descriptor = _fixture_descriptor(
        source, manifests, fixture_content_sha256
    )
    fixture_path = output / "fixture.json"
    fixture_sha256 = _write_canonical_json(fixture_path, fixture_descriptor)

    model_sha256 = verify_model_content(
        model, frozen_candidate["model_content_sha256"]
    )
    initial_model_identity = model_identity(model)
    if not ROOT_AUTHORITY and os.access(model, os.W_OK):
        raise ValueError("campaign model is writable by the benchmark owner")
    (output / "model.sha256").write_text(model_sha256 + "\n", encoding="ascii")
    engine_build_path = output / "engine-build.json"
    shutil.copyfile(freeze_dir / "engine-build.json", engine_build_path)
    engine_build_sha256 = sha256_file(engine_build_path)

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
        "composite_candidate_sha256": composite_candidate_sha256,
        "binary_sha256": binary_sha256,
        "model_path": str(model),
        "model_content_sha256": model_sha256,
        "tokenizer_content_sha256": model_sha256,
        "engine_build_sha256": engine_build_sha256,
        "engine_source_sha256": frozen_candidate["engine_source_sha256"],
        "build_log_sha256": sha256_file(freeze_dir / "clean-build.log"),
        "fixture_sha256": fixture_sha256,
        "fixture_content_sha256": fixture_content_sha256,
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
        "lineage": lineage,
        "safety": SAFE_ENVIRONMENT,
    }
    configuration_path = output / "configuration.json"
    configuration_sha256 = _write_canonical_json(
        configuration_path, configuration
    )

    campaign_path = output / "campaign.json"
    campaign = {
        "record_type": "w1_affine_raw_campaign",
        "harness_candidate_hash": harness_commit,
        "engine_candidate_hash": engine_commit,
        "composite_candidate_sha256": composite_candidate_sha256,
        "seed_sha256": seed,
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "fixture_sha256": fixture_sha256,
        "fixture_content_sha256": fixture_content_sha256,
        "model_content_sha256": model_sha256,
        "tokenizer_content_sha256": model_sha256,
        "engine_build_sha256": engine_build_sha256,
        "engine_source_sha256": frozen_candidate["engine_source_sha256"],
        "build_log_sha256": sha256_file(freeze_dir / "clean-build.log"),
        "baseline_environment_sha256": baseline_environment_sha256,
        "candidate_environment_sha256": candidate_environment_sha256,
        "candidate_arm": candidate_arm(seed),
        "lineage": lineage,
        "fixture_blocks": [
            {
                "block": block["block"],
                "manifest_sha256": block["manifest_sha256"],
                "ordered_case_ids": block["ordered_case_ids"],
            }
            for block in fixture_descriptor["blocks"]
        ],
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
        "model_content_sha256": model_sha256,
        "model_identity": initial_model_identity,
        "engine_source": str(source),
        "frozen_binary": str(frozen / "ds4-server"),
        "fixture_manifests": [str(path) for path in manifests],
    })
    _atomic_json(output / "randomness.json", lineage["randomness"])
    _atomic_json(campaign_path, campaign)

    flattened = [
        (block, sequence, arm)
        for block, order in enumerate(schedules(seed))
        for sequence, arm in enumerate(order)
    ]
    for index in range(len(campaign["attempts"]), 20):
        verified, verified_harness, verified_engine, verified_model = (
            verify_frozen_candidate(freeze_dir)
        )
        if (
            verified != frozen_candidate
            or verified_harness != harness_source
            or verified_engine != source
            or verified_model != model
        ):
            raise ValueError("frozen candidate changed before attempt")
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
        if before != fixture_content_sha256:
            raise ValueError("fixture bytes changed before attempt")
        model_before = model_identity(model)
        if model_before != initial_model_identity:
            raise ValueError("model identity changed before attempt")
        cursor = _journal_cursor()
        result_path = output / f"attempt-{index:02d}.tsv"
        witness_result_path = artifact_root / f"attempt-{index:02d}.tsv"
        log_path = output / f"attempt-{index:02d}.launcher.log"
        kernel_path = output / f"attempt-{index:02d}.kernel.log"
        if result_path.exists() or witness_result_path.exists():
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
                "GLM_SAFE_WITNESS_NONCE": hashlib.sha256(
                    f"{seed}:{index}:W1-witness".encode()
                ).hexdigest(),
                "GLM_SAFE_WITNESS_ARTIFACT": str(witness_result_path),
            }
        )
        witness_nonce = environment["GLM_SAFE_WITNESS_NONCE"]
        command = [
            str(LAUNCHER),
            "--tag",
            f"w1-{seed[:8]}-{index:02d}-{arm}",
            "--",
            str(frozen / "ds4-server"),
            str(model),
            str(manifests[block]),
            str(witness_result_path),
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
        if ROOT_AUTHORITY and witness_result_path.is_file():
            os.replace(witness_result_path, result_path)
        kernel = _kernel_since(cursor)
        kernel_path.write_text(kernel, encoding="utf-8")
        failures: list[str] = []
        if result.returncode != 0:
            failures.append(f"launcher_rc={result.returncode}")
        safe_dirs = re.findall(r"SAFE_RUN_DONE .* dir=(\S+)$", result.stdout, re.MULTILINE)
        main_log = ""
        samples = ""
        command_log = ""
        journal_witness = {}
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
        witness_messages = re.findall(
            r"^(W1_WITNESS .+)$", result.stdout, re.MULTILINE
        )
        if len(witness_messages) != 1:
            failures.append("journal_witness_message_missing_or_duplicated")
        else:
            try:
                journal_witness = _journal_witness(
                    witness_messages[0], witness_nonce
                )
            except ValueError as exc:
                failures.append(str(exc))
        try:
            resolved_mode, store_count, changed_values = parse_attestation(
                command_log
            )
        except ValueError as exc:
            failures.append(str(exc))
            resolved_mode, store_count, changed_values = 0, 0, 0
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
        if after != fixture_content_sha256:
            failures.append("fixture_bytes_changed_after_attempt")
        model_after = model_identity(model)
        if model_after != model_before:
            failures.append("model_identity_changed_after_attempt")
        fault_text = "\n".join((kernel, main_log, command_log))
        oom = bool(re.search(r"oom-kill|Out of memory|memory event", fault_text, re.I))
        xid = bool(re.search(r"NVRM.*Xid|NV_ERR_NO_MEMORY", fault_text, re.I))
        if FAULT_RE.search(fault_text):
            failures.append("kernel_or_memory_fault")
        if "memory_swap_max=0" not in main_log:
            failures.append("zero_swap_cgroup_not_attested")
        if is_candidate and (
            resolved_mode != 2 or store_count <= 0 or changed_values <= 0
        ):
            failures.append("affine_device_effect_not_attested")
        if not is_candidate and (
            resolved_mode != 0 or store_count != 0 or changed_values != 0
        ):
            failures.append("baseline_mode_not_default_off")
        attempt = {
            "block": block,
            "sequence": sequence,
            "arm": arm,
            "fixture_content_sha256_before": before,
            "fixture_content_sha256_after": after,
            "model_identity_before": model_before,
            "model_identity_after": model_after,
            "evidence": {
                "launcher_log": result.stdout,
                "main_log": main_log,
                "cmd_log": command_log,
                "samples_log": samples,
                "kernel_log": kernel,
                "quality_tsv": (
                    result_path.read_text(encoding="utf-8")
                    if result_path.is_file()
                    else ""
                ),
                "journal_witness": json.dumps(
                    journal_witness,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
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
    final_model_sha256 = verify_model_content(model, model_sha256)
    final_model_identity = model_identity(model)
    if (
        final_model_sha256 != model_sha256
        or final_model_identity != initial_model_identity
    ):
        raise ValueError("model content or identity changed after campaign")
    _atomic_json(output / "summary.json", summary)
    (output / "raw.jsonl").write_text(
        json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    destination = _finalize_controller_attempt(
        output,
        campaign,
        summary,
        frozen / "ds4-server",
        final_model_identity,
        harness_source,
        freeze_dir,
    )
    print(f"controller_attempt={destination}", flush=True)
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
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--harness-source", type=Path)
    freeze_parser.add_argument("--engine-source", required=True, type=Path)
    freeze_parser.add_argument("--engine-candidate-hash")
    freeze_parser.add_argument("--model", required=True, type=Path)
    freeze_parser.add_argument("--model-sha256", required=True)
    freeze_parser.add_argument("--freeze-dir", required=True, type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--drand-json", required=True, type=Path)
    run_parser.add_argument("--freeze-dir", required=True, type=Path)
    run_parser.add_argument("--output", type=Path)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("campaign", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "freeze":
            return freeze_candidate(args)
        return run(args) if args.command == "run" else status(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"glm52-w1-affine: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
