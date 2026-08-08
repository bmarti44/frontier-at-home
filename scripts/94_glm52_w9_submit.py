#!/usr/bin/env python3
"""Root-owned execution and publication boundary for the W9 FP4 falsifier."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


OWNER_UID = 1000
OWNER_STATE = pathlib.Path("/home/bmarti44/.local/state")
INSTALL_ROOT = pathlib.Path("/usr/local/libexec/glm52-w9")
RUNNER_PATH = pathlib.Path("/usr/local/sbin/glm52-w9-submit")
REPOSITORY = pathlib.Path("/usr/local/libexec/glm52-w9/repository")
SCORER = REPOSITORY / "scripts/93_score_w9_fp4_falsifier.py"
NODE = INSTALL_ROOT / "node"
NOBLE = INSTALL_ROOT / "noble"
INSTALL_MANIFEST = INSTALL_ROOT / "install.json"
ROOT_NUMPY = pathlib.Path("/usr/local/libexec/glm52-w1/python")
STATE_ROOT = pathlib.Path("/var/lib/glm52-w9")
WORK_ROOT = pathlib.Path("/var/lib/glm52-w9/work")
PUBLISH_ROOT = pathlib.Path("/var/lib/glm52-w9/attempts")
FAILURE_ROOT = pathlib.Path("/var/lib/glm52-w9/failures")
CAPTURE = pathlib.Path(
    "/home/bmarti44/.local/state/glm52-w9-real-capture/"
    "attempt-73838408ccb1d126ade7b67c8d86fa00/on/capture"
)
REPLAY_ARTIFACTS = (
    "manifest.json", "raw.jsonl", "summary.json", "terminal-receipt.json")
REPLAY_DIR = "replay"
ATTESTATION_NAME = "root-attestation.json"
ATTEMPT = re.compile(r"attempt-[a-z0-9](?:[a-z0-9-]{0,79})$")
HASH40 = re.compile(r"[0-9a-f]{40}$")
HASH64 = re.compile(r"[0-9a-f]{64}$")
FIXED_ENVIRONMENT = {
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GLM52_W9_ROOT_RUNNER": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: pathlib.Path, *, required_uid: int = 0) -> str:
    root = pathlib.Path(root)
    root_info = root.lstat()
    if (not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode) or
            root_info.st_uid != required_uid or root_info.st_mode & 0o022):
        raise ValueError("installed tree root is mutable")
    rows: list[tuple[str, int, str]] = []
    for base, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort(); files.sort()
        base_path = pathlib.Path(base)
        base_info = base_path.lstat()
        if (not stat.S_ISDIR(base_info.st_mode) or stat.S_ISLNK(base_info.st_mode) or
                base_info.st_uid != required_uid or base_info.st_mode & 0o022):
            raise ValueError("installed tree directory is mutable")
        for name in files:
            path = base_path / name
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    info.st_uid != required_uid or info.st_mode & 0o022):
                raise ValueError("installed tree file is mutable")
            rows.append((path.relative_to(root).as_posix(), info.st_size, sha256_file(path)))
    digest = hashlib.sha256()
    for name, size, value in rows:
        digest.update(name.encode() + b"\0" + str(size).encode() + b"\0" + value.encode() + b"\n")
    return digest.hexdigest()


def strict_json(value: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in values:
            if key in result:
                raise ValueError(f"duplicate key in {label}")
            result[key] = item
        return result
    parsed = json.loads(
        value.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite value in {label}: {token}")),
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be an object")
    return parsed


class BoundOwnerInput:
    """Consume a bounded, invoking-owner, single-link regular-file generation."""

    def __init__(self, path: pathlib.Path, owner_uid: int, maximum_bytes: int) -> None:
        self.path = pathlib.Path(path)
        self.owner_uid = owner_uid
        self.maximum_bytes = maximum_bytes
        self.fd = -1
        self.identity: tuple[int, int, int, int, int] | None = None
        self.sha256 = ""

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _hash(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.pread(self.fd, 1 << 20, offset):
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()

    def __enter__(self) -> "BoundOwnerInput":
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.fd = os.open(self.path, flags)
        except OSError as error:
            raise ValueError("randomness receipt cannot be opened safely") from error
        try:
            info = os.fstat(self.fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    info.st_uid != self.owner_uid or info.st_size < 2 or
                    info.st_size > self.maximum_bytes):
                raise ValueError("randomness receipt identity or size is invalid")
            self.identity = self._identity(info)
            self.sha256 = self._hash()
            return self
        except BaseException:
            os.close(self.fd); self.fd = -1
            raise

    def read_bytes(self) -> bytes:
        if self.fd < 0 or self.identity is None:
            raise ValueError("randomness receipt is not open")
        result = bytearray()
        offset = 0
        while offset < self.identity[2]:
            chunk = os.pread(self.fd, self.identity[2] - offset, offset)
            if not chunk:
                raise ValueError("randomness receipt read was short")
            result.extend(chunk); offset += len(chunk)
        return bytes(result)

    def verify_final(self) -> None:
        if self.fd < 0 or self.identity is None:
            raise ValueError("randomness receipt is not open")
        descriptor = os.fstat(self.fd)
        pathname = os.stat(self.path, follow_symlinks=False)
        if (self._identity(descriptor) != self.identity or
                self._identity(pathname) != self.identity or
                descriptor.st_nlink != 1 or pathname.st_nlink != 1 or
                self._hash() != self.sha256):
            raise ValueError("randomness receipt changed")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if exc_type is None:
                self.verify_final()
        finally:
            if self.fd >= 0:
                os.close(self.fd); self.fd = -1


def parse_request(argv: list[str]) -> tuple[str, ...]:
    if len(argv) == 3 and argv[0] == "run" and ATTEMPT.fullmatch(argv[2]):
        receipt = pathlib.Path(argv[1])
        if receipt.is_absolute() and receipt.is_relative_to(OWNER_STATE):
            return tuple(argv)
    if len(argv) == 2 and argv[0] == "verify" and ATTEMPT.fullmatch(argv[1]):
        return tuple(argv)
    raise ValueError("usage: glm52-w9-submit run RECEIPT ATTEMPT | verify ATTEMPT")


def compare_replays(first: pathlib.Path, second: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in REPLAY_ARTIFACTS:
        left = (first / name).read_bytes()
        right = (second / name).read_bytes()
        if left != right:
            raise ValueError(f"fresh replay differs: {name}")
        result[name] = sha256_bytes(left)
    if {path.name for path in first.iterdir()} != set(REPLAY_ARTIFACTS) or {
            path.name for path in second.iterdir()} != set(REPLAY_ARTIFACTS):
        raise ValueError("fresh replay inventory differs")
    return result


def rename_noreplace(source: pathlib.Path, destination: pathlib.Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), str(destination))


def _fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def seal_tree(root: pathlib.Path, *, uid: int = 0, gid: int = 0) -> None:
    root = pathlib.Path(root)
    for base, directories, files in os.walk(root, topdown=False, followlinks=False):
        base_path = pathlib.Path(base)
        for name in files:
            path = base_path / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("publication contains unsafe file")
            os.chown(path, uid, gid); os.chmod(path, 0o444)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        for name in directories:
            path = base_path / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError("publication contains unsafe directory")
            os.chown(path, uid, gid); os.chmod(path, 0o555); _fsync_directory(path)
    os.chown(root, uid, gid); os.chmod(root, 0o555); _fsync_directory(root)


def build_attestation(*, attempt_name: str, candidate_hash: str,
                      review_commit: str, randomness_sha256: str,
                      randomness_round: int, replay_sha256: dict[str, str],
                      first_exit: int, second_exit: int,
                      runner_sha256: str = "0" * 64,
                      scorer_sha256: str = "0" * 64,
                      runtime_tree_sha256: str = "0" * 64,
                      capture_hashes: dict[str, str] | None = None,
                      execution_logs_sha256: dict[str, str] | None = None) -> dict[str, Any]:
    if (not ATTEMPT.fullmatch(attempt_name) or not HASH40.fullmatch(candidate_hash) or
            not HASH40.fullmatch(review_commit) or not HASH64.fullmatch(randomness_sha256) or
            type(randomness_round) is not int or randomness_round < 1 or
            any(not HASH64.fullmatch(value) for value in (
                runner_sha256, scorer_sha256, runtime_tree_sha256)) or
            set(replay_sha256) != set(REPLAY_ARTIFACTS) or
            any(not HASH64.fullmatch(value) for value in replay_sha256.values()) or
            first_exit != second_exit or first_exit not in (0, 3)):
        raise ValueError("root attestation inputs are invalid")
    return {
        "schema": "glm52-w9-root-attestation-v1",
        "attempt_name": attempt_name,
        "candidate_hash": candidate_hash,
        "review_commit": review_commit,
        "runner_sha256": runner_sha256,
        "scorer_sha256": scorer_sha256,
        "runtime_tree_sha256": runtime_tree_sha256,
        "capture_hashes": capture_hashes or {},
        "randomness_receipt_sha256": randomness_sha256,
        "randomness_round": randomness_round,
        "first_exit": first_exit,
        "second_exit": second_exit,
        "primary_sha256": replay_sha256,
        "replay_sha256": replay_sha256,
        "execution_logs_sha256": execution_logs_sha256 or {
            "first.log": "0" * 64, "second.log": "0" * 64},
        "randomness_artifact_sha256": randomness_sha256,
        "byte_identical": True,
    }


def _require_owned(path: pathlib.Path, uid: int, mode: int, *, directory: bool) -> None:
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (not expected(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != uid or
            stat.S_IMODE(info.st_mode) != mode or (not directory and info.st_nlink != 1)):
        raise ValueError("published ownership or mode mismatch")


def validate_published_attempt(attempt: pathlib.Path, attempt_name: str,
                               *, required_uid: int = 0) -> dict[str, Any]:
    attempt = pathlib.Path(attempt)
    if attempt.name != attempt_name or not ATTEMPT.fullmatch(attempt_name):
        raise ValueError("published attempt identity mismatch")
    _require_owned(attempt, required_uid, 0o555, directory=True)
    expected = set(REPLAY_ARTIFACTS) | {
        REPLAY_DIR, ATTESTATION_NAME, "randomness.json", "first.log", "second.log"}
    if {path.name for path in attempt.iterdir()} != expected:
        raise ValueError("root attestation or publication inventory is missing")
    replay = attempt / REPLAY_DIR
    _require_owned(replay, required_uid, 0o555, directory=True)
    if {path.name for path in replay.iterdir()} != set(REPLAY_ARTIFACTS):
        raise ValueError("replay inventory mismatch")
    for base in (attempt, replay):
        for name in REPLAY_ARTIFACTS:
            _require_owned(base / name, required_uid, 0o444, directory=False)
    attestation_path = attempt / ATTESTATION_NAME
    _require_owned(attestation_path, required_uid, 0o444, directory=False)
    attestation = strict_json(attestation_path.read_bytes(), "root attestation")
    if attestation.get("schema") != "glm52-w9-root-attestation-v1" or attestation.get(
            "attempt_name") != attempt_name or attestation.get("byte_identical") is not True:
        raise ValueError("root attestation attempt binding mismatch")
    for label, base in (("primary_sha256", attempt), ("replay_sha256", replay)):
        expected_hashes = attestation.get(label)
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(REPLAY_ARTIFACTS):
            raise ValueError("root attestation artifact inventory mismatch")
        for name, expected_hash in expected_hashes.items():
            if not HASH64.fullmatch(expected_hash) or sha256_file(base / name) != expected_hash:
                raise ValueError("root attestation artifact digest mismatch")
    if attestation["primary_sha256"] != attestation["replay_sha256"]:
        raise ValueError("root attestation replays differ")
    for name in ("randomness.json", "first.log", "second.log"):
        _require_owned(attempt / name, required_uid, 0o444, directory=False)
    if sha256_file(attempt / "randomness.json") != attestation.get(
            "randomness_artifact_sha256"):
        raise ValueError("root attestation randomness artifact differs")
    logs = attestation.get("execution_logs_sha256")
    if not isinstance(logs, dict) or set(logs) != {"first.log", "second.log"}:
        raise ValueError("root attestation log inventory differs")
    for name, expected_hash in logs.items():
        if not HASH64.fullmatch(expected_hash) or sha256_file(attempt / name) != expected_hash:
            raise ValueError("root attestation log differs")
    return attestation


def _load_install() -> dict[str, Any]:
    if os.geteuid() != 0:
        raise PermissionError("glm52-w9-submit must execute as root")
    _require_owned(INSTALL_ROOT, 0, 0o555, directory=True)
    _require_owned(INSTALL_MANIFEST, 0, 0o444, directory=False)
    manifest = strict_json(INSTALL_MANIFEST.read_bytes(), "install manifest")
    if (manifest.get("schema") != "glm52-w9-install-v1" or
            not HASH40.fullmatch(str(manifest.get("candidate_hash", ""))) or
            not HASH40.fullmatch(str(manifest.get("review_commit", ""))) or
            sha256_file(pathlib.Path("/proc/self/exe")) != manifest.get("python_sha256")):
        raise ValueError("installed W9 identity mismatch")
    for path, key, mode in ((RUNNER_PATH, "runner_sha256", 0o555),
                            (SCORER, "scorer_sha256", 0o555),
                            (NODE, "node_sha256", 0o555)):
        _require_owned(path, 0, mode, directory=False)
        if sha256_file(path) != manifest.get(key):
            raise ValueError("installed W9 component differs")
    if (tree_sha256(REPOSITORY) != manifest.get("repository_tree_sha256") or
            tree_sha256(NOBLE) != manifest.get("noble_tree_sha256")):
        raise ValueError("installed W9 tree differs")
    runtime_combined = hashlib.sha256(
        (tree_sha256(ROOT_NUMPY / "numpy") + "\0" +
         tree_sha256(ROOT_NUMPY / "numpy.libs")).encode()).hexdigest()
    if runtime_combined != manifest.get("runtime_tree_sha256"):
        raise ValueError("installed numerical runtime differs")
    return manifest


def _run_scorer(output: pathlib.Path, receipt: pathlib.Path) -> int:
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(SCORER),
         "--capture-root", str(CAPTURE), "--randomness-receipt", str(receipt),
         "--output", str(output)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=7200, check=False, cwd=REPOSITORY,
        env=FIXED_ENVIRONMENT,
    )
    (output.parent / f"{output.name}.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode not in (0, 3):
        raise ValueError("frozen scorer failed")
    return completed.returncode


def run_attempt(receipt_path: pathlib.Path, attempt_name: str) -> pathlib.Path:
    install = _load_install()
    for root in (STATE_ROOT, WORK_ROOT, PUBLISH_ROOT, FAILURE_ROOT):
        root.mkdir(mode=0o700 if root in (WORK_ROOT, FAILURE_ROOT) else 0o555,
                   parents=True, exist_ok=True)
        os.chown(root, 0, 0)
    for root, mode in ((STATE_ROOT, 0o555), (WORK_ROOT, 0o700),
                       (PUBLISH_ROOT, 0o555), (FAILURE_ROOT, 0o700)):
        _require_owned(root, 0, mode, directory=True)
    destination = PUBLISH_ROOT / attempt_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    workspace = pathlib.Path(tempfile.mkdtemp(prefix=f"{attempt_name}.", dir=WORK_ROOT))
    try:
        with BoundOwnerInput(receipt_path, OWNER_UID, 32768) as source:
            receipt_bytes = source.read_bytes()
            receipt = strict_json(receipt_bytes, "randomness receipt")
            records = receipt.get("relay_records")
            if not isinstance(records, list) or len(records) != 3 or not records[0] == records[1] == records[2]:
                raise ValueError("randomness receipt relay agreement is invalid")
            round_number = records[0].get("round") if isinstance(records[0], dict) else None
            if type(round_number) is not int:
                raise ValueError("randomness round is invalid")
            frozen_receipt = workspace / "randomness.json"
            frozen_receipt.write_bytes(receipt_bytes); os.chmod(frozen_receipt, 0o400)
        first, second = workspace / "first", workspace / "second"
        first_exit = _run_scorer(first, frozen_receipt)
        second_exit = _run_scorer(second, frozen_receipt)
        replay_hashes = compare_replays(first, second)
        shutil.move(str(second), str(first / REPLAY_DIR))
        shutil.move(str(workspace / "first.log"), str(first / "first.log"))
        shutil.move(str(workspace / "second.log"), str(first / "second.log"))
        shutil.copy2(frozen_receipt, first / "randomness.json")
        if _load_install() != install:
            raise ValueError("installed closure changed across replays")
        manifest = strict_json((first / "manifest.json").read_bytes(), "metric manifest")
        attestation = build_attestation(
            attempt_name=attempt_name, candidate_hash=install["candidate_hash"],
            review_commit=install["review_commit"], randomness_sha256=sha256_bytes(receipt_bytes),
            randomness_round=round_number, replay_sha256=replay_hashes,
            first_exit=first_exit, second_exit=second_exit,
            runner_sha256=install["runner_sha256"], scorer_sha256=install["scorer_sha256"],
            runtime_tree_sha256=install["runtime_tree_sha256"],
            capture_hashes=manifest.get("capture_hashes", {}),
            execution_logs_sha256={
                name: sha256_file(first / name) for name in ("first.log", "second.log")},
        )
        (first / ATTESTATION_NAME).write_text(
            json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8")
        seal_tree(first)
        validate_published_attempt(first, attempt_name)
        rename_noreplace(first, destination)
        _fsync_directory(PUBLISH_ROOT)
        validate_published_attempt(destination, attempt_name)
        return destination
    except BaseException:
        failure = FAILURE_ROOT / workspace.name
        if workspace.exists() and not failure.exists():
            rename_noreplace(workspace, failure)
        raise
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def main(argv: list[str] | None = None) -> int:
    request = parse_request(sys.argv[1:] if argv is None else argv)
    if os.geteuid() != 0:
        raise PermissionError("glm52-w9-submit must execute as root")
    if request[0] == "run":
        destination = run_attempt(pathlib.Path(request[1]), request[2])
        print(json.dumps({"status": "PUBLISHED", "path": str(destination)}, sort_keys=True))
        return 0
    attempt = PUBLISH_ROOT / request[1]
    attestation = validate_published_attempt(attempt, request[1])
    print(json.dumps(attestation, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PermissionError, subprocess.SubprocessError) as error:
        print(f"glm52-w9-submit: {error}", file=sys.stderr)
        raise SystemExit(1)
