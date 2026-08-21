#!/usr/bin/env python3
"""Fixtures for executing engine-switch production functions safely.

The switch is sourced with ``ENGINE_SWITCH_TESTING`` absent.  The only test
seam is a source-only fixture root, so the production function definitions are
used while state, artifacts, repository helpers, and service-manager calls are
contained below a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "52_engine_switch.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proc_identity(pid: int) -> tuple[int, int]:
    """Return the real process group and Linux start ticks for *pid*."""

    stat_line = Path(f"/proc/{pid}/stat").read_text()
    fields = stat_line.rsplit(") ", 1)[1].split()
    return int(fields[2]), int(fields[19])


class SwitchSafetyFixture:
    """A rootless fixture for production-path switch function calls."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="switch-safety-fixture-"
        )
        self.root = Path(self._temporary.name)
        self.state = self.root / "state"
        self.repo = self.root / "repo"
        self.shims = self.root / "shims"
        self.artifacts = self.root / "artifacts"
        self._children: list[subprocess.Popen[str]] = []
        for directory in (
            self.state,
            self.repo / "scripts",
            self.repo / "configs" / "build-manifests",
            self.shims,
            self.artifacts / "bin",
            self.artifacts / "models",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._install_systemctl_shim()
        self._install_guard_shim()
        self._install_memwatch_fixture()
        self.set_systemctl_responses()

    def __enter__(self) -> "SwitchSafetyFixture":
        return self

    def __exit__(self, *_args: object) -> None:
        for child in reversed(self._children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in reversed(self._children):
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=2)
        self._temporary.cleanup()

    def _write_executable(self, path: Path, source: str) -> None:
        path.write_text(source)
        path.chmod(0o755)

    def _install_systemctl_shim(self) -> None:
        self._write_executable(
            self.shims / "systemctl",
            """#!/usr/bin/python3
import json
import os
import pathlib
import sys

root = pathlib.Path(os.environ["SWITCH_SHIM_ROOT"])
responses_path = root / "systemctl.responses.json"
counter_path = root / "systemctl.counter"
calls_path = root / "systemctl.calls.jsonl"
responses = json.loads(responses_path.read_text())
try:
    index = int(counter_path.read_text())
except (FileNotFoundError, ValueError):
    index = 0
with calls_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:], separators=(",", ":")) + "\\n")
if index >= len(responses):
    print("unexpected systemctl call", file=sys.stderr)
    raise SystemExit(97)
counter_path.write_text(str(index + 1))
response = responses[index]
expected = response.get("argv_prefix")
if expected is not None and sys.argv[1:1 + len(expected)] != expected:
    print(f"systemctl argv mismatch: {sys.argv[1:]!r}", file=sys.stderr)
    raise SystemExit(98)
sys.stdout.write(response.get("stdout", ""))
sys.stderr.write(response.get("stderr", ""))
raise SystemExit(response.get("returncode", 0))
""",
        )

    def _install_guard_shim(self) -> None:
        self._write_executable(
            self.repo / "scripts" / "03_memory_guard.py",
            """#!/usr/bin/python3
import os
import pathlib
import sys

pathlib.Path(os.environ["SWITCH_FIXTURE_ROOT"], "guard.called").write_text(
    " ".join(sys.argv[1:])
)
raise SystemExit(int(os.environ.get("SWITCH_GUARD_EXIT", "1")))
""",
        )

    def _install_memwatch_fixture(self) -> None:
        self._write_executable(
            self.repo / "scripts" / "01_memwatch.sh",
            """#!/usr/bin/env bash
set -Eeuo pipefail
target=
ready=
while (( $# )); do
    case $1 in
        --target-file) target=$2; shift 2 ;;
        --ready-file) ready=$2; shift 2 ;;
        *) shift ;;
    esac
done
[[ -n $target && -n $ready ]]
trap 'exit 0' TERM INT
while [[ ! -s $target ]]; do /usr/bin/sleep 0.01; done
read -r command pid pgid ticks <"$target"
[[ $command == DISARM ]]
if [[ ${SWITCH_MEMWATCH_MODE:-ack} == ack ]]; then
    printf 'DISARMED %s %s %s\n' "$pid" "$pgid" "$ticks" >"$ready"
fi
""",
        )

    def set_systemctl_responses(self, *responses: dict[str, Any]) -> None:
        (self.root / "systemctl.responses.json").write_text(
            json.dumps(list(responses))
        )
        (self.root / "systemctl.counter").write_text("0")
        (self.root / "systemctl.calls.jsonl").unlink(missing_ok=True)

    def systemctl_calls(self) -> list[list[str]]:
        path = self.root / "systemctl.calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def run_function(
        self, body: str, *, timeout: float = 10
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("ENGINE_SWITCH_TESTING", None)
        env.pop("ENGINE_SWITCH_TEST_ROOT", None)
        env.update(
            {
                "ENGINE_SWITCH_SOURCE_ONLY_FIXTURE_ROOT": str(self.root),
                "PATH": f"{self.shims}:/usr/bin:/bin",
                "SWITCH_FIXTURE_ROOT": str(self.root),
                "SWITCH_GUARD_EXIT": "1",
                "SWITCH_SHIM_ROOT": str(self.root),
            }
        )
        command = f'source "$1"\n{body}'
        return subprocess.run(
            ["bash", "-c", command, "switch-safety", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

    def spawn_sleep(self) -> subprocess.Popen[str]:
        child = subprocess.Popen(
            ["/usr/bin/sleep", "60"],
            text=True,
            start_new_session=True,
        )
        self._children.append(child)
        proc_identity(child.pid)
        return child

    def stop_child(self, child: subprocess.Popen[str]) -> None:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=2)

    def spawn_memwatch(self, mode: str = "ack") -> subprocess.Popen[str]:
        env = os.environ.copy()
        env["SWITCH_MEMWATCH_MODE"] = mode
        child = subprocess.Popen(
            [
                str(self.repo / "scripts" / "01_memwatch.sh"),
                "--target-file",
                str(self.state / "qwen38.memwatch.target"),
                "--ready-file",
                str(self.state / "qwen38.memwatch.ready"),
            ],
            env=env,
            text=True,
            start_new_session=True,
        )
        self._children.append(child)
        proc_identity(child.pid)
        # Reap promptly so the production ``/proc/$pid`` disappearance check
        # observes a completed memwatch rather than this test parent retaining
        # it as a zombie until fixture teardown.
        threading.Thread(target=child.wait, daemon=True).start()
        return child

    def write_qwen_record(
        self,
        engine_pid: int,
        engine_pgid: int,
        engine_ticks: int,
        *,
        memwatch_pid: int,
        memwatch_ticks: int,
        exe_sha256: str = "0" * 64,
    ) -> Path:
        record = self.state / "qwen38.process.json"
        record.write_text(
            json.dumps(
                {
                    "pid": engine_pid,
                    "pgid": engine_pgid,
                    "start_ticks": engine_ticks,
                    "exe_sha256": exe_sha256,
                    "unit": "qwen38-engine.service",
                    "memwatch_pid": memwatch_pid,
                    "memwatch_start_ticks": memwatch_ticks,
                }
            )
        )
        return record

    def process_exe_sha256(self, pid: int) -> str:
        return _sha256(Path(f"/proc/{pid}/exe"))

    def install_laguna_artifacts(self) -> dict[str, Path]:
        binary = self.artifacts / "bin" / "laguna-server"
        library = self.artifacts / "bin" / "liblaguna.so"
        model_prefix = self.artifacts / "models" / "laguna"
        shards = [
            Path(f"{model_prefix}-0000{index}-of-00003.gguf")
            for index in (1, 2, 3)
        ]
        draft = self.artifacts / "models" / "laguna-dflash.gguf"
        payloads = {
            binary: b"fixture-laguna-binary\n",
            library: b"fixture-laguna-library\n",
            shards[0]: b"fixture-laguna-shard-one\n",
            shards[1]: b"fixture-laguna-shard-two\n",
            shards[2]: b"fixture-laguna-shard-three\n",
            draft: b"fixture-laguna-draft\n",
        }
        for path, payload in payloads.items():
            path.write_bytes(payload)

        profile = {
            "schema_version": 3,
            "profile": "laguna",
            "port": 8013,
            "context_cap": 393216,
            "binary_path": str(binary),
            "binary_bytes": binary.stat().st_size,
            "binary_sha256": _sha256(binary),
            "model_path": str(shards[0]),
            "model_sha256": _sha256(shards[0]),
            "model_shards": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in shards
            ],
            "draft_model_path": str(draft),
            "draft_model_bytes": draft.stat().st_size,
            "draft_model_sha256": _sha256(draft),
        }
        (self.repo / "configs" / "laguna-production-profile.json").write_text(
            json.dumps(profile)
        )
        build = {
            "schema_version": 1,
            "binaries": {
                "llama-server": {
                    "path": str(binary),
                    "sha256": _sha256(binary),
                }
            },
            "shared_libraries": {
                library.name: {"sha256": _sha256(library)}
            },
        }
        (
            self.repo
            / "configs"
            / "build-manifests"
            / "llamacpp-laguna-06f8cebd.json"
        ).write_text(json.dumps(build))
        return {
            "binary": binary,
            "library": library,
            "shard1": shards[0],
            "shard2": shards[1],
            "shard3": shards[2],
            "draft": draft,
        }

    @staticmethod
    def tamper_same_size(path: Path) -> None:
        original = path.read_bytes()
        replacement = bytes([original[0] ^ 0x01]) + original[1:]
        path.write_bytes(replacement)

    def wait_for_exit(self, child: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + 2
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if child.poll() is None:
            raise AssertionError(f"fixture process {child.pid} did not exit")
