#!/usr/bin/env python3
"""Evidence-only Python identity monitor inside the established GLM wrapper.

The child blocks before importing the probe until the parent verifies its real
executable, command, environment, start ticks and cgroup. Sampling continues
until exit. This is not a production server lifecycle or a host-safety scorer;
the enclosing cgroup wrapper remains responsible for memory, timeout and escaped
descendants. Never use this instrumentation for headline serving performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import select
import signal
import subprocess
import sys
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def process_stat(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    tail = text.rsplit(")", 1)[1].split()
    return {"state": tail[0], "ppid": int(tail[1]), "pgid": int(tail[2]), "start_ticks": int(tail[19])}


def live_group(pgid):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            value = process_stat(int(path.name))
            if value["pgid"] == pgid and value["state"] not in ("Z", "X"):
                result.append({"pid": int(path.name), **value})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def terminate_group(pgid):
    """Signal only identity-rechecked pidfds, including surviving descendants."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        members = live_group(pgid)
        if not members:
            return []
        for member in members:
            descriptor = None
            try:
                descriptor = os.pidfd_open(member["pid"])
                current = process_stat(member["pid"])
                require(current["start_ticks"] == member["start_ticks"] and current["pgid"] == pgid, "cleanup process identity changed")
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        time.sleep(0.05)
    return live_group(pgid)


def snapshot(pid, expected, first=False):
    before = process_stat(pid)
    if before["state"] in ("Z", "X"):
        raise ProcessLookupError("process exited during identity observation")
    require(before["ppid"] == os.getpid() and before["pgid"] == pid, "process parent/group identity changed")
    if not first:
        require(before["start_ticks"] == expected["start_ticks"], "process start ticks changed")
    proc = Path(f"/proc/{pid}")
    executable = (proc / "exe").resolve(strict=True)
    info = (proc / "exe").stat()
    require(str(executable) == expected["executable"] and [info.st_dev, info.st_ino] == expected["device_inode"] and
            sha(proc / "exe") == expected["binary_sha256"], "executable identity changed")
    require((proc / "cmdline").read_bytes() == expected["argv_bytes"], "argv identity changed")
    environment = (proc / "environ").read_bytes().split(b"\0")
    require(environment[-1] == b"" and sorted(environment[:-1]) == expected["environment_bytes"], "environment identity changed")
    require((proc / "cgroup").read_text() == expected["cgroup"], "cgroup identity changed")
    after = process_stat(pid)
    require(after["start_ticks"] == before["start_ticks"] and after["pgid"] == pid, "process identity changed during sample")
    for path, digest in expected["source_hashes"].items():
        require(sha(path) == digest, "frozen probe source changed")
    return {"event": "identity", "pid": pid, "start_ticks": before["start_ticks"], "pgid": pid,
            "cgroup": expected["cgroup"], "executable_verified": True, "argv_verified": True,
            "environment_verified": True, "binary_sha256": expected["binary_sha256"]}


def child():
    ready, release = int(sys.argv[2]), int(sys.argv[3])
    target = Path(sys.argv[4]).resolve(strict=True)
    os.write(ready, b"R")
    os.close(ready)
    require(os.read(release, 1) == b"G", "parent did not release verified child")
    os.close(release)
    sys.argv = [str(target), *sys.argv[5:]]
    runpy.run_path(str(target), run_name="__main__")


def main():
    require(not sys.flags.optimize, "optimized Python is forbidden")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(bool(command), "probe script required")
    target = Path(command[0]).resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    guard = Path(__file__).resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    argv = [str(executable), "-I", "-B", str(guard), "--child", str(ready_write), str(release_read), str(target), *command[1:]]
    environment = dict(os.environ)
    info = executable.stat()
    expected = {"executable": str(executable), "device_inode": [info.st_dev, info.st_ino],
                "binary_sha256": sha(executable), "argv_bytes": b"\0".join(os.fsencode(v) for v in argv) + b"\0",
                "environment_bytes": sorted(os.fsencode(k + "=" + v) for k, v in environment.items()),
                "cgroup": Path("/proc/self/cgroup").read_text(),
                "source_hashes": {str(guard): sha(guard), str(target): sha(target)}}
    manifest = {k: v for k, v in expected.items() if not k.endswith("_bytes")}
    manifest.update(argv=argv, environment_sha256=hashlib.sha256(b"\0".join(expected["environment_bytes"])).hexdigest(),
                    qualification="Python_probe_identity_only", period_seconds=0.25, start_unix=time.time())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    process = None
    samples = 0
    failure = None
    survivors = []
    with (output / "raw.jsonl").open("w") as raw:
        def record(row):
            raw.write(json.dumps({"time_unix": time.time(), "monotonic_ns": time.monotonic_ns(), **row}, allow_nan=False) + "\n")
            raw.flush()
        try:
            process = subprocess.Popen(argv, env=environment, pass_fds=(ready_write, release_read), start_new_session=True)
            os.close(ready_write); ready_write = None
            os.close(release_read); release_read = None
            require(bool(select.select([ready_read], [], [], 10)[0]) and os.read(ready_read, 1) == b"R", "child readiness timeout or missing handshake")
            value = snapshot(process.pid, expected, first=True)
            expected["start_ticks"] = value["start_ticks"]
            record(value); samples += 1
            os.write(release_write, b"G")
            os.close(release_write); release_write = None
            while process.poll() is None:
                time.sleep(0.25)
                if process.poll() is not None:
                    break
                try:
                    value = snapshot(process.pid, expected)
                except (FileNotFoundError, ProcessLookupError):
                    # A confirmed exit during a sample is distinct from an
                    # identity contradiction while the process remains live.
                    if process.poll() is not None:
                        break
                    if process_stat(process.pid)["state"] in ("Z", "X"):
                        process.wait(timeout=2)
                        break
                    raise
                record(value); samples += 1
            require(process.returncode == 0, f"probe exit status {process.returncode}")
            require(samples >= 2, "insufficient continuous identity coverage")
            survivors = live_group(process.pid)
            require(not survivors, "surviving probe descendant")
        except Exception as error:
            failure = repr(error)
            record({"event": "failure", "failure": failure})
        finally:
            for descriptor in (ready_read, ready_write, release_read, release_write):
                if descriptor is not None:
                    os.close(descriptor)
            if process is not None:
                survivors = terminate_group(process.pid)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    failure = failure or "probe survived cleanup timeout"
            if survivors:
                failure = failure or "probe descendants survived cleanup"
            record({"event": "cleanup", "live_process_group_after": survivors})
    summary = {"verdict": "PASS" if failure is None else "FAIL", "qualification": "Python_probe_identity_only",
               "identity_samples": samples, "probe_exit_code": process.returncode if process else None,
               "live_process_group_after": survivors, "raw_sha256": sha(output / "raw.jsonl"), "failure": failure}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(0 if failure is None else 1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child()
    else:
        main()
