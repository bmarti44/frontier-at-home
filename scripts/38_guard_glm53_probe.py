#!/usr/bin/env python3
"""Evidence-only Python identity monitor inside the established GLM wrapper.

The child blocks before importing the probe until the parent verifies its real
executable, command, environment, start ticks and cgroup. Sampling continues
until exit. This is not a production server lifecycle or a host-safety scorer;
the enclosing cgroup wrapper remains responsible for memory, timeout and escaped
descendants. Never use this instrumentation for headline serving performance.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import runpy
import resource
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


def pidfd_open(pid):
    """Use the same Linux pidfd ABI when standalone CPython omits its binding."""
    require(type(pid) is int and 0 < pid <= 2147483647, 'invalid pidfd process ID')
    native = getattr(os, 'pidfd_open', None)
    if native is not None: return native(pid)
    function = ctypes.CDLL(None, use_errno=True).pidfd_open
    function.argtypes = [ctypes.c_int, ctypes.c_uint]; function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    return descriptor


def pidfd_send_signal(descriptor, number):
    require(type(descriptor) is int and 0 <= descriptor <= 2147483647 and isinstance(number, int) and
            not isinstance(number, bool) and 0 <= number < signal.NSIG, 'invalid pidfd signal arguments')
    native = getattr(signal, 'pidfd_send_signal', None)
    if native is not None: return native(descriptor, number)
    function = ctypes.CDLL(None, use_errno=True).pidfd_send_signal
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]; function.restype = ctypes.c_int
    result = function(descriptor, number, None, 0)
    if result < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    require(result == 0, 'unexpected pidfd signal result')


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


def terminate_group(pgid, leader_start_ticks):
    """Signal only identity-rechecked pidfds, including surviving descendants."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        anchor = process_stat(pgid)
        require(anchor["start_ticks"] == leader_start_ticks and anchor["pgid"] == pgid and
                anchor["ppid"] == os.getpid(), "cleanup group anchor changed")
        members = live_group(pgid)
        if not members:
            return []
        for member in members:
            descriptor = None
            try:
                descriptor = pidfd_open(member["pid"])
                current = process_stat(member["pid"])
                require(current["start_ticks"] == member["start_ticks"] and current["pgid"] == pgid, "cleanup process identity changed")
                pidfd_send_signal(descriptor, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        time.sleep(0.05)
    return live_group(pgid)


def exit_observation(pid):
    # Keep the leader waitable and its PID/PGID reserved through group cleanup.
    value = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if value is None:
        return None
    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status


def snapshot(pid, expected, first=False, terminal=False):
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
    status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines())
    filters = int(status["Seccomp_filters"].strip())
    if terminal:
        require(status["NoNewPrivs"].strip() == "1" and status["Seccomp"].strip() == "2" and
                filters > expected["initial_seccomp_filters"], "terminal exec filter is missing")
    after = process_stat(pid)
    require(after["start_ticks"] == before["start_ticks"] and after["pgid"] == pid, "process identity changed during sample")
    for path, digest in expected["source_hashes"].items():
        require(sha(path) == digest, "frozen probe source changed")
    return {"event": "identity", "pid": pid, "start_ticks": before["start_ticks"], "pgid": pid,
            "cgroup": expected["cgroup"], "executable_verified": True, "argv_verified": True,
            "environment_verified": True, "binary_sha256": expected["binary_sha256"],
            "seccomp_filters": filters, "terminal_exec_filter_verified": terminal}


def prohibit_terminal_exec():
    """Deny replacement after the final sample, including every existing thread.

    Installed only after the frozen probe has returned. Normal Python/C exit
    handlers still run; an exec attempt terminates the process with SIGSYS.
    This guard is selected only for evidence probes, never serving.
    Linux UAPI: seccomp(2), SECCOMP_FILTER_FLAG_TSYNC, RET_KILL_PROCESS.
    """
    architectures = {
        "aarch64": (0xC00000B7, 277, (221, 281)),
        "x86_64": (0xC000003E, 317, (59, 322)),
    }
    require(platform.machine() in architectures, "unsupported terminal filter architecture")
    arch, syscall, exec_calls = architectures[platform.machine()]

    class Instruction(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Instruction))]

    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, arch), (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0)]
    if platform.machine() == "x86_64":
        # This CPython uses the native ABI; reject all x32 syscall encodings.
        instructions.extend([(0x45, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000)])
    for number in exec_calls:
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x80000000)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (Instruction * len(instructions))(*(Instruction(*row) for row in instructions))
    program = Program(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    libc.syscall.restype = ctypes.c_long
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    require(libc.prctl(38, 1, 0, 0, 0) == 0, "cannot set no-new-privileges for terminal filter")
    result = libc.syscall(ctypes.c_long(syscall), ctypes.c_uint(1), ctypes.c_uint(1), ctypes.byref(program))
    require(result == 0, f"terminal filter synchronization failed: result={result} errno={ctypes.get_errno()}")


def child():
    ready, release = int(sys.argv[2]), int(sys.argv[3])
    target = Path(sys.argv[4]).resolve(strict=True)
    os.set_inheritable(ready, False)
    os.set_inheritable(release, False)
    os.write(ready, b"R")
    require(os.read(release, 1) == b"G", "parent did not release verified child")
    sys.argv = [str(target), *sys.argv[5:]]
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as error:
        if error.code is not None and error.code != 0:
            raise
    prohibit_terminal_exec()
    os.write(ready, b"C")
    require(os.read(release, 1) == b"E", "parent did not verify probe completion")
    os.close(ready)
    os.close(release)


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
    manifest = {k: v for k, v in expected.items() if not k.endswith("_bytes") and k != "source_hashes"}
    manifest["source_files"] = [{"path": path, "sha256": digest} for path, digest in expected["source_hashes"].items()]
    manifest.update(argv=argv, environment_sha256=hashlib.sha256(b"\0".join(expected["environment_bytes"])).hexdigest(),
                    qualification="Python_probe_identity_only", period_seconds=0.25, start_unix=time.time())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    process = None
    leader_start_ticks = None
    samples = 0
    failure = None
    survivors = []
    with (output / "raw.jsonl").open("w") as raw:
        def record(row):
            raw.write(json.dumps({"time_unix": time.time(), "monotonic_ns": time.monotonic_ns(), **row}, allow_nan=False) + "\n")
            raw.flush()
        try:
            process = subprocess.Popen(argv, env=environment, pass_fds=(ready_write, release_read), start_new_session=True)
            leader_start_ticks = process_stat(process.pid)["start_ticks"]
            os.close(ready_write); ready_write = None
            os.close(release_read); release_read = None
            require(bool(select.select([ready_read], [], [], 10)[0]) and os.read(ready_read, 1) == b"R", "child readiness timeout or missing handshake")
            value = snapshot(process.pid, expected, first=True)
            expected["start_ticks"] = value["start_ticks"]
            expected["initial_seccomp_filters"] = value["seccomp_filters"]
            record(value); samples += 1
            os.write(release_write, b"G")
            completed = False
            while exit_observation(process.pid) is None:
                readable = select.select([ready_read], [], [], 0.25)[0]
                if readable:
                    marker = os.read(ready_read, 1)
                    if marker != b"C" and exit_observation(process.pid) is None:
                        try:
                            snapshot(process.pid, expected)
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                    require(marker == b"C", "missing verified completion handshake")
                    value = snapshot(process.pid, expected, terminal=True)
                    value["completion_verified"] = True
                    record(value); samples += 1
                    completed = True
                    os.write(release_write, b"E")
                    os.close(release_write); release_write = None
                    break
                try:
                    value = snapshot(process.pid, expected)
                except (FileNotFoundError, ProcessLookupError):
                    raise ValueError("process exited without verified completion")
                record(value); samples += 1
            require(completed, "missing verified completion handshake")
            deadline = time.monotonic() + 5
            while exit_observation(process.pid) is None and time.monotonic() < deadline:
                time.sleep(0.05)
            result = exit_observation(process.pid)
            require(result == 0, f"probe exit status {result}")
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
                try:
                    require(leader_start_ticks is not None, "missing cleanup group anchor")
                    survivors = terminate_group(process.pid, leader_start_ticks)
                except Exception as error:
                    failure = failure or repr(error)
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
