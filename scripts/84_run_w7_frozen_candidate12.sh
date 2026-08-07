#!/usr/bin/env -S -i HOME=/home/bmarti44 PATH=/usr/bin:/bin /usr/bin/python3 -I -B
"""Execute the exact reviewed W7 harness from a kernel-sealed memory file."""

import fcntl
import hashlib
import json
import os
import pwd
import subprocess
import sys

REPO = "/home/bmarti44/spark-deepseek-v4-flash"
CANDIDATE_COMMIT = "f559546f3cbc51840819aa47fd84d9ef38fdf3a0"
HARNESS_SHA256 = "066a475ded03bd4a54ac27363b27fd77562d56340c66b4c7c69ddebf9ed5e91e"
HARNESS_PATH = "results/glm52-gates/harness/w7_resume_compiled_red_v1.sh"
BASE_ENV = {
    "HOME": "/home/bmarti44",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "USER": "bmarti44",
    "LOGNAME": "bmarti44",
}


def sealed_candidate() -> int:
    if pwd.getpwuid(os.getuid()).pw_name != "bmarti44":
        raise SystemExit(2)
    completed = subprocess.run(
        ["/usr/bin/git", "-C", REPO, "show", f"{CANDIDATE_COMMIT}:{HARNESS_PATH}"],
        env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=True,
    )
    candidate = completed.stdout
    if hashlib.sha256(candidate).hexdigest() != HARNESS_SHA256:
        raise SystemExit("frozen W7 candidate digest mismatch")
    fd = os.memfd_create("glm52-w7-frozen", os.MFD_ALLOW_SEALING)
    view = memoryview(candidate)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    seals = (
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_SEAL
    )
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != seals:
        raise SystemExit("frozen W7 candidate seal mismatch")
    sealed_bytes = os.pread(fd, len(candidate), 0)
    if sealed_bytes != candidate or hashlib.sha256(sealed_bytes).hexdigest() != HARNESS_SHA256:
        raise SystemExit("sealed W7 candidate verification failed")
    os.set_inheritable(fd, True)
    return fd


def self_test(fd: int) -> None:
    path = f"/proc/{os.getpid()}/fd/{fd}"
    mutation = r'''
import json
import os
import sys
path = sys.argv[1]
result = {}
def overwrite():
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.write(descriptor, b"mutated")
    finally:
        os.close(descriptor)
for name, operation in (
    ("write", overwrite),
    ("truncate", lambda: os.truncate(path, 0)),
):
    try:
        operation()
    except OSError as error:
        result[name] = error.errno
    else:
        result[name] = 0
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if all(result.values()) else 1)
'''
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-B", "-c", mutation, path],
        env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
        pass_fds=(fd,),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("sealed W7 mutation unexpectedly succeeded")
    if hashlib.sha256(os.pread(fd, os.fstat(fd).st_size, 0)).hexdigest() != HARNESS_SHA256:
        raise SystemExit("sealed W7 bytes changed after mutation")
    print("W7_SEALED_MUTATION_REJECTED", result.stdout.strip())
    env = dict(BASE_ENV)
    env.update(
        W7_EXECUTED_HARNESS_SHA256=HARNESS_SHA256,
        W7_FROZEN_CANDIDATE_COMMIT=CANDIDATE_COMMIT,
    )
    completed = subprocess.run(
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            f"/proc/self/fd/{fd}",
            "--validate-execution-authority",
            HARNESS_SHA256,
            CANDIDATE_COMMIT,
        ],
        env=env,
        pass_fds=(fd,),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("W7_FROZEN_LAUNCHER_OK")


def main() -> None:
    os.environ.clear()
    os.environ.update(HOME="/home/bmarti44", PATH="/usr/bin:/bin")
    fd = sealed_candidate()
    if sys.argv[1:] == ["--self-test"]:
        self_test(fd)
        return
    if len(sys.argv) != 1:
        raise SystemExit(2)
    env = dict(BASE_ENV)
    env.update(
        W7_EXECUTED_HARNESS_SHA256=HARNESS_SHA256,
        W7_FROZEN_CANDIDATE_COMMIT=CANDIDATE_COMMIT,
    )
    os.execve(
        "/usr/bin/bash",
        ["/usr/bin/bash", "--noprofile", "--norc", f"/proc/self/fd/{fd}"],
        env,
    )


if __name__ == "__main__":
    main()
