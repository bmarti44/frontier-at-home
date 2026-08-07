#!/usr/bin/env -S -i HOME=/home/bmarti44 PATH=/usr/bin:/bin /usr/bin/python3 -I -B
"""Run the exact W7 equivalence inputs from kernel-sealed memory files."""

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import re
import subprocess


REPO = "/home/bmarti44/spark-deepseek-v4-flash"
CANDIDATE_COMMIT = "713c93dbc5bd7657e419241b98d532495a5a398d"
FROZEN = {
    "harness": (
        "results/glm52-gates/harness/w7_resume_equivalence_v1.sh",
        "532d711bf6370153aa60bfc6ec24e0501b17f98f70b5128f30c808f6e10653ce",
    ),
    "scorer": (
        "scripts/85_score_w7_resume_equivalence.py",
        "197131308e73bb83a783f3d697157f78b0802c04ec2e14f280c609a934ff20ce",
    ),
    "trace_scorer": (
        "scripts/83_score_w7_deployed_trace.py",
        "6cec5063906a52c577617b4173a1deed14d0ae2fffebff19bbef6e96442dc985",
    ),
}
SEALS = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_SEAL
)
BASE_ENV = {
    "HOME": "/home/bmarti44",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "USER": "bmarti44",
    "LOGNAME": "bmarti44",
}


def _git_blob(path: str) -> bytes:
    result = subprocess.run(
        ["/usr/bin/git", "-C", REPO, "show", f"{CANDIDATE_COMMIT}:{path}"],
        env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=True,
    )
    return result.stdout


def _seal(name: str, path: str, expected_sha256: str) -> int:
    content = _git_blob(path)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise SystemExit(f"frozen {name} digest mismatch")
    fd = os.memfd_create(f"glm52-w7-{name}", os.MFD_ALLOW_SEALING)
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != SEALS:
        raise SystemExit(f"frozen {name} seal mismatch")
    sealed = os.pread(fd, len(content), 0)
    if sealed != content or hashlib.sha256(sealed).hexdigest() != expected_sha256:
        raise SystemExit(f"sealed {name} verification failed")
    os.set_inheritable(fd, True)
    return fd


def _sealed_inputs() -> dict[str, int]:
    if pwd.getpwuid(os.getuid()).pw_name != "bmarti44":
        raise SystemExit(2)
    return {
        name: _seal(name, path, digest)
        for name, (path, digest) in FROZEN.items()
    }


def _mutation_test(fds: dict[str, int]) -> None:
    results = {}
    for name, fd in fds.items():
        path = f"/proc/{os.getpid()}/fd/{fd}"
        mutation_fd = None
        try:
            mutation_fd = os.open(path, os.O_WRONLY)
            os.write(mutation_fd, b"mutated")
        except OSError as error:
            results[name] = error.errno
        else:
            results[name] = 0
        finally:
            if mutation_fd is not None:
                os.close(mutation_fd)
        expected = FROZEN[name][1]
        actual = hashlib.sha256(os.pread(fd, os.fstat(fd).st_size, 0)).hexdigest()
        if actual != expected:
            raise SystemExit(f"sealed {name} changed after mutation")
    if not all(results.values()):
        raise SystemExit("sealed W7 mutation unexpectedly succeeded")
    print("W7_EQUIVALENCE_SEALS_OK", json.dumps(results, sort_keys=True))


def _environment(fds: dict[str, int], seed_sha256: str) -> dict[str, str]:
    environment = dict(BASE_ENV)
    environment.update(
        W7_EXECUTED_HARNESS_SHA256=FROZEN["harness"][1],
        W7_FROZEN_CANDIDATE_COMMIT=CANDIDATE_COMMIT,
        W7_RANDOM_SEED_SHA256=seed_sha256,
        W7_SEALED_HARNESS_FD=str(fds["harness"]),
        W7_SEALED_SCORER_FD=str(fds["scorer"]),
        W7_SEALED_TRACE_SCORER_FD=str(fds["trace_scorer"]),
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    os.environ.clear()
    os.environ.update(HOME="/home/bmarti44", PATH="/usr/bin:/bin")
    fds = _sealed_inputs()
    _mutation_test(fds)
    if args.self_test:
        if args.seed_sha256 is not None:
            raise SystemExit(2)
        return
    if args.seed_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", args.seed_sha256) is None:
        raise SystemExit("--seed-sha256 must be one lowercase SHA-256 digest")
    environment = _environment(fds, args.seed_sha256)
    os.execve(
        "/usr/bin/bash",
        [
            "/usr/bin/bash", "--noprofile", "--norc",
            f"/proc/self/fd/{fds['harness']}",
        ],
        environment,
    )


if __name__ == "__main__":
    main()
