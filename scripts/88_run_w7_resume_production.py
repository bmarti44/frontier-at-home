#!/usr/bin/env -S -i HOME=/home/bmarti44 PATH=/usr/bin:/bin /usr/bin/python3 -I -B
"""Run the exact W7 production equivalence inputs from kernel-sealed memory files."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys


REPO = "/home/bmarti44/spark-deepseek-v4-flash"
DRAND_TARGET_ROUND = 6356615
CANDIDATE_COMMIT = "6b6ca50b3a9ee0f93c981a572508383691707ca6"
HARNESS_SHA256 = "ddc2c790a2db57cb8f53358d3077ce6d25d850952f296a7361e9029ed11a2b19"
SCORER_SHA256 = "3ac564627df70cbc0fd103679d7126c6f6815f7f879f56d2cf928e0475fdebea"
TRACE_SCORER_SHA256 = "6cec5063906a52c577617b4173a1deed14d0ae2fffebff19bbef6e96442dc985"
DRAND_VERIFIER_SHA256 = "c191d301e1ff8460fffaea9dfeaab7d0fce0d63f92d3fdfcfa20442ccfdc2131"
DRAND_NODE_SHA256 = "3159f9115ab4be7d318b7c28e946837a4dceb7f2b3c43232aa2f2e3852550b90"
DRAND_RUNTIME_TREE_SHA256 = "38161b0df115fbd3c2e1dda87759a1eb1e87500109661f0f2fa18874d6e1a0e4"
DRAND_NODE = Path("/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node")
DRAND_RUNTIME = Path("/home/bmarti44/.cache/glm52-drand-client-1.4.2")
DRAND_VERIFIER = Path(REPO) / "scripts/89_verify_drand_receipt.mjs"
FROZEN = {
    "harness": (
        "results/glm52-gates/harness/w7_resume_production_v1.sh",
        HARNESS_SHA256,
    ),
    "scorer": (
        "scripts/87_score_w7_resume_production.py",
        SCORER_SHA256,
    ),
    "trace_scorer": (
        "scripts/83_score_w7_deployed_trace.py",
        TRACE_SCORER_SHA256,
    ),
    "drand_verifier": (
        "scripts/89_verify_drand_receipt.mjs",
        DRAND_VERIFIER_SHA256,
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
EXECUTED_LAUNCHER_COMMIT = ""
EXECUTED_LAUNCHER_SHA256 = ""


def _sealed_launcher_reexec() -> None:
    global EXECUTED_LAUNCHER_COMMIT, EXECUTED_LAUNCHER_SHA256
    marker = os.environ.get("W7_SEALED_LAUNCHER_FD")
    if marker is not None:
        if not marker.isdigit():
            raise SystemExit("invalid sealed launcher descriptor")
        fd = int(marker)
        if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != SEALS:
            raise SystemExit("launcher descriptor is not sealed")
        content = os.pread(fd, os.fstat(fd).st_size, 0)
        expected = os.environ.get("W7_EXECUTED_LAUNCHER_SHA256", "")
        commit = os.environ.get("W7_EXECUTED_LAUNCHER_COMMIT", "")
        if (
            re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or hashlib.sha256(content).hexdigest() != expected
        ):
            raise SystemExit("sealed launcher digest mismatch")
        clean = subprocess.run(
            ["/usr/bin/git", "-C", REPO, "status", "--porcelain"],
            env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
            capture_output=True, check=True,
        )
        head = subprocess.run(
            ["/usr/bin/git", "-C", REPO, "rev-parse", "HEAD"],
            env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
            capture_output=True, check=True, text=True,
        ).stdout.strip()
        blob = subprocess.run(
            ["/usr/bin/git", "-C", REPO, "show",
             f"{commit}:scripts/88_run_w7_resume_production.py"],
            env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
            capture_output=True, check=True,
        ).stdout
        if clean.stdout or commit != head or blob != content:
            raise SystemExit("sealed launcher is not the clean HEAD blob")
        EXECUTED_LAUNCHER_COMMIT = os.environ.get(
            "W7_EXECUTED_LAUNCHER_COMMIT", ""
        )
        EXECUTED_LAUNCHER_SHA256 = expected
        return
    status = subprocess.run(
        ["/usr/bin/git", "-C", REPO, "status", "--porcelain"],
        env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
        capture_output=True, check=True,
    )
    if status.stdout:
        raise SystemExit("repository is not clean")
    head = subprocess.run(
        ["/usr/bin/git", "-C", REPO, "rev-parse", "HEAD"],
        env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    path = "scripts/88_run_w7_resume_production.py"
    blob = subprocess.run(
        ["/usr/bin/git", "-C", REPO, "show", f"{head}:{path}"],
        env={"HOME": "/home/bmarti44", "PATH": "/usr/bin:/bin"},
        capture_output=True, check=True,
    ).stdout
    if Path(__file__).read_bytes() != blob:
        raise SystemExit("launcher differs from clean HEAD")
    fd = os.memfd_create("glm52-w7-production-launcher", os.MFD_ALLOW_SEALING)
    os.write(fd, blob)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
    os.set_inheritable(fd, True)
    environment = dict(BASE_ENV)
    environment.update(
        W7_SEALED_LAUNCHER_FD=str(fd),
        W7_EXECUTED_LAUNCHER_COMMIT=head,
        W7_EXECUTED_LAUNCHER_SHA256=hashlib.sha256(blob).hexdigest(),
    )
    os.execve(
        "/usr/bin/python3",
        ["/usr/bin/python3", "-I", "-B", f"/proc/self/fd/{fd}", *sys.argv[1:]],
        environment,
    )


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
    print("W7_PRODUCTION_EQUIVALENCE_SEALS_OK", json.dumps(results, sort_keys=True))


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate drand key: {key}")
        result[key] = value
    return result


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths or any(path.is_symlink() for path in paths):
        raise SystemExit("invalid drand runtime tree")
    for path in paths:
        relative, data = path.relative_to(root).as_posix().encode(), path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big")); digest.update(relative)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def _public_randomness() -> tuple[str, str]:
    if (
        hashlib.sha256(DRAND_NODE.read_bytes()).hexdigest() != DRAND_NODE_SHA256
        or _tree_sha256(DRAND_RUNTIME) != DRAND_RUNTIME_TREE_SHA256
        or hashlib.sha256(DRAND_VERIFIER.read_bytes()).hexdigest()
            != DRAND_VERIFIER_SHA256
    ):
        raise SystemExit("drand verifier dependency mismatch")
    records = []
    for host in ("api.drand.sh", "api2.drand.sh", "api3.drand.sh"):
        response = subprocess.run(
            [
                "/usr/bin/curl", "--disable", "--silent", "--show-error",
                "--fail", "--max-time", "15", "--proto", "=https",
                f"https://{host}/public/{DRAND_TARGET_ROUND}",
            ],
            env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
            capture_output=True,
            check=True,
        )
        record = json.loads(
            response.stdout, object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite drand value: {value}")
            ),
        )
        records.append(record)
    if not records[0] == records[1] == records[2]:
        raise SystemExit("drand latest relays disagree")
    record = records[0]
    if set(record) != {"round", "randomness", "signature", "previous_signature"}:
        raise SystemExit("unexpected drand record schema")
    round_number = record["round"]
    randomness = record["randomness"]
    signature = record["signature"]
    previous = record["previous_signature"]
    if (
        type(round_number) is not int or round_number != DRAND_TARGET_ROUND
        or not isinstance(randomness, str) or len(randomness) != 64
        or not isinstance(signature, str) or len(signature) != 192
        or not isinstance(previous, str) or len(previous) != 192
    ):
        raise SystemExit("invalid or stale drand record")
    try:
        signature_bytes = bytes.fromhex(signature)
        bytes.fromhex(previous)
        bytes.fromhex(randomness)
    except ValueError as error:
        raise SystemExit("non-hex drand record") from error
    if hashlib.sha256(bytes.fromhex(signature)).hexdigest() != randomness:
        raise SystemExit("drand randomness does not derive from signature")
    verification = subprocess.run(
        [str(DRAND_NODE), str(DRAND_VERIFIER), str(round_number), randomness,
         signature, previous],
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
        capture_output=True, check=False, timeout=30,
    )
    if verification.returncode != 0 or verification.stdout != b"DRAND_BLS_RECEIPT_OK\n":
        raise SystemExit("drand BLS verification failed")
    receipt = {
        "schema_version": 1,
        "source": "drand-default-preregistered-three-relay",
        "freeze_floor_round": DRAND_TARGET_ROUND - 1,
        **record,
        "relay_agreement": ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"],
        "launcher_commit": EXECUTED_LAUNCHER_COMMIT,
        "launcher_sha256": EXECUTED_LAUNCHER_SHA256,
        "frozen_gate_commit": CANDIDATE_COMMIT,
    }
    receipt_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    seed_material = (
        b"GLM52-W7-ARM-ORDER-V1\0" + CANDIDATE_COMMIT.encode() + b"\0"
        + EXECUTED_LAUNCHER_SHA256.encode() + b"\0"
        + str(round_number).encode() + b"\0" + randomness.encode()
    )
    return hashlib.sha256(seed_material).hexdigest(), receipt_json


def _environment(
    fds: dict[str, int], seed_sha256: str, receipt_json: str,
) -> dict[str, str]:
    environment = dict(BASE_ENV)
    environment.update(
        W7_EXECUTED_HARNESS_SHA256=FROZEN["harness"][1],
        W7_FROZEN_CANDIDATE_COMMIT=CANDIDATE_COMMIT,
        W7_RANDOM_SEED_SHA256=seed_sha256,
        W7_RANDOMNESS_RECEIPT_JSON=receipt_json,
        W7_SEALED_HARNESS_FD=str(fds["harness"]),
        W7_SEALED_SCORER_FD=str(fds["scorer"]),
        W7_SEALED_TRACE_SCORER_FD=str(fds["trace_scorer"]),
    )
    return environment


def main() -> None:
    _sealed_launcher_reexec()
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    os.environ.clear()
    os.environ.update(HOME="/home/bmarti44", PATH="/usr/bin:/bin")
    fds = _sealed_inputs()
    _mutation_test(fds)
    if args.self_test:
        environment = _environment(fds, "0" * 64, "{}")
        result = subprocess.run(
            [
                "/usr/bin/bash", "--noprofile", "--norc",
                f"/proc/self/fd/{fds['harness']}", "--validate-sealed-runtime",
            ],
            env=environment, pass_fds=tuple(fds.values()), text=True,
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.returncode)
        print(result.stdout, end="")
        return
    seed_sha256, receipt_json = _public_randomness()
    environment = _environment(fds, seed_sha256, receipt_json)
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
