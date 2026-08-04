#!/usr/bin/env python3
"""Run the contained R0b production-path trace smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
FREEZE = ROOT / "results/glm52-gates/R0b-union-trace-smoke-freeze.json"
RANDOMNESS = ROOT / "results/glm52-gates/R0b-union-trace-smoke-randomness.json"
SHARED_PATH = ROOT / "scripts/73_run_glm_shared_router_probe.py"
SCORER_PATH = ROOT / "scripts/75_glm_union_trace_score.py"
FROZEN_RUNTIME_DEPENDENCIES = (
    "scripts/73_run_glm_shared_router_probe.py",
    "scripts/30_bench_speed.py",
    "scripts/glm52_goal.py",
    "scripts/03_memory_guard.py",
    "results/glm52-gates/harness/glm_safe_run.sh",
    "fixtures/ctx-32k.txt",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = _load("shared_router_runner", SHARED_PATH)
TRACE_SCORER = _load("union_trace_scorer", SCORER_PATH)
TRACE_LAYER = 4
TRACE_CONTEXT_CAP = 8192
TRACE_BYTES_PER_TOKEN_LAYER = (7168 + 256 + 256 + 8 + 256) * 4
MAX_TRACE_BYTES = TRACE_CONTEXT_CAP * TRACE_BYTES_PER_TOKEN_LAYER
TRACE_DISK_RESERVE_BYTES = 20 * 1024**3
TRACE_NAMES = ",".join((
    "glm_indexed_ffn_norm",
    "glm_indexed_router_logits",
    "glm_indexed_router_selected",
    "glm_indexed_router_bias",
    "glm_indexed_router_probs",
))
ENV_NAMES = sorted(set(SHARED.ENV_NAMES) | {
    "DS4_GLM_SYNC_TRACE",
    "DS4_METAL_GRAPH_DUMP_PREFIX",
    "DS4_METAL_GRAPH_DUMP_NAME",
    "DS4_METAL_GRAPH_DUMP_LAYER",
})
SYNC_RE = re.compile(
    r"ds4: GLM sync branch=full_indexed pos=(\d+) chunk=(\d+) logits=\d+"
)
DRAND_GENESIS_UNIX = 1595431050
DRAND_PERIOD_SECONDS = 30


def randomness_is_after_freeze(round_number: int, freeze_commit_time: int) -> bool:
    if round_number < 1 or freeze_commit_time < 0:
        return False
    round_time = DRAND_GENESIS_UNIX + (round_number - 1) * DRAND_PERIOD_SECONDS
    return round_time > freeze_commit_time


def validate_randomness_order() -> None:
    relative = str(FREEZE.relative_to(ROOT))
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", relative],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    freeze_time = int(completed.stdout.strip())
    randomness = SHARED.strict_json(RANDOMNESS)
    if not randomness_is_after_freeze(int(randomness["round"]), freeze_time):
        raise ValueError("public randomness does not postdate the freeze commit")


def configuration_sha256(values: dict[str, str]) -> str:
    canonical = b"".join(
        name.encode("ascii") + b"=" + values.get(name, "<UNSET>").encode() + b"\n"
        for name in ENV_NAMES
    )
    return hashlib.sha256(canonical).hexdigest()


def trace_environment(mode: str, out: Path) -> dict[str, str]:
    if mode not in {"off", "on"}:
        raise ValueError("invalid trace arm")
    values = dict(SHARED.COMMON_ENV)
    values.update({
        "DS4_LOCK_FILE": str(out / "runtime.lock"),
        "DS4_GLM_SYNC_TRACE": "1",
    })
    if mode == "on":
        values.update({
            "DS4_METAL_GRAPH_DUMP_PREFIX": str(out / "trace/request"),
            "DS4_METAL_GRAPH_DUMP_NAME": TRACE_NAMES,
            "DS4_METAL_GRAPH_DUMP_LAYER": "4",
        })
    return values


def matched_configuration_sha256() -> str:
    values = dict(SHARED.COMMON_ENV)
    values.update({"DS4_LOCK_FILE": "<ARM_LOCAL>", "DS4_GLM_SYNC_TRACE": "1"})
    return configuration_sha256(values)


def full_indexed_chunks(log: Path) -> list[list[int]]:
    rows = [[int(match.group(1)), int(match.group(2))]
            for match in SYNC_RE.finditer(log.read_text(encoding="utf-8", errors="strict"))]
    if not rows or len({tuple(row) for row in rows}) != len(rows):
        raise ValueError("full-indexed chunk evidence is missing or duplicated")
    for index, (pos, count) in enumerate(rows):
        if count <= 0 or (index and pos != rows[index - 1][0] + rows[index - 1][1]):
            raise ValueError("full-indexed chunks overlap or have a gap")
    return rows


def smoke_verdict(
    off: dict[str, Any],
    on: dict[str, Any],
    trace_score: dict[str, Any],
    off_containment: dict[str, Any],
    on_containment: dict[str, Any],
) -> dict[str, Any]:
    common_hashes = (
        "binary_sha256", "model_sha256", "tokenizer_sha256",
        "fixture_sha256", "configuration_sha256",
    )
    prompt_tokens = off.get("prompt_tokens")
    chunks = off.get("full_indexed_chunks")
    exact_coverage = (
        isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and
        prompt_tokens >= 4096 and on.get("prompt_tokens") == prompt_tokens and
        isinstance(chunks, list) and bool(chunks) and chunks[0][0] == 0 and
        sum(row[1] for row in chunks) == prompt_tokens and
        chunks[-1][0] + chunks[-1][1] == prompt_tokens
    )
    checks = {
        "arm_modes": off.get("mode") == "off" and on.get("mode") == "on",
        "frozen_identity": all(off.get(key) == on.get(key) for key in common_hashes),
        "byte_and_token_identity": off.get("response_signature") == on.get("response_signature"),
        "matched_indexed_chunks": off.get("full_indexed_chunks") == on.get("full_indexed_chunks"),
        "prompt_tokens_and_exact_coverage": exact_coverage,
        "off_emitted_no_trace": off.get("trace_files") == 0,
        "on_emitted_trace": isinstance(on.get("trace_files"), int) and on.get("trace_files", 0) > 0,
        "trace_score_passed": trace_score.get("verdict") == "PASS",
        "containment_clean": off_containment.get("clean") is True and on_containment.get("clean") is True,
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def _arm(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    binary = args.binary.resolve()
    expected = trace_environment(args.mode, out)
    observed = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
    if observed != expected:
        raise ValueError("trace arm environment differs from frozen configuration")
    if (SHARED.sha256(binary) != args.binary_sha256 or
            args.model_sha256 != SHARED.MODEL_SHA256 or
            SHARED.sha256(SHARED.TOKENIZER) != SHARED.TOKENIZER_SHA256):
        raise ValueError("candidate binary changed")
    if out.exists() or not str(out).startswith("/home/bmarti44/.local/state/glm52-"):
        raise ValueError("unsafe or existing output directory")
    out.mkdir(mode=0o700, parents=True)
    trace = out / "trace"
    trace.mkdir(mode=0o700)
    result_path = out / "result.json"
    server_log = out / "server.log"
    command = SHARED.server_command(binary, args.port)
    server = None
    with server_log.open("xb") as log:
        try:
            server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=False)
            SHARED.wait_ready(server, args.port)
            completed = subprocess.run([
                sys.executable, str(SHARED.BENCH),
                "--base-url", f"http://127.0.0.1:{args.port}",
                "--out", str(result_path), "--stack-label", f"union-trace-{args.mode}",
                "--model-id", "glm-5.2", "--tokenizer-path", str(SHARED.TOKENIZER),
                "--tokenizer-sha256", SHARED.TOKENIZER_SHA256,
                "--output-tokenizer-path", str(SHARED.TOKENIZER),
                "--output-tokenizer-sha256", SHARED.TOKENIZER_SHA256,
                "--token-timing-log", str(server_log), "--reps", "1", "--warmup", "0",
                "--context-levels", "4096", "--max-tokens", "128",
                "--min-completion-tokens", "128", "--request-timeout", "2700",
                "--seed", str(args.seed),
            ], stdin=subprocess.DEVNULL, capture_output=True, timeout=3000, check=False)
            (out / "bench.stdout.log").write_bytes(completed.stdout)
            (out / "bench.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"benchmark failed rc={completed.returncode}")
            signature = SHARED.response_signature(result_path)
            payload = SHARED.strict_json(result_path)
            reps = payload["cells"][0]["reps"]
            prompt_tokens = reps[0].get("prompt_tokens") if len(reps) == 1 else None
            if (not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or
                    prompt_tokens < 4096):
                raise ValueError("benchmark prompt-token coverage is insufficient")
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
            log.flush()
            os.fsync(log.fileno())
    if server is None or server.returncode != 0:
        raise RuntimeError(f"server did not exit cleanly rc={getattr(server, 'returncode', None)}")
    chunks = full_indexed_chunks(server_log)
    files = [path for path in trace.iterdir()]
    total_trace_bytes = sum(path.stat().st_size for path in files if path.is_file())
    if total_trace_bytes > MAX_TRACE_BYTES:
        raise RuntimeError("trace exceeded its context-derived byte ceiling")
    if args.mode == "off" and files:
        raise RuntimeError("off arm emitted trace files")
    if args.mode == "on" and not files:
        raise RuntimeError("on arm emitted no trace files")
    record = {
        "mode": args.mode,
        "binary_sha256": args.binary_sha256,
        "model_sha256": args.model_sha256,
        "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
        "fixture_sha256": signature["request_sha256"],
        "configuration_sha256": matched_configuration_sha256(),
        "environment_sha256": configuration_sha256(expected),
        "response_signature": signature,
        "prompt_tokens": prompt_tokens,
        "full_indexed_chunks": chunks,
        "trace_files": len(files),
        "trace_bytes": total_trace_bytes,
        "result_sha256": SHARED.sha256(result_path),
        "server_log_sha256": SHARED.sha256(server_log),
    }
    (out / "arm.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", args.tag) is None:
        raise ValueError("invalid tag")
    freeze = SHARED.frozen_inputs(FREEZE, RANDOMNESS)
    validate_randomness_order()
    candidate = Path(str(freeze["candidate_directory"])).resolve()
    binary = candidate / "ds4-server"
    SHARED.no_other_inference()
    available_gib = int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:"))) / 1048576
    if available_gib < 110:
        raise RuntimeError(f"only {available_gib:.3f} GiB available")
    root = Path(f"/home/bmarti44/.local/state/glm52-{args.tag}")
    if root.exists():
        raise FileExistsError(root)
    usage = shutil.disk_usage(root.parent)
    if usage.free < MAX_TRACE_BYTES + TRACE_DISK_RESERVE_BYTES:
        raise RuntimeError("insufficient trace disk space plus preservation reserve")
    root.mkdir(mode=0o700, parents=True)

    containment: dict[str, dict[str, Any]] = {}
    for index, mode in enumerate(("off", "on")):
        out = root / mode
        values = trace_environment(mode, out)
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        environment.update({
            "GLM_CANDIDATE_SRC": str(candidate),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": str(freeze["binary_sha256"]),
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(ENV_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": configuration_sha256(values),
            "GLM_SAFE_MEMORY_HIGH_GIB": "69",
            "GLM_SAFE_KILL_FLOOR_GIB": "18",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "3600",
            "GLM_SAFE_FINAL_ARTIFACTS": ",".join((
                str(out / "arm.json"), str(out / "result.json"), str(out / "server.log"),
            )),
        })
        completed = subprocess.run([
            str(CGROUP), "--tag", f"{args.tag}-{mode}", "--", sys.executable,
            str(Path(__file__).resolve()), "_arm", "--mode", mode,
            "--out", str(out), "--binary", str(binary),
            "--binary-sha256", str(freeze["binary_sha256"]),
            "--model-sha256", str(freeze["model_sha256"]),
            "--seed", str(freeze["seed"]), "--port", str(args.port + index),
        ], env=environment, stdin=subprocess.DEVNULL, capture_output=True,
           timeout=3700, check=False)
        (root / f"{mode}.containment.stdout.log").write_bytes(completed.stdout)
        (root / f"{mode}.containment.stderr.log").write_bytes(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"contained {mode} arm failed rc={completed.returncode}")
        record = SHARED.containment_record(root / f"{mode}.containment.stdout.log")
        containment[mode] = {"clean": True, **record}
        (root / f"{mode}.containment.json").write_text(
            json.dumps(containment[mode], sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        SHARED.no_other_inference()

    off = SHARED.strict_json(root / "off/arm.json")
    on = SHARED.strict_json(root / "on/arm.json")
    expected_chunks = [tuple(row) for row in off["full_indexed_chunks"]]
    trace_score = TRACE_SCORER.score_trace(
        root / "on/trace", root / "on/server.log", max_bytes=MAX_TRACE_BYTES,
        expected_layers={TRACE_LAYER}, expected_chunks=expected_chunks,
    )
    verdict = smoke_verdict(off, on, trace_score, containment["off"], containment["on"])
    SHARED.frozen_inputs(FREEZE, RANDOMNESS)
    summary = {
        "schema_version": 1,
        "candidate_hash": freeze["candidate_hash"],
        "engine_commit": freeze["engine_commit"],
        "binary_sha256": freeze["binary_sha256"],
        "model_sha256": freeze["model_sha256"],
        "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
        "seed": freeze["seed"],
        "max_trace_bytes": MAX_TRACE_BYTES,
        "off_arm_sha256": SHARED.sha256(root / "off/arm.json"),
        "on_arm_sha256": SHARED.sha256(root / "on/arm.json"),
        "off_containment_sha256": SHARED.sha256(root / "off.containment.json"),
        "on_containment_sha256": SHARED.sha256(root / "on.containment.json"),
        "trace_score": trace_score,
        **verdict,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0 if summary["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    public = sub.add_parser("run")
    public.add_argument("--tag", required=True)
    public.add_argument("--port", type=int, default=18090)
    public.set_defaults(func=run)
    internal = sub.add_parser("_arm")
    internal.add_argument("--mode", choices=("off", "on"), required=True)
    internal.add_argument("--out", type=Path, required=True)
    internal.add_argument("--binary", type=Path, required=True)
    internal.add_argument("--binary-sha256", required=True)
    internal.add_argument("--model-sha256", required=True)
    internal.add_argument("--seed", type=int, required=True)
    internal.add_argument("--port", type=int, required=True)
    internal.set_defaults(func=_arm)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
