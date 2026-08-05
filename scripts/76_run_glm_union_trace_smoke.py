#!/usr/bin/env python3
"""Run a contained R0b union-trace OFF/ON qualification.

The default mode preserves the qualified short single-batch check.  The explicit
high-row mode additionally requires exact contiguous coverage across multiple
indexed-prefill chunks, including a full 2,048-row chunk.
"""

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
CORPUS_FREEZE = ROOT / "results/glm52-gates/R0b-union-corpus-runtime-freeze.json"
CORPUS_RANDOMNESS = ROOT / "results/glm52-gates/R0b-union-corpus-runtime-randomness.json"
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
MIN_PROMPT_TOKENS = 512
MAX_CONTEXT_LEVEL = 8192
TRACE_BYTES_PER_TOKEN_LAYER = (6144 + 256 + 256 + 8 + 256) * 4
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
    "DS4_GLM_UNION_TRACE_CORPUS",
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


def validate_randomness_order(
    freeze_path: Path = FREEZE, randomness_path: Path = RANDOMNESS,
) -> None:
    relative = str(freeze_path.relative_to(ROOT))
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", relative],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    freeze_time = int(completed.stdout.strip())
    randomness = SHARED.strict_json(randomness_path)
    if not randomness_is_after_freeze(int(randomness["round"]), freeze_time):
        raise ValueError("public randomness does not postdate the freeze commit")


def configuration_sha256(values: dict[str, str]) -> str:
    canonical = b"".join(
        name.encode("ascii") + b"=" + values.get(name, "<UNSET>").encode() + b"\n"
        for name in ENV_NAMES
    )
    return hashlib.sha256(canonical).hexdigest()


def trace_environment(mode: str, out: Path, *, corpus_smoke: bool = False) -> dict[str, str]:
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
            "DS4_METAL_GRAPH_DUMP_LAYER": "all" if corpus_smoke else "4",
        })
        if corpus_smoke:
            values["DS4_GLM_UNION_TRACE_CORPUS"] = "1"
    return values


def matched_configuration_sha256() -> str:
    values = dict(SHARED.COMMON_ENV)
    values.update({"DS4_LOCK_FILE": "<ARM_LOCAL>", "DS4_GLM_SYNC_TRACE": "1"})
    return configuration_sha256(values)


def full_indexed_chunks(log: Path) -> list[list[int]]:
    return full_indexed_chunks_text(log.read_text(encoding="utf-8", errors="strict"))


def full_indexed_chunks_text(text: str) -> list[list[int]]:
    rows = [[int(match.group(1)), int(match.group(2))]
            for match in SYNC_RE.finditer(text)]
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
    *,
    min_prompt_tokens: int = MIN_PROMPT_TOKENS,
    require_multichunk: bool = False,
    expected_corpus_seed: int | None = None,
) -> dict[str, Any]:
    common_hashes = (
        "binary_sha256", "model_sha256", "tokenizer_sha256",
        "fixture_sha256", "configuration_sha256",
    )
    prompt_tokens = off.get("prompt_tokens")
    chunks = off.get("full_indexed_chunks")
    chunks_well_formed = (
        isinstance(chunks, list) and bool(chunks) and
        all(
            isinstance(row, list) and len(row) == 2 and
            isinstance(row[0], int) and not isinstance(row[0], bool) and
            isinstance(row[1], int) and not isinstance(row[1], bool) and
            row[0] >= 0 and row[1] > 0
            for row in chunks
        )
    )
    chunks_contiguous = chunks_well_formed and all(
        chunks[index][0] == chunks[index - 1][0] + chunks[index - 1][1]
        for index in range(1, len(chunks))
    )
    exact_coverage = (
        isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and
        prompt_tokens >= min_prompt_tokens and on.get("prompt_tokens") == prompt_tokens and
        chunks_well_formed and chunks_contiguous and chunks[0][0] == 0 and
        sum(row[1] for row in chunks) == prompt_tokens and
        chunks[-1][0] + chunks[-1][1] == prompt_tokens and
        (not require_multichunk or (
            prompt_tokens > 2048 and len(chunks) >= 2 and
            any(row[1] == 2048 for row in chunks)
        ))
    )
    off_corpus = off.get("corpus_requests")
    on_corpus = on.get("corpus_requests")
    corpus_mode = off_corpus is not None or on_corpus is not None
    corpus_scope = True
    corpus_event_floor = True
    if corpus_mode:
        def valid_requests(value: Any) -> bool:
            if (not isinstance(value, list) or len(value) != 2 or
                    not isinstance(expected_corpus_seed, int) or
                    isinstance(expected_corpus_seed, bool)):
                return False
            for expected_id, item in enumerate(value, 1):
                if (not isinstance(item, dict) or item.get("request_id") != expected_id or
                        item.get("seed") != expected_corpus_seed + expected_id - 1 or
                        not isinstance(item.get("prompt_tokens"), int) or
                        isinstance(item.get("prompt_tokens"), bool) or
                        item["prompt_tokens"] < MIN_PROMPT_TOKENS):
                    return False
                item_chunks = item.get("full_indexed_chunks")
                if (not isinstance(item_chunks, list) or not item_chunks or
                        any(not isinstance(row, list) or len(row) != 2 or
                            any(not isinstance(number, int) or isinstance(number, bool)
                                for number in row) or row[1] <= 0
                            for row in item_chunks)):
                    return False
                if (item_chunks[0][0] != 0 or
                        any(item_chunks[index][0] !=
                            item_chunks[index - 1][0] + item_chunks[index - 1][1]
                            for index in range(1, len(item_chunks))) or
                        sum(row[1] for row in item_chunks) != item["prompt_tokens"]):
                    return False
                signature = item.get("response_signature")
                request_sha256 = (signature.get("request_sha256")
                                  if isinstance(signature, dict) else None)
                if (not isinstance(request_sha256, str) or
                        re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None):
                    return False
            return True

        off_hashes = ({item["response_signature"]["request_sha256"] for item in off_corpus}
                      if valid_requests(off_corpus) else set())
        on_hashes = ({item["response_signature"]["request_sha256"] for item in on_corpus}
                     if valid_requests(on_corpus) else set())
        corpus_scope = (
            valid_requests(off_corpus) and valid_requests(on_corpus) and
            len(off_hashes) == 2 and off_hashes == on_hashes and
            all(
                all(left.get(key) == right.get(key) for key in (
                    "request_id", "seed", "prompt_tokens", "full_indexed_chunks",
                    "response_signature",
                ))
                for left, right in zip(off_corpus, on_corpus)
            ) and trace_score.get("requests") == 2
        )
        corpus_event_floor = (
            isinstance(trace_score.get("token_layer_events"), int) and
            trace_score["token_layer_events"] >= 76800
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
        "corpus_request_scope": corpus_scope,
        "corpus_event_floor": corpus_event_floor,
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def _arm(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    binary = args.binary.resolve()
    expected = trace_environment(args.mode, out, corpus_smoke=args.corpus_smoke)
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
            request_records: list[dict[str, Any]] = []
            for request_index in range(2 if args.corpus_smoke else 1):
                current_result = (out / f"result-{request_index + 1}.json"
                                  if args.corpus_smoke else result_path)
                log.flush()
                os.fsync(log.fileno())
                log_start = server_log.stat().st_size
                completed = subprocess.run([
                    sys.executable, str(SHARED.BENCH),
                    "--base-url", f"http://127.0.0.1:{args.port}",
                    "--out", str(current_result),
                    "--stack-label", f"union-trace-{args.mode}-r{request_index + 1}",
                    "--model-id", "glm-5.2", "--tokenizer-path", str(SHARED.TOKENIZER),
                    "--tokenizer-sha256", SHARED.TOKENIZER_SHA256,
                    "--output-tokenizer-path", str(SHARED.TOKENIZER),
                    "--output-tokenizer-sha256", SHARED.TOKENIZER_SHA256,
                    "--token-timing-log", str(server_log), "--reps", "1", "--warmup", "0",
                    "--context-levels", str(args.context_level), "--max-tokens", "128",
                    "--min-completion-tokens", "128", "--request-timeout", "1200",
                    "--seed", str(args.seed + request_index),
                ], stdin=subprocess.DEVNULL, capture_output=True, timeout=1350, check=False)
                (out / f"bench-{request_index + 1}.stdout.log").write_bytes(completed.stdout)
                (out / f"bench-{request_index + 1}.stderr.log").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"benchmark request {request_index + 1} failed rc={completed.returncode}"
                    )
                log.flush()
                os.fsync(log.fileno())
                with server_log.open("rb") as log_reader:
                    log_reader.seek(log_start)
                    request_log = log_reader.read().decode("utf-8", errors="strict")
                signature = SHARED.response_signature(current_result)
                payload = SHARED.strict_json(current_result)
                reps = payload["cells"][0]["reps"]
                prompt_tokens = reps[0].get("prompt_tokens") if len(reps) == 1 else None
                if (not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or
                        prompt_tokens < max(MIN_PROMPT_TOKENS, args.context_level)):
                    raise ValueError("benchmark prompt-token coverage is insufficient")
                request_records.append({
                    "request_id": request_index + 1,
                    "seed": args.seed + request_index,
                    "prompt_tokens": prompt_tokens,
                    "full_indexed_chunks": full_indexed_chunks_text(request_log),
                    "response_signature": signature,
                    "result_sha256": SHARED.sha256(current_result),
                })
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
    if not request_records:
        raise RuntimeError("arm produced no requests")
    chunks = request_records[0]["full_indexed_chunks"]
    files = [path for path in trace.iterdir()]
    total_trace_bytes = sum(path.stat().st_size for path in files if path.is_file())
    if total_trace_bytes > args.max_trace_bytes:
        raise RuntimeError("trace exceeded its context-derived byte ceiling")
    if args.mode == "off" and files:
        raise RuntimeError("off arm emitted trace files")
    if args.mode == "on" and not files:
        raise RuntimeError("on arm emitted no trace files")
    fixture_digest = (hashlib.sha256(
        b"".join(bytes.fromhex(str(item["response_signature"]["request_sha256"]))
                 for item in request_records)
    ).hexdigest() if args.corpus_smoke else
        str(request_records[0]["response_signature"]["request_sha256"]))
    result_digest = (hashlib.sha256(
        b"".join(bytes.fromhex(str(item["result_sha256"])) for item in request_records)
    ).hexdigest() if args.corpus_smoke else str(request_records[0]["result_sha256"]))
    record = {
        "mode": args.mode,
        "binary_sha256": args.binary_sha256,
        "model_sha256": args.model_sha256,
        "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
        "fixture_sha256": fixture_digest,
        "configuration_sha256": matched_configuration_sha256(),
        "environment_sha256": configuration_sha256(expected),
        "response_signature": ([item["response_signature"] for item in request_records]
                               if args.corpus_smoke else request_records[0]["response_signature"]),
        "prompt_tokens": request_records[0]["prompt_tokens"],
        "full_indexed_chunks": chunks,
        "trace_files": len(files),
        "trace_bytes": total_trace_bytes,
        "result_sha256": result_digest,
        "server_log_sha256": SHARED.sha256(server_log),
    }
    if args.corpus_smoke:
        record["corpus_requests"] = request_records
    (out / "arm.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", args.tag) is None:
        raise ValueError("invalid tag")
    if not MIN_PROMPT_TOKENS <= args.context_level <= MAX_CONTEXT_LEVEL:
        raise ValueError("context level is outside the bounded trace range")
    if args.require_multichunk and args.context_level <= 2048:
        raise ValueError("multi-chunk qualification requires context level above 2048")
    if args.require_multichunk and args.corpus_smoke:
        raise ValueError("corpus smoke and single-request multichunk modes are exclusive")
    layer_count = 75 if args.corpus_smoke else 1
    request_count = 2 if args.corpus_smoke else 1
    max_trace_bytes = (
        (args.context_level + 1024) * TRACE_BYTES_PER_TOKEN_LAYER *
        layer_count * request_count
    )
    freeze_path = CORPUS_FREEZE if args.corpus_smoke else FREEZE
    randomness_path = CORPUS_RANDOMNESS if args.corpus_smoke else RANDOMNESS
    freeze = SHARED.frozen_inputs(freeze_path, randomness_path)
    validate_randomness_order(freeze_path, randomness_path)
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
    if usage.free < max_trace_bytes + TRACE_DISK_RESERVE_BYTES:
        raise RuntimeError("insufficient trace disk space plus preservation reserve")
    root.mkdir(mode=0o700, parents=True)

    containment: dict[str, dict[str, Any]] = {}
    for index, mode in enumerate(("off", "on")):
        out = root / mode
        values = trace_environment(mode, out, corpus_smoke=args.corpus_smoke)
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        final_artifacts = [str(out / "arm.json")]
        if args.corpus_smoke:
            final_artifacts.extend(str(out / f"result-{request_id}.json")
                                   for request_id in (1, 2))
        else:
            final_artifacts.append(str(out / "result.json"))
        final_artifacts.append(str(out / "server.log"))
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
            "GLM_SAFE_FINAL_ARTIFACTS": ",".join(final_artifacts),
        })
        completed = subprocess.run([
            str(CGROUP), "--tag", f"{args.tag}-{mode}", "--", sys.executable,
            str(Path(__file__).resolve()), "_arm", "--mode", mode,
            "--out", str(out), "--binary", str(binary),
            "--binary-sha256", str(freeze["binary_sha256"]),
            "--model-sha256", str(freeze["model_sha256"]),
            "--seed", str(freeze["seed"]), "--port", str(args.port + index),
            "--context-level", str(args.context_level),
            "--max-trace-bytes", str(max_trace_bytes),
            *(["--require-multichunk"] if args.require_multichunk else []),
            *(["--corpus-smoke"] if args.corpus_smoke else []),
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
    if args.corpus_smoke:
        expected_requests = {
            int(item["request_id"]): [tuple(row) for row in item["full_indexed_chunks"]]
            for item in on["corpus_requests"]
        }
        trace_score = TRACE_SCORER.score_trace(
            root / "on/trace", root / "on/server.log", max_bytes=max_trace_bytes,
            expected_layers=set(range(3, 78)), expected_chunks=[],
            expected_requests=expected_requests,
        )
    else:
        expected_chunks = [tuple(row) for row in off["full_indexed_chunks"]]
        trace_score = TRACE_SCORER.score_trace(
            root / "on/trace", root / "on/server.log", max_bytes=max_trace_bytes,
            expected_layers={TRACE_LAYER}, expected_chunks=expected_chunks,
        )
    verdict = smoke_verdict(
        off, on, trace_score, containment["off"], containment["on"],
        min_prompt_tokens=(2049 if args.require_multichunk else MIN_PROMPT_TOKENS),
        require_multichunk=args.require_multichunk,
        expected_corpus_seed=(int(freeze["seed"]) if args.corpus_smoke else None),
    )
    SHARED.frozen_inputs(freeze_path, randomness_path)
    if args.corpus_smoke:
        qualification = {
            "scope": "multi_request_all_routed_layer_corpus_smoke",
            "high_row_2048_status": "OPEN",
            "minimum_token_layer_events": 76800,
        }
    elif args.require_multichunk:
        qualification = {
            "scope": "high_row_multichunk",
            "high_row_2048_status": "PASS" if verdict["verdict"] == "PASS" else "FAIL",
        }
    else:
        qualification = {
            "scope": "short_single_indexed_batch_only",
            "high_row_2048_status": "OPEN",
        }
    summary = {
        "schema_version": 1,
        **qualification,
        "candidate_hash": freeze["candidate_hash"],
        "engine_commit": freeze["engine_commit"],
        "binary_sha256": freeze["binary_sha256"],
        "model_sha256": freeze["model_sha256"],
        "tokenizer_sha256": SHARED.TOKENIZER_SHA256,
        "seed": freeze["seed"],
        "context_level": args.context_level,
        "max_trace_bytes": max_trace_bytes,
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
    public.add_argument("--context-level", type=int, default=512)
    public.add_argument("--require-multichunk", action="store_true")
    public.add_argument("--corpus-smoke", action="store_true")
    public.set_defaults(func=run)
    internal = sub.add_parser("_arm")
    internal.add_argument("--mode", choices=("off", "on"), required=True)
    internal.add_argument("--out", type=Path, required=True)
    internal.add_argument("--binary", type=Path, required=True)
    internal.add_argument("--binary-sha256", required=True)
    internal.add_argument("--model-sha256", required=True)
    internal.add_argument("--seed", type=int, required=True)
    internal.add_argument("--port", type=int, required=True)
    internal.add_argument("--context-level", type=int, required=True)
    internal.add_argument("--max-trace-bytes", type=int, required=True)
    internal.add_argument("--require-multichunk", action="store_true")
    internal.add_argument("--corpus-smoke", action="store_true")
    internal.set_defaults(func=_arm)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
