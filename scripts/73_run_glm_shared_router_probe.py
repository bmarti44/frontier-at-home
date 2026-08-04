#!/usr/bin/env python3
"""Run the matched R0a recall falsifier through existing GLM containment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
from glm52_goal import paired_ratio_bound, validate_ab_blocks


ROOT = Path(__file__).resolve().parents[1]
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SCORER = ROOT / "scripts/72_glm_shared_router_score.py"
BENCH = ROOT / "scripts/30_bench_speed.py"
FREEZE_MANIFEST = ROOT / "results/glm52-gates/R0a-shared-router-freeze.json"
RANDOMNESS_RECEIPT = ROOT / "results/glm52-gates/R0a-shared-router-randomness.json"
PERF_FREEZE_MANIFEST = ROOT / "results/glm52-gates/R0a-shared-router-perf-freeze.json"
PERF_RANDOMNESS_RECEIPT = ROOT / "results/glm52-gates/R0a-shared-router-perf-randomness.json"
CAMPAIGN_FREEZE_MANIFEST = ROOT / "results/glm52-gates/R0a-shared-router-campaign-freeze.json"
CAMPAIGN_RANDOMNESS_RECEIPT = ROOT / "results/glm52-gates/R0a-shared-router-campaign-randomness.json"
DRAND_ENDPOINT = "https://api.drand.sh/public/{round}"
MODEL = Path("/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf")
MODEL_BYTES = 211075856448
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
TOKENIZER = Path("/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json")
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
ENV_NAMES = sorted((
    "DS4_CUDA_EXPERT_CACHE_GB",
    "DS4_CUDA_EXPERT_CACHE_PIN",
    "DS4_CUDA_EXPERT_CACHE_SLRU",
    "DS4_CUDA_FETCH_THREADS",
    "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
    "DS4_GLM_PREDACC_SHARED",
    "DS4_GLM_PREFETCH",
    "DS4_GLM_PREFETCH_SHARED_CORRECTION",
    "DS4_GLM_PREFETCH_THREADS",
    "DS4_LOCK_FILE",
    "DS4_TOKEN_TIMING_LOG",
))
COMMON_ENV = {
    "DS4_CUDA_EXPERT_CACHE_GB": "68",
    "DS4_CUDA_EXPERT_CACHE_PIN": "1",
    "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
    "DS4_CUDA_FETCH_THREADS": "8",
    "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
    "DS4_TOKEN_TIMING_LOG": "1",
}
# ds4-server defaults to one slot unless --batched-sessions is supplied. The
# runner issues exactly one request and never supplies that option.
SINGLE_REQUEST_SLOT = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def frozen_inputs(
    freeze_path: Path = FREEZE_MANIFEST,
    randomness_path: Path = RANDOMNESS_RECEIPT,
) -> dict[str, object]:
    freeze = strict_json(freeze_path)
    randomness = strict_json(randomness_path)
    required = {
        "schema_version", "repository_parent_commit", "engine_commit",
        "candidate_directory", "binary_sha256", "binary_mtime_ns",
        "model_sha256", "model_stat", "artifacts",
    }
    if set(freeze) != required or freeze["schema_version"] != 1:
        raise ValueError("freeze manifest schema is invalid")
    if not isinstance(freeze["artifacts"], dict):
        raise ValueError("freeze artifact map is invalid")
    for relative, digest in freeze["artifacts"].items():
        path = (ROOT / str(relative)).resolve()
        if (not str(path).startswith(str(ROOT) + "/") or
                not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or
                sha256(path) != digest):
            raise ValueError(f"frozen artifact changed: {relative}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()
    clean = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                           capture_output=True, check=True).stdout
    ancestor = subprocess.run([
        "git", "merge-base", "--is-ancestor", str(freeze["repository_parent_commit"]), head
    ], cwd=ROOT, check=False)
    if clean or ancestor.returncode != 0:
        raise ValueError("repository is dirty or not descended from frozen harness")
    candidate = Path(str(freeze["candidate_directory"])).resolve()
    binary = candidate / "ds4-server"
    if (not str(candidate).startswith("/home/bmarti44/.cache/glm52-") or
            sha256(binary) != freeze["binary_sha256"] or
            binary.stat().st_mtime_ns != freeze["binary_mtime_ns"]):
        raise ValueError("frozen candidate binary changed")
    expected_stat = freeze["model_stat"]
    current = MODEL.stat()
    observed_stat = [current.st_dev, current.st_ino, current.st_size,
                     current.st_mtime_ns, current.st_ctime_ns]
    if (freeze["model_sha256"] != MODEL_SHA256 or observed_stat != expected_stat):
        raise ValueError("previously full-hashed model identity changed")
    if set(randomness) != {"schema_version", "round", "randomness", "seed"}:
        raise ValueError("randomness receipt schema is invalid")
    raw = randomness["randomness"]
    if (randomness["schema_version"] != 1 or
            not isinstance(randomness["round"], int) or
            not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{64}", raw) is None or
            randomness["seed"] != int(raw[:16], 16) % 2147483647):
        raise ValueError("public-randomness seed derivation is invalid")
    with urllib.request.urlopen(
        DRAND_ENDPOINT.format(round=randomness["round"]), timeout=15
    ) as response:
        public = json.loads(response.read().decode("utf-8"))
    if (public.get("round") != randomness["round"] or
            public.get("randomness") != raw):
        raise ValueError("committed randomness differs from the public drand round")
    return {**freeze, "candidate_hash": head, "seed": randomness["seed"]}


def environment_for(mode: str, lock_file: Path) -> dict[str, str]:
    result = dict(COMMON_ENV)
    result["DS4_LOCK_FILE"] = str(lock_file)
    if mode == "on":
        result["DS4_GLM_PREDACC_SHARED"] = "1"
    return result


def performance_environment_for(mode: str, lock_file: Path) -> dict[str, str]:
    if mode not in {"off", "corrected"}:
        raise ValueError("invalid performance arm")
    result = dict(COMMON_ENV)
    result["DS4_LOCK_FILE"] = str(lock_file)
    if mode == "corrected":
        result.update({
            "DS4_GLM_PREFETCH": "1",
            "DS4_GLM_PREFETCH_SHARED_CORRECTION": "1",
            "DS4_GLM_PREFETCH_THREADS": "8",
        })
    return result


def performance_verdict(
    off_decode: float, corrected_decode: float, output_identity: bool
) -> dict[str, object]:
    valid = all(math.isfinite(value) and value > 0.0
                for value in (off_decode, corrected_decode))
    ratio = corrected_decode / off_decode if valid else 0.0
    checks = {
        "finite_positive_decode": valid,
        "byte_and_token_identity": output_identity,
        "decode_non_regression": valid and ratio >= 0.95,
    }
    return {
        "formula": "corrected_decode_tokens_per_second / off_decode_tokens_per_second",
        "minimum_decode_ratio": 0.95,
        "decode_ratio": ratio,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def campaign_schedule(flip: bool) -> tuple[tuple[int, int, str], ...]:
    if not isinstance(flip, bool):
        raise ValueError("campaign orientation must be boolean")
    rows: list[tuple[int, int, str]] = []
    for block in range(5):
        order = "ABBA" if (block + int(flip)) % 2 == 0 else "BAAB"
        rows.extend(
            (block, sequence, "off" if arm == "A" else "corrected")
            for sequence, arm in enumerate(order)
        )
    return tuple(rows)


def campaign_verdict(rows: list[dict[str, object]], flip: bool) -> dict[str, object]:
    expected_keys = {
        "block", "sequence", "mode", "decode_tokens_per_second",
        "ttft_seconds", "completion_tokens", "response_signature",
        "fixture_sha256", "server_boot_id", "binary_sha256",
        "configuration_sha256",
    }
    if len(rows) != 20 or any(not isinstance(row, dict) or set(row) != expected_keys
                              for row in rows):
        raise ValueError("campaign requires 20 exact-schema rows")
    expected = campaign_schedule(flip)
    observed = tuple((row["block"], row["sequence"], row["mode"]) for row in rows)
    if observed != expected:
        raise ValueError("campaign execution order differs from the frozen schedule")
    validation_rows = []
    signatures: set[str] = set()
    for row in rows:
        mode = row["mode"]
        if mode not in {"off", "corrected"}:
            raise ValueError("campaign row has invalid mode")
        decode = row["decode_tokens_per_second"]
        ttft = row["ttft_seconds"]
        if (isinstance(decode, bool) or not isinstance(decode, (int, float)) or
                not math.isfinite(float(decode)) or float(decode) <= 0 or
                isinstance(ttft, bool) or not isinstance(ttft, (int, float)) or
                not math.isfinite(float(ttft)) or float(ttft) <= 0 or
                row["completion_tokens"] != 128):
            raise ValueError("campaign timing or completion coverage is invalid")
        signature = json.dumps(
            row["response_signature"], sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        signatures.add(signature)
        validation_rows.append({
            "block": row["block"],
            "sequence": row["sequence"],
            "arm": "A" if mode == "off" else "B",
            "fixture_sha256": row["fixture_sha256"],
            "server_boot_id": row["server_boot_id"],
            "binary_sha256": row["binary_sha256"],
            "configuration_sha256": row["configuration_sha256"],
        })
    if len(signatures) != 1:
        raise ValueError("campaign response signatures differ")
    validate_ab_blocks(validation_rows, flip=flip)
    off_decode: list[float] = []
    corrected_decode: list[float] = []
    off_ttft: list[float] = []
    corrected_ttft: list[float] = []
    for block in range(5):
        for mode, decode_target, ttft_target in (
            ("off", off_decode, off_ttft),
            ("corrected", corrected_decode, corrected_ttft),
        ):
            group = [row for row in rows if row["block"] == block and row["mode"] == mode]
            if len(group) != 2:
                raise ValueError("campaign block lacks two observations per arm")
            decode_target.append(statistics.fmean(
                float(row["decode_tokens_per_second"]) for row in group
            ))
            ttft_target.append(statistics.fmean(float(row["ttft_seconds"]) for row in group))
    decode_lower = paired_ratio_bound(corrected_decode, off_decode, side="lower")
    ttft_upper = paired_ratio_bound(corrected_ttft, off_ttft, side="upper")
    checks = {
        "decode_improvement_lower_95": decode_lower > 1.0,
        "ttft_non_regression_upper_95": ttft_upper <= 1.05,
        "byte_and_token_identity": len(signatures) == 1,
    }
    return {
        "formula": "one-sided 95% paired geometric ratio bounds over five block means",
        "acceptance": {
            "decode_ratio_lower_95_strictly_greater_than": 1.0,
            "ttft_ratio_upper_95_maximum": 1.05,
        },
        "decode_ratio_lower_95": decode_lower,
        "ttft_ratio_upper_95": ttft_upper,
        "block_decode_tokens_per_second": {
            "off": off_decode, "corrected": corrected_decode,
        },
        "block_ttft_seconds": {"off": off_ttft, "corrected": corrected_ttft},
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def server_command(binary: Path, port: int) -> list[str]:
    return [
        str(binary), "--cuda", "-m", str(MODEL), "-c", "8192",
        "--host", "127.0.0.1", "--port", str(port),
        "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
    ]


def environment_sha256(values: dict[str, str]) -> str:
    canonical = b"".join(
        name.encode("ascii") + b"=" + values.get(name, "<UNSET>").encode() + b"\n"
        for name in ENV_NAMES
    )
    return hashlib.sha256(canonical).hexdigest()


def no_other_inference() -> None:
    found = subprocess.run(
        ["pgrep", "-x", "ds4-server"], capture_output=True, text=True, check=False
    )
    if found.returncode == 0 and found.stdout.strip():
        raise RuntimeError(f"another ds4-server is active: {found.stdout.strip()}")
    if found.returncode not in (0, 1):
        raise RuntimeError("could not inspect ds4-server processes")


def wait_ready(server: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={server.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("server readiness timeout")


def response_signature(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("suite_valid") is not True or len(payload.get("cells", [])) != 1:
        raise ValueError("benchmark result is incomplete")
    reps = payload["cells"][0].get("reps", [])
    if len(reps) != 1 or reps[0].get("valid") is not True:
        raise ValueError("benchmark repetition is incomplete")
    rep = reps[0]
    if rep.get("completion_tokens", 0) < 128:
        raise ValueError("completion is too short")
    keys = (
        "request_sha256", "token_ids", "completion_tokens",
        "generated_reasoning_sha256", "generated_reasoning_bytes",
        "generated_content_sha256", "generated_content_bytes",
    )
    signature = {key: rep.get(key) for key in keys}
    if not isinstance(signature["token_ids"], list) or not signature["token_ids"]:
        raise ValueError("generated_token_ids are absent")
    return signature


def containment_record(stdout_path: Path) -> dict[str, object]:
    text = stdout_path.read_text(encoding="utf-8", errors="strict")
    matches = re.findall(r"SAFE_RUN_DONE rc=0 killed=no dir=([^\s]+)", text)
    if len(matches) != 1:
        raise ValueError("contained arm lacks one clean SAFE_RUN receipt")
    crash = Path(matches[0]).resolve()
    main = crash / "main.log"
    samples = crash / "samples.log"
    kernel = crash / "kernel.log"
    if not all(path.is_file() and not path.is_symlink() for path in (main, samples, kernel)):
        raise ValueError("contained safety artifacts are missing")
    main_text = main.read_text(encoding="utf-8", errors="strict")
    if "executed candidate was verified alive at least once" not in main_text:
        raise ValueError("candidate identity was not continuously sampled")
    # memory.events.local is captured by glm_safe_run in cgroup_final; the
    # wrapper exits nonzero for any high/max/oom/oom_kill delta.
    if "cgroup_final " not in main_text or "SAFE_RUN end rc=0 killed=no" not in main_text:
        raise ValueError("cgroup or clean-exit evidence is absent")
    if re.search(r"NVRM.*Xid", kernel.read_text(encoding="utf-8", errors="replace"), re.I):
        raise ValueError("kernel Xid appeared during arm")
    return {
        "crash_directory": str(crash), "main_sha256": sha256(main),
        "samples_sha256": sha256(samples), "kernel_sha256": sha256(kernel),
    }


def arm(args: argparse.Namespace) -> int:
    if ((args.purpose == "recall" and args.mode not in {"off", "on"}) or
            (args.purpose == "performance" and args.mode not in {"off", "corrected"})):
        raise ValueError("arm purpose/mode mismatch")
    binary = args.binary.resolve()
    out = args.out.resolve()
    expected = (
        environment_for(args.mode, out / "runtime.lock")
        if args.purpose == "recall"
        else performance_environment_for(args.mode, out / "runtime.lock")
    )
    observed = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
    if observed != expected:
        raise ValueError("arm environment is not the fixed configuration")
    if (sha256(binary) != args.binary_sha256 or args.model_sha256 != MODEL_SHA256
            or MODEL.stat().st_size != MODEL_BYTES):
        raise ValueError("binary or model identity changed")
    if out.exists() or not str(out).startswith("/home/bmarti44/.local/state/glm52-"):
        raise ValueError("unsafe or existing arm output")
    out.mkdir(mode=0o700, parents=True)
    result_path = out / "result.json"
    server_log_path = out / "server.log"
    arm_path = out / "arm.json"
    command = server_command(binary, args.port)
    server: subprocess.Popen[bytes] | None = None
    with server_log_path.open("xb") as log:
        try:
            server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=False)
            wait_ready(server, args.port)
            completed = subprocess.run([
                sys.executable, str(BENCH), "--base-url", f"http://127.0.0.1:{args.port}",
                "--out", str(result_path), "--stack-label",
                f"shared-router-{args.purpose}-{args.mode}",
                "--model-id", "glm-5.2", "--output-tokenizer-path", str(TOKENIZER),
                "--output-tokenizer-sha256", TOKENIZER_SHA256, "--token-timing-log",
                str(server_log_path), "--reps", "1", "--warmup", "0",
                "--request-timeout", "2700", "--context-levels", "0",
                "--max-tokens", "128", "--min-completion-tokens", "128",
                "--seed", str(args.seed),
            ], stdin=subprocess.DEVNULL, capture_output=True, timeout=3000, check=False)
            (out / "bench.stdout.log").write_bytes(completed.stdout)
            (out / "bench.stderr.log").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"benchmark failed rc={completed.returncode}")
            signature = response_signature(result_path)
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
    log_text = server_log_path.read_text(encoding="utf-8", errors="strict")
    pair_count = sum(line.startswith("PREDPAIR ") for line in log_text.splitlines())
    if args.purpose == "recall":
        if args.mode == "off" and pair_count != 0:
            raise ValueError("off arm emitted probe rows")
        if args.mode == "on" and pair_count < 1036:
            raise ValueError("on arm emitted too few probe rows")
    else:
        if pair_count != 0:
            raise ValueError("performance arm emitted diagnostic probe rows")
        correction_marker = "ds4: shared-expert router correction enabled"
        if (args.mode == "corrected") != (correction_marker in log_text):
            raise ValueError("performance correction marker mismatch")
    record = {
        "schema_version": 1, "purpose": args.purpose, "mode": args.mode,
        "engine_commit": args.engine_commit,
        "binary_sha256": args.binary_sha256, "model_sha256": args.model_sha256,
        "tokenizer_sha256": TOKENIZER_SHA256, "environment_sha256": environment_sha256(expected),
        "seed": args.seed, "pair_rows": pair_count, "single_request_slots": SINGLE_REQUEST_SLOT,
        "result_sha256": sha256(result_path), "server_log_sha256": sha256(server_log_path),
        "response_signature": signature,
    }
    arm_path.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    return 0


def run(args: argparse.Namespace) -> int:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", args.tag) is None:
        raise ValueError("invalid tag")
    freeze = frozen_inputs()
    candidate = Path(str(freeze["candidate_directory"])).resolve()
    binary = candidate / "ds4-server"
    binary_sha256 = str(freeze["binary_sha256"])
    model_sha256 = str(freeze["model_sha256"])
    engine_commit = str(freeze["engine_commit"])
    seed = int(freeze["seed"])
    if sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity mismatch")
    no_other_inference()
    available_gib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))) / 1048576
    if available_gib < 110:
        raise RuntimeError(f"only {available_gib:.3f} GiB available")
    root = Path(f"/home/bmarti44/.local/state/glm52-{args.tag}")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(mode=0o700, parents=True)
    for index, mode in enumerate(("off", "on")):
        out = root / mode
        values = environment_for(mode, out / "runtime.lock")
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        environment.update({
            "GLM_CANDIDATE_SRC": str(candidate),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1", "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": binary_sha256,
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(ENV_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": environment_sha256(values),
            "GLM_SAFE_MEMORY_HIGH_GIB": "69", "GLM_SAFE_KILL_FLOOR_GIB": "18",
            "GLM_SAFE_MIN_START_GIB": "110", "GLM_SAFE_TIMEOUT_S": "3600",
            "GLM_SAFE_FINAL_ARTIFACTS": ",".join((str(out / "arm.json"),
                                                       str(out / "result.json"),
                                                       str(out / "server.log"))),
        })
        completed = subprocess.run([
            str(CGROUP), "--tag", f"{args.tag}-{mode}", "--", sys.executable,
            str(Path(__file__).resolve()), "_arm", "--purpose", "recall",
            "--mode", mode, "--out", str(out),
            "--binary", str(binary), "--binary-sha256", binary_sha256,
            "--model-sha256", model_sha256, "--engine-commit", engine_commit,
            "--seed", str(seed), "--port", str(args.port + index),
        ], env=environment, stdin=subprocess.DEVNULL, capture_output=True, timeout=3700, check=False)
        (root / f"{mode}.containment.stdout.log").write_bytes(completed.stdout)
        (root / f"{mode}.containment.stderr.log").write_bytes(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"contained {mode} arm failed rc={completed.returncode}")
        containment = containment_record(root / f"{mode}.containment.stdout.log")
        (root / f"{mode}.containment.json").write_text(
            json.dumps(containment, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        no_other_inference()
    off = json.loads((root / "off/arm.json").read_text())
    on = json.loads((root / "on/arm.json").read_text())
    identity = off["response_signature"] == on["response_signature"]
    score_path = root / "score.json"
    scored = subprocess.run([sys.executable, str(SCORER), str(root / "on/server.log"),
                             "--out", str(score_path)], check=False)
    score = json.loads(score_path.read_text())
    # Recheck the stat-bound, previously full-hashed model after both arms.
    frozen_inputs()
    summary = {
        "schema_version": 1, "candidate_hash": freeze["candidate_hash"],
        "engine_commit": engine_commit, "binary_sha256": binary_sha256,
        "model_sha256": model_sha256, "seed": seed,
        "off_arm_sha256": sha256(root / "off/arm.json"),
        "on_arm_sha256": sha256(root / "on/arm.json"),
        "off_containment_sha256": sha256(root / "off.containment.json"),
        "on_containment_sha256": sha256(root / "on.containment.json"),
        "score_sha256": sha256(score_path), "byte_and_token_identity": identity,
        "score": score,
        "verdict": "PASS" if identity and scored.returncode == 0 else "FAIL",
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0 if summary["verdict"] == "PASS" else 1


def _decode_rate(path: Path) -> float:
    payload = strict_json(path)
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("performance result has an invalid cell count")
    reps = cells[0].get("reps") if isinstance(cells[0], dict) else None
    if not isinstance(reps, list) or len(reps) != 1 or reps[0].get("valid") is not True:
        raise ValueError("performance result has no valid repetition")
    value = reps[0].get("decode_tok_s")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("performance result has invalid decode throughput")
    return float(value)


def performance_run(args: argparse.Namespace) -> int:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", args.tag) is None:
        raise ValueError("invalid tag")
    freeze = frozen_inputs(PERF_FREEZE_MANIFEST, PERF_RANDOMNESS_RECEIPT)
    candidate = Path(str(freeze["candidate_directory"])).resolve()
    binary = candidate / "ds4-server"
    binary_sha256 = str(freeze["binary_sha256"])
    model_sha256 = str(freeze["model_sha256"])
    engine_commit = str(freeze["engine_commit"])
    seed = int(freeze["seed"])
    if sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity mismatch")
    no_other_inference()
    available_gib = int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )) / 1048576
    if available_gib < 110:
        raise RuntimeError(f"only {available_gib:.3f} GiB available")
    root = Path(f"/home/bmarti44/.local/state/glm52-{args.tag}")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(mode=0o700, parents=True)
    order = ("off", "corrected") if seed % 2 == 0 else ("corrected", "off")
    for index, mode in enumerate(order):
        out = root / mode
        values = performance_environment_for(mode, out / "runtime.lock")
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        environment.update({
            "GLM_CANDIDATE_SRC": str(candidate),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": binary_sha256,
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(ENV_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": environment_sha256(values),
            "GLM_SAFE_MEMORY_HIGH_GIB": "69",
            "GLM_SAFE_KILL_FLOOR_GIB": "18",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "3600",
            "GLM_SAFE_FINAL_ARTIFACTS": ",".join((
                str(out / "arm.json"), str(out / "result.json"),
                str(out / "server.log"),
            )),
        })
        completed = subprocess.run([
            str(CGROUP), "--tag", f"{args.tag}-{mode}", "--", sys.executable,
            str(Path(__file__).resolve()), "_arm", "--purpose", "performance",
            "--mode", mode, "--out", str(out), "--binary", str(binary),
            "--binary-sha256", binary_sha256, "--model-sha256", model_sha256,
            "--engine-commit", engine_commit, "--seed", str(seed),
            "--port", str(args.port + index),
        ], env=environment, stdin=subprocess.DEVNULL, capture_output=True,
           timeout=3700, check=False)
        (root / f"{mode}.containment.stdout.log").write_bytes(completed.stdout)
        (root / f"{mode}.containment.stderr.log").write_bytes(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"contained {mode} performance arm failed rc={completed.returncode}")
        containment = containment_record(root / f"{mode}.containment.stdout.log")
        (root / f"{mode}.containment.json").write_text(
            json.dumps(containment, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        no_other_inference()
    off = strict_json(root / "off/arm.json")
    corrected = strict_json(root / "corrected/arm.json")
    identity = off["response_signature"] == corrected["response_signature"]
    off_decode = _decode_rate(root / "off/result.json")
    corrected_decode = _decode_rate(root / "corrected/result.json")
    score = performance_verdict(off_decode, corrected_decode, identity)
    frozen_inputs(PERF_FREEZE_MANIFEST, PERF_RANDOMNESS_RECEIPT)
    summary = {
        "schema_version": 1,
        "candidate_hash": freeze["candidate_hash"],
        "engine_commit": engine_commit,
        "binary_sha256": binary_sha256,
        "model_sha256": model_sha256,
        "seed": seed,
        "arm_order": list(order),
        "off_decode_tokens_per_second": off_decode,
        "corrected_decode_tokens_per_second": corrected_decode,
        "off_arm_sha256": sha256(root / "off/arm.json"),
        "corrected_arm_sha256": sha256(root / "corrected/arm.json"),
        "off_containment_sha256": sha256(root / "off.containment.json"),
        "corrected_containment_sha256": sha256(root / "corrected.containment.json"),
        "score": score,
        "verdict": score["verdict"],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0 if summary["verdict"] == "PASS" else 1


def _result_metrics(path: Path) -> tuple[float, float, int]:
    payload = strict_json(path)
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("campaign result has an invalid cell count")
    reps = cells[0].get("reps") if isinstance(cells[0], dict) else None
    if not isinstance(reps, list) or len(reps) != 1 or reps[0].get("valid") is not True:
        raise ValueError("campaign result has no valid repetition")
    decode = reps[0].get("decode_tok_s")
    ttft = reps[0].get("ttft_s")
    completion = reps[0].get("completion_tokens")
    if (isinstance(decode, bool) or not isinstance(decode, (int, float)) or
            not math.isfinite(float(decode)) or float(decode) <= 0 or
            isinstance(ttft, bool) or not isinstance(ttft, (int, float)) or
            not math.isfinite(float(ttft)) or float(ttft) <= 0 or
            completion != 128):
        raise ValueError("campaign result has invalid timing or completion")
    return float(decode), float(ttft), int(completion)


def campaign_run(args: argparse.Namespace) -> int:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,19}", args.tag) is None:
        raise ValueError("invalid campaign tag")
    freeze = frozen_inputs(CAMPAIGN_FREEZE_MANIFEST, CAMPAIGN_RANDOMNESS_RECEIPT)
    candidate = Path(str(freeze["candidate_directory"])).resolve()
    binary = candidate / "ds4-server"
    binary_sha256 = str(freeze["binary_sha256"])
    model_sha256 = str(freeze["model_sha256"])
    engine_commit = str(freeze["engine_commit"])
    seed = int(freeze["seed"])
    flip = bool(seed & 1)
    if sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity mismatch")
    no_other_inference()
    available_gib = int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )) / 1048576
    if available_gib < 110:
        raise RuntimeError(f"only {available_gib:.3f} GiB available")
    root = Path(f"/home/bmarti44/.local/state/glm52-{args.tag}")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(mode=0o700, parents=True)
    rows: list[dict[str, object]] = []
    for block, sequence, mode in campaign_schedule(flip):
        label = f"b{block}s{sequence}-{mode}"
        out = root / label
        values = performance_environment_for(mode, out / "runtime.lock")
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        environment.update({
            "GLM_CANDIDATE_SRC": str(candidate),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": binary_sha256,
            "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST": ",".join(ENV_NAMES),
            "GLM_SAFE_EXPECTED_ENV_SHA256": environment_sha256(values),
            "GLM_SAFE_MEMORY_HIGH_GIB": "69",
            "GLM_SAFE_KILL_FLOOR_GIB": "18",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "3600",
            "GLM_SAFE_FINAL_ARTIFACTS": ",".join((
                str(out / "arm.json"), str(out / "result.json"),
                str(out / "server.log"),
            )),
        })
        completed = subprocess.run([
            str(CGROUP), "--tag", f"{args.tag}-b{block}s{sequence}-{mode}",
            "--", sys.executable, str(Path(__file__).resolve()), "_arm",
            "--purpose", "performance", "--mode", mode, "--out", str(out),
            "--binary", str(binary), "--binary-sha256", binary_sha256,
            "--model-sha256", model_sha256, "--engine-commit", engine_commit,
            "--seed", str(seed), "--port", str(args.port),
        ], env=environment, stdin=subprocess.DEVNULL, capture_output=True,
           timeout=3700, check=False)
        stdout_path = root / f"{label}.containment.stdout.log"
        stderr_path = root / f"{label}.containment.stderr.log"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"contained campaign arm {label} failed rc={completed.returncode}")
        containment = containment_record(stdout_path)
        containment_path = root / f"{label}.containment.json"
        containment_path.write_text(
            json.dumps(containment, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        arm_record = strict_json(out / "arm.json")
        decode, ttft, completion = _result_metrics(out / "result.json")
        row = {
            "block": block,
            "sequence": sequence,
            "mode": mode,
            "decode_tokens_per_second": decode,
            "ttft_seconds": ttft,
            "completion_tokens": completion,
            "response_signature": arm_record["response_signature"],
            "fixture_sha256": arm_record["response_signature"]["request_sha256"],
            "server_boot_id": containment["crash_directory"],
            "binary_sha256": binary_sha256,
            "configuration_sha256": arm_record["environment_sha256"],
        }
        row_path = root / f"{label}.json"
        row_path.write_text(
            json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append(row)
        no_other_inference()
    score = campaign_verdict(rows, flip)
    frozen_inputs(CAMPAIGN_FREEZE_MANIFEST, CAMPAIGN_RANDOMNESS_RECEIPT)
    summary = {
        "schema_version": 1,
        "candidate_hash": freeze["candidate_hash"],
        "engine_commit": engine_commit,
        "binary_sha256": binary_sha256,
        "model_sha256": model_sha256,
        "seed": seed,
        "schedule_flip": flip,
        "row_sha256": {
            f"b{row['block']}s{row['sequence']}-{row['mode']}.json": sha256(
                root / f"b{row['block']}s{row['sequence']}-{row['mode']}.json"
            ) for row in rows
        },
        "score": score,
        "verdict": score["verdict"],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0 if summary["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    public = sub.add_parser("run")
    public.add_argument("--tag", required=True)
    public.add_argument("--port", type=int, default=8040)
    public.set_defaults(func=run)
    perf = sub.add_parser("perf")
    perf.add_argument("--tag", required=True)
    perf.add_argument("--port", type=int, default=8040)
    perf.set_defaults(func=performance_run)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--tag", required=True)
    campaign.add_argument("--port", type=int, default=8040)
    campaign.set_defaults(func=campaign_run)
    internal = sub.add_parser("_arm")
    internal.add_argument("--purpose", choices=("recall", "performance"), required=True)
    internal.add_argument("--mode", choices=("off", "on", "corrected"), required=True)
    internal.add_argument("--out", type=Path, required=True)
    internal.add_argument("--binary", type=Path, required=True)
    internal.add_argument("--binary-sha256", required=True)
    internal.add_argument("--model-sha256", required=True)
    internal.add_argument("--engine-commit", required=True)
    internal.add_argument("--seed", type=int, required=True)
    internal.add_argument("--port", type=int, required=True)
    internal.set_defaults(func=arm)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.func(parsed))
