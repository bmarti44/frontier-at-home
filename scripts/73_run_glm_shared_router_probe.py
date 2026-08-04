#!/usr/bin/env python3
"""Run the matched R0a recall falsifier through existing GLM containment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SCORER = ROOT / "scripts/72_glm_shared_router_score.py"
BENCH = ROOT / "scripts/30_bench_speed.py"
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


def environment_for(mode: str) -> dict[str, str]:
    result = dict(COMMON_ENV)
    if mode == "on":
        result["DS4_GLM_PREDACC_SHARED"] = "1"
    return result


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
    if rep.get("completion_tokens", 0) < 64:
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
    binary = args.binary.resolve()
    out = args.out.resolve()
    expected = environment_for(args.mode)
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
    command = [
        str(binary), "--cuda", "-m", str(MODEL), "-c", "8192",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
    ]
    server: subprocess.Popen[bytes] | None = None
    with server_log_path.open("xb") as log:
        try:
            server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=False)
            wait_ready(server, args.port)
            completed = subprocess.run([
                sys.executable, str(BENCH), "--base-url", f"http://127.0.0.1:{args.port}",
                "--out", str(result_path), "--stack-label", f"shared-router-{args.mode}",
                "--model-id", "glm-5.2", "--output-tokenizer-path", str(TOKENIZER),
                "--output-tokenizer-sha256", TOKENIZER_SHA256, "--token-timing-log",
                str(server_log_path), "--reps", "1", "--warmup", "0",
                "--request-timeout", "2700", "--context-levels", "0",
                "--max-tokens", "64", "--min-completion-tokens", "64",
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
    if args.mode == "off" and pair_count != 0:
        raise ValueError("off arm emitted probe rows")
    if args.mode == "on" and pair_count < 1036:
        raise ValueError("on arm emitted too few probe rows")
    record = {
        "schema_version": 1, "mode": args.mode, "engine_commit": args.engine_commit,
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
    binary = args.candidate.resolve() / "ds4-server"
    if not str(binary).startswith("/home/bmarti44/.cache/glm52-"):
        raise ValueError("candidate is outside the frozen cache")
    if (sha256(binary) != args.binary_sha256 or args.model_sha256 != MODEL_SHA256
            or MODEL.stat().st_size != MODEL_BYTES):
        raise ValueError("frozen artifact identity mismatch")
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
        values = environment_for(mode)
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("DS4_") or name.startswith("GLM_SAFE_"):
                environment.pop(name)
        environment.update(values)
        environment.update({
            "GLM_CANDIDATE_SRC": str(args.candidate.resolve()),
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1", "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": args.binary_sha256,
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
            str(Path(__file__).resolve()), "_arm", "--mode", mode, "--out", str(out),
            "--binary", str(binary), "--binary-sha256", args.binary_sha256,
            "--model-sha256", args.model_sha256, "--engine-commit", args.engine_commit,
            "--seed", str(args.seed), "--port", str(args.port + index),
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
    summary = {
        "schema_version": 1, "candidate_hash": args.candidate_hash,
        "engine_commit": args.engine_commit, "binary_sha256": args.binary_sha256,
        "model_sha256": args.model_sha256, "seed": args.seed,
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    public = sub.add_parser("run")
    public.add_argument("--tag", required=True)
    public.add_argument("--candidate", type=Path, required=True)
    public.add_argument("--candidate-hash", required=True)
    public.add_argument("--engine-commit", required=True)
    public.add_argument("--binary-sha256", required=True)
    public.add_argument("--model-sha256", required=True)
    public.add_argument("--seed", type=int, required=True)
    public.add_argument("--port", type=int, default=8040)
    public.set_defaults(func=run)
    internal = sub.add_parser("_arm")
    internal.add_argument("--mode", choices=("off", "on"), required=True)
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
