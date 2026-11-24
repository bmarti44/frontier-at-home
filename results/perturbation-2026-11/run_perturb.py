"""Run the perturbation experiment's jobs against one served model, through the accuracy bench's code.

Usage:
  run_perturb.py generate --repo R --items items.jsonl --out DIR --base-url URL --api-key-file F
                          --encoder qwen38|inkling2 [--effort low] [--concurrency 3] --expect-model ID [--retry-invalid]
  run_perturb.py score-humaneval --repo R --out DIR [--min-available-gib 10]

Every job is rendered, sent and scored by scripts/31_bench_accuracy.py's own functions (render_item,
Client.complete, score_gsm8k, score_mmlu, run_humaneval), with the bench's budgets (MAX_TOKENS,
HUMANEVAL_STOPS), temperature 0 and seed. Only the rows differ: the original arm (O) sends the
benchmark rows, the perturbed arm (P) sends their variants. Jobs run in the items file's order,
which is already a seeded shuffle of both arms, so drift during the run hits both arms alike.

HumanEval completions are stored by `generate` and executed by `score-humaneval` one at a time,
each sandbox start waiting for MemAvailable >= the floor, so no container starts beside a busy
generation pass. A job whose transcript exists is skipped, so both phases resume.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import re
import socket
import threading
import urllib.error
import time
from datetime import datetime, timezone
from pathlib import Path

INVALID_PREFIXES = ("invalid:", "unparseable:", "sandbox timeout", "sandbox host timeout",
                    "sandbox launch failed", "no anchored answer", "truncated without answer")
# The holdout draws' request clock (review A-L1).
REQUEST_TIMEOUT_S = 2700
SANDBOX_INFRA = re.compile(r"^sandbox launch failed|^sandbox failed with exit (125|126|127)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bench(repo: Path):
    path = repo / "scripts/31_bench_accuracy.py"
    spec = importlib.util.spec_from_file_location("bench31", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def write_json(path: Path, document) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    raise RuntimeError("MemAvailable missing")


def generate(a) -> int:
    bench, bench_path = load_bench(a.repo)
    jobs = [json.loads(line) for line in a.items.read_text().splitlines()]
    assert len({j["job"] for j in jobs}) == len(jobs), "duplicate job ids"
    out = a.out
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    # The scoring phase reads the items from the run directory; a resumed run must use the same list.
    if (out / "items.jsonl").exists():
        assert (out / "items.jsonl").read_bytes() == a.items.read_bytes(), "items differ from the run's"
    else:
        (out / "items.jsonl").write_bytes(a.items.read_bytes())
    encoder = bench.load_encoder(a.encoder)
    client = bench.Client(a.base_url, bench.load_api_key(a.api_key_file), "/v1/completions", None,
                          REQUEST_TIMEOUT_S)
    model, models_response = client.get_model()
    if model != a.expect_model:
        raise SystemExit(f"served model {model!r} is not {a.expect_model!r}")
    run = {
        "started_at": utc_now(), "model": model, "models_response": models_response,
        "base_url": a.base_url, "encoder": a.encoder, "reasoning_effort": a.effort,
        "thinking_mode": "thinking", "concurrency": a.concurrency, "request_timeout_s": REQUEST_TIMEOUT_S,
        "max_tokens": bench.MAX_TOKENS, "humaneval_stops": bench.HUMANEVAL_STOPS,
        "temperature": 0, "seed": bench.SEED,
        "items_sha256": sha256_file(a.items), "bench_sha256": sha256_file(bench_path),
        "encoder_sha256": sha256_file(Path(bench.ENCODER_PATHS[a.encoder])),
        "runner_sha256": sha256_file(Path(__file__)), "jobs": len(jobs),
    }
    write_json(out / f"run-generate-{run['started_at'][:19].replace(':', '')}.json", run)
    lock = threading.Lock()
    done = {"n": 0}

    def one(job: dict) -> None:
        path = out / "transcripts" / f"{job['job'].replace(':', '_')}.json"
        attempt = 1
        if path.exists():
            previous = json.loads(path.read_text())
            # A transport error is retried once, on request. A timeout is final, as in the holdout
            # contract (review A-L1), and so is every scored outcome.
            if not (a.retry_invalid and previous["reason"].startswith("invalid:")
                    and not previous.get("timed_out") and previous.get("attempt", 1) < 2):
                return
            attempt = 2
        suite, item = job["suite"], job["item"]
        rendered, rendering = bench.render_item(suite, item, None if suite == "humaneval" else encoder,
                                                "thinking", a.encoder, a.effort)
        record = {"job": job["job"], "suite": suite, "row": job["row"], "split": job["split"],
                  "arm": job["arm"], "variant": job["variant"],
                  "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                  "rendering": rendering, "attempt": attempt, "started_at": utc_now()}
        try:
            completion, response, request, finish = client.complete(
                model, rendered, bench.MAX_TOKENS[suite],
                bench.HUMANEVAL_STOPS if suite == "humaneval" else None)
        except Exception as error:  # noqa: BLE001 - a transport failure, recorded like the bench records it
            timed_out = isinstance(error, (TimeoutError, socket.timeout)) or (
                isinstance(error, urllib.error.URLError) and isinstance(error.reason, (TimeoutError, socket.timeout)))
            record.update(completion="", timed_out=timed_out, scored_correct=False, expected=None,
                          reason=f"invalid: {type(error).__name__}: {str(error).replace(rendered, '<prompt>')}",
                          finished_at=utc_now())
        else:
            record.update(completion=completion, finish_reason=finish, request=request,
                          usage=(response or {}).get("usage"))
            # Scoring is outside the transport try (review A-M3): a scorer error is a bug and stops the run.
            if suite == "gsm8k":
                ok, expected, reason, fallback = bench.score_gsm8k(completion, item["answer"])
                record["used_fallback"] = fallback
            elif suite == "mmlu-pro":
                ok, expected, reason = bench.score_mmlu(completion, item["answer"], finish)
            else:
                ok, expected, reason = None, None, "pending sandbox"
            record.update(scored_correct=ok, expected=expected, reason=reason, finished_at=utc_now())
        write_json(path, record)
        with lock:
            done["n"] += 1
            print(f"[{done['n']}] {job['job']} {record['reason']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, j) for j in jobs]):
            future.result()
    print("generate finished", utc_now(), flush=True)
    return 0


def score_humaneval(a) -> int:
    bench, _ = load_bench(a.repo)
    out = a.out
    items = {j["job"]: j for j in (json.loads(line) for line in (out / "items.jsonl").read_text().splitlines())}
    paths = sorted((out / "transcripts").glob("humaneval_*.json"))
    with tempfile.TemporaryDirectory(prefix="perturb-humaneval-") as tmp:
        for n, path in enumerate(paths):
            record = json.loads(path.read_text())
            # Pending completions are executed; a sandbox infrastructure failure is re-executed once
            # (review A-M2); every other outcome is final.
            infra_retry = bool(SANDBOX_INFRA.search(record["reason"])) and record.get("sandbox_runs", 1) < 2
            if record["reason"] != "pending sandbox" and not infra_retry:
                continue
            waited = 0
            while mem_available_gib() < a.min_available_gib:
                if waited >= 600:
                    raise SystemExit(f"MemAvailable stayed below {a.min_available_gib} GiB for 600 s")
                time.sleep(5)
                waited += 5
            ok, expected, reason, execution = bench.run_humaneval(
                items[record["job"]]["item"], record["completion"], Path(tmp), f"case-{n:05d}")
            record.update(scored_correct=ok, expected=expected, reason=reason, execution=execution,
                          sandbox_runs=record.get("sandbox_runs", 0) + 1, scored_at=utc_now())
            write_json(path, record)
            print(f"[{n + 1}/{len(paths)}] {record['job']} {reason}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--repo", type=Path, required=True)
    g.add_argument("--items", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--base-url", required=True)
    g.add_argument("--api-key-file", type=Path, required=True)
    g.add_argument("--encoder", required=True, choices=("qwen38", "inkling2"))
    g.add_argument("--effort", default="low")
    g.add_argument("--concurrency", type=int, default=3)
    g.add_argument("--expect-model", required=True)
    g.add_argument("--retry-invalid", action="store_true")
    s = sub.add_parser("score-humaneval")
    s.add_argument("--repo", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--min-available-gib", type=float, default=10.0)
    a = ap.parse_args()
    return generate(a) if a.cmd == "generate" else score_humaneval(a)


if __name__ == "__main__":
    sys.exit(main())
