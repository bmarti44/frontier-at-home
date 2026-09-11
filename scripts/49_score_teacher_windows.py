#!/usr/bin/env python3
"""Score a running OpenAI-compatible vLLM server against public BF16 teacher logits.

Per window: verify the token .npy and logits safetensors SHA-256 against
dataset-manifest.json, reduce the teacher (FP64 logsumexp minus target logit,
first argmax), then POST the frozen prompt_logprobs=1 request and reduce the
candidate response. Output follows the repo evidence contract (manifest.json,
raw.jsonl, summary.json, responses/<window_id>.json).

The engine's `--logprobs-mode` is NOT verifiable from the client: the server
response field name cannot establish whether values are raw logprobs or
processed ones. The caller's profile MUST launch the server with
`--logprobs-mode raw_logprobs`; the manifest records this as an assumption.

Exit 0 on PASS, 1 on FAIL, 2 on usage/dataset errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm52_goal import _weighted_upper_95  # noqa: E402

CHUNK = 8 * 1024 * 1024
FORMULAS = {
    "teacher_nll": "NLL = log(sum(exp(full-vocabulary FP32 logits))) - target logit, FP64 stable accumulation; top1 is first argmax; row i predicts token i+1",
    "candidate_nll": "NLL = -prompt_logprobs[position][true token].logprob; top1 is the rank==1 entry",
    "delta_nll": "per window (candidate_nll_sum - teacher_nll_sum) / tokens; token-weighted mean and one-sided 95% upper bound (glm52_goal._weighted_upper_95, effective-n t quantile)",
    "top1_loss_pp": "per window 100 * (teacher_top1_correct - candidate_top1_correct) / tokens; token-weighted mean and one-sided 95% upper bound",
    "top1_agreement": "token-weighted mean of candidate_top1_agrees_teacher / tokens",
}


class DatasetError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_ids(value, vocab_size):
    return isinstance(value, list) and all(type(token) is int and 0 <= token < vocab_size for token in value)


def reduce_prompt_response(response, token_ids, model_name, vocab_size):
    """Validate and reduce a native vLLM prompt_logprobs completion response.

    Returns tokens, nll_sum, top1_correct, and the per-position candidate top-1 ids.
    """
    if type(vocab_size) is not int or vocab_size <= 1:
        raise ValueError("invalid frozen vocabulary size")
    if not _token_ids(token_ids, vocab_size) or len(token_ids) < 2:
        raise ValueError("invalid frozen prompt token IDs")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("invalid frozen model name")
    if not isinstance(response, dict) or response.get("model") != model_name:
        raise ValueError("native response model mismatch")
    choices, usage = response.get("choices"), response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected one native completion choice")
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] != expected
            for key, expected in (("prompt_tokens", len(token_ids)), ("completion_tokens", 1), ("total_tokens", len(token_ids) + 1))):
        raise ValueError("native token usage mismatch")
    choice = choices[0]
    if type(choice.get("index")) is not int or choice["index"] != 0 or choice.get("finish_reason") not in ("length", "stop"):
        raise ValueError("native completion is missing or unfinished")
    generated = choice.get("token_ids")
    if not _token_ids(generated, vocab_size) or len(generated) != 1:
        raise ValueError("native completion token is missing")
    prompt = choice.get("prompt_token_ids")
    if not _token_ids(prompt, vocab_size) or prompt != token_ids:
        raise ValueError("native prompt token alignment mismatch")
    rows = choice.get("prompt_logprobs")
    if not isinstance(rows, list) or len(rows) != len(token_ids) or rows[0] is not None:
        raise ValueError("native scored-position coverage mismatch")
    nll, correct, top1_ids = [], 0, []
    for target, row in zip(token_ids[1:], rows[1:], strict=True):
        if not isinstance(row, dict) or len(row) not in (1, 2) or str(target) not in row:
            raise ValueError("native truth/top1 logprobs are missing")
        top = []
        for key, entry in row.items():
            if not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key) or int(key) >= vocab_size:
                raise ValueError("native logprob token ID is invalid")
            if not isinstance(entry, dict):
                raise ValueError("native logprob entry is invalid")
            logprob, rank = entry.get("logprob"), entry.get("rank")
            if type(logprob) not in (int, float) or not math.isfinite(logprob) or logprob > 0:
                raise ValueError("native logprob is nonfinite or invalid")
            if type(rank) is not int or not 1 <= rank <= vocab_size:
                raise ValueError("native token rank is invalid")
            if rank == 1:
                top.append(key)
        if len(top) != 1 or set(row) != {str(target), top[0]}:
            raise ValueError("native top1 is missing or ambiguous")
        truth, best = row[str(target)], row[top[0]]
        if str(target) != top[0] and truth["logprob"] > best["logprob"]:
            raise ValueError("native logprob order disagrees with rank")
        nll.append(-truth["logprob"])
        correct += top[0] == str(target)
        top1_ids.append(int(top[0]))
    total = math.fsum(nll)
    if not math.isfinite(total):
        raise ValueError("native NLL aggregate is nonfinite")
    return {"tokens": len(nll), "nll_sum": total, "top1_correct": correct, "top1_ids": top1_ids}


def load_token_ids(path: Path, expected_sha: str, window_len: int) -> list[int]:
    if not path.is_file():
        raise DatasetError(f"missing token file {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise DatasetError(f"token sha256 mismatch for {path}: {actual} != {expected_sha}")
    ids = np.load(path)
    if ids.shape != (window_len,) or not np.issubdtype(ids.dtype, np.integer):
        raise DatasetError(f"token array {path} has shape {ids.shape} dtype {ids.dtype}, expected ({window_len},) int")
    return [int(v) for v in ids]


def reduce_teacher(path: Path, expected_sha: str, ids: list[int], vocab: int) -> dict:
    """Stream one safetensors logits file; return per-position NLL and argmax."""
    if not path.is_file():
        raise DatasetError(f"missing logits file {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise DatasetError(f"logits sha256 mismatch for {path}: {actual} != {expected_sha}")
    positions = len(ids) - 1
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < 1048576:
            raise DatasetError(f"bad safetensors header length in {path}")
        header = json.loads(f.read(n))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if len(tensors) != 1:
            raise DatasetError(f"expected one tensor in {path}")
        name, tensor = next(iter(tensors.items()))
        if tensor["dtype"] != "F32" or tensor["shape"] != [positions, vocab]:
            raise DatasetError(f"tensor {name} in {path} is {tensor['dtype']} {tensor['shape']}, expected F32 [{positions}, {vocab}]")
        if tensor["data_offsets"] != [0, positions * vocab * 4] or path.stat().st_size != 8 + n + positions * vocab * 4:
            raise DatasetError(f"safetensors payload size mismatch in {path}")
        nlls, top1 = [], []
        for target in ids[1:]:
            data = f.read(vocab * 4)
            if len(data) != vocab * 4:
                raise DatasetError(f"short read in {path}")
            values = np.frombuffer(data, dtype="<f4").astype(np.float64)
            if not np.isfinite(values).all():
                raise DatasetError(f"nonfinite teacher logits in {path}")
            top = int(values.argmax())
            peak = float(values[top])
            lse = peak + math.log(float(np.exp(values - peak).sum(dtype=np.float64)))
            nll = lse - float(values[target])
            if not math.isfinite(nll) or nll < 0:
                raise DatasetError(f"invalid teacher NLL in {path}")
            nlls.append(nll)
            top1.append(top)
        if f.read(1):
            raise DatasetError(f"trailing bytes in {path}")
    return {"logits_sha256": actual, "tokens": positions, "nll": nlls, "top1_ids": top1,
            "nll_sum": math.fsum(nlls), "top1_correct": sum(t == g for t, g in zip(top1, ids[1:])),
            "source_tensor": name}


def teacher_cached(cache_dir: Path, window_id: str, logits_sha: str, compute) -> dict:
    cache_path = cache_dir / f"{window_id}.{logits_sha}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("logits_sha256") == logits_sha and cached.get("window_id") == window_id:
            cached["cache_hit"] = True
            return cached
    result = compute()
    result["window_id"] = window_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False))
    os.replace(tmp, cache_path)
    result["cache_hit"] = False
    return result


def http_json(url: str, api_key: str | None, payload: dict | None, timeout: float) -> tuple[dict, bytes]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        body = reply.read()
    return json.loads(body), body


def parse_windows(spec: str | None, count: int) -> list[int]:
    if not spec:
        return list(range(count))
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if re.fullmatch(r"\d+-\d+", part):
            lo, hi = (int(x) for x in part.split("-"))
            chosen.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", part):
            chosen.add(int(part))
        else:
            raise DatasetError(f"bad --windows spec {spec!r}")
    bad = [i for i in chosen if not 0 <= i < count]
    if bad:
        raise DatasetError(f"window index out of range: {bad}")
    return sorted(chosen)


def host_facts() -> dict:
    mem = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"uname": " ".join(platform.uname()), "nproc": os.cpu_count(), "mem_total_bytes": mem}


def score_windows(args: argparse.Namespace) -> int:
    started = time.time()
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "dataset-manifest.json"
    if not manifest_path.is_file():
        raise DatasetError(f"missing {manifest_path}")
    dataset_manifest = json.loads(manifest_path.read_text())
    vocab = int(dataset_manifest["vocab_size"])
    files = dataset_manifest["logit_files"]
    indices = parse_windows(args.windows, len(files))
    cache_dir = Path(args.teacher_cache) if args.teacher_cache else dataset_dir / "teacher-cache"
    api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else None
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    template = {"model": args.model, "prompt": "<token ids>", "add_special_tokens": False, "prompt_logprobs": 1,
                "return_token_ids": True, "n": 1, "max_tokens": 1, "stream": False, "temperature": 0}

    # Dataset verification happens before any server contact so dataset errors exit 2 cleanly.
    prepared = []
    for index in indices:
        entry = files[index]
        window_id = entry["window_id"]
        window_len = int(entry["prediction_positions"]) + 1
        ids = load_token_ids(dataset_dir / "calibration" / "panel-v1" / "arrays" / f"{window_id}.tokens.npy",
                             entry["token_ids_sha256"], window_len)
        logits_path = dataset_dir / entry["path"]
        teacher = teacher_cached(cache_dir, window_id, entry["sha256"],
                                 lambda: reduce_teacher(logits_path, entry["sha256"], ids, vocab))
        if teacher["tokens"] != window_len - 1:
            raise DatasetError(f"teacher cache for {window_id} has wrong token count")
        print(f"teacher {window_id}: nll_sum={teacher['nll_sum']:.6f} top1={teacher['top1_correct']} cache_hit={teacher['cache_hit']}", flush=True)
        prepared.append((entry, ids, teacher))

    try:
        models_response, _ = http_json(f"{args.base_url}/v1/models", api_key, None, args.request_timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        models_response = {"error": f"{type(exc).__name__}: {exc}"}

    rows, failures = [], []
    with (out / "raw.jsonl").open("w") as raw:
        for entry, ids, teacher in prepared:
            window_id = entry["window_id"]
            row = {"window_id": window_id, "domain": entry.get("domain"), "tokens": teacher["tokens"],
                   "teacher_nll_sum": teacher["nll_sum"], "candidate_nll_sum": None,
                   "teacher_top1_correct": teacher["top1_correct"], "candidate_top1_correct": None,
                   "candidate_top1_agrees_teacher": None, "request_wall_seconds": None,
                   "prompt_tokens": None, "completion_tokens": None, "response_sha256": None, "error": None}
            payload = dict(template, prompt=ids)
            t0 = time.monotonic()
            try:
                response, body = http_json(f"{args.base_url}/v1/completions", api_key, payload, args.request_timeout)
                row["request_wall_seconds"] = time.monotonic() - t0
                (out / "responses" / f"{window_id}.json").write_bytes(body)
                row["response_sha256"] = hashlib.sha256(body).hexdigest()
                usage = response.get("usage") if isinstance(response, dict) else None
                if isinstance(usage, dict):
                    row["prompt_tokens"], row["completion_tokens"] = usage.get("prompt_tokens"), usage.get("completion_tokens")
                reduced = reduce_prompt_response(response, ids, args.model, vocab)
                row["candidate_nll_sum"] = reduced["nll_sum"]
                row["candidate_top1_correct"] = reduced["top1_correct"]
                row["candidate_top1_agrees_teacher"] = sum(a == b for a, b in zip(reduced["top1_ids"], teacher["top1_ids"], strict=True))
            except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
                row["request_wall_seconds"] = row["request_wall_seconds"] or time.monotonic() - t0
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures.append({"window_id": window_id, "error": row["error"]})
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            print(f"candidate {window_id}: {row['error'] or f'nll_sum={row['candidate_nll_sum']:.6f} top1={row['candidate_top1_correct']}'}", flush=True)

    summary = summarize(rows, failures, args.nll_gate, args.top1_gate_pp)
    ended = time.time()
    manifest = {
        "script": Path(__file__).name, "script_sha256": sha256_file(Path(__file__)),
        "dataset_manifest_path": str(manifest_path), "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"), "model_revision": dataset_manifest.get("model_revision"),
        "vocab_size": vocab,
        "windows": [{"window_id": e["window_id"], "logits_sha256": e["sha256"], "token_ids_sha256": e["token_ids_sha256"],
                     "teacher_cache_hit": t["cache_hit"]} for e, _, t in prepared],
        "base_url": args.base_url, "model": args.model, "stack_label": args.stack_label, "profile_id": args.profile_id,
        "request_template": template, "models_response": models_response,
        "logprobs_mode_assumed": "raw_logprobs",
        "logprobs_mode_note": "not verifiable from the client; the launching profile must set --logprobs-mode raw_logprobs",
        "start_unix": started, "end_unix": ended, "host": host_facts(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    print(f"\n{'window':<12}{'domain':<28}{'dNLL':>12}{'top1 loss pp':>14}")
    for row in rows:
        if row["error"]:
            print(f"{row['window_id']:<12}{str(row['domain']):<28}{'FAILED':>12}  {row['error'][:60]}")
        else:
            d = (row["candidate_nll_sum"] - row["teacher_nll_sum"]) / row["tokens"]
            loss = 100.0 * (row["teacher_top1_correct"] - row["candidate_top1_correct"]) / row["tokens"]
            print(f"{row['window_id']:<12}{str(row['domain']):<28}{d:>12.6f}{loss:>14.4f}")
    m = summary["metrics"]
    print(f"\n{summary['verdict']}: delta_nll mean={m.get('delta_nll_mean')} upper95={m.get('delta_nll_upper_95')} "
          f"top1_loss_pp mean={m.get('top1_loss_pp_mean')} upper95={m.get('top1_loss_pp_upper_95')} "
          f"scored={summary['windows_scored']} failed={summary['windows_failed']}")
    return 0 if summary["verdict"] == "PASS" else 1


def summarize(rows: list[dict], failures: list[dict], nll_gate: float, top1_gate_pp: float) -> dict:
    good = [r for r in rows if r["error"] is None]
    metrics: dict = {}
    domains: dict = {}
    reasons = [f"{f['window_id']}: {f['error']}" for f in failures]
    if good:
        weights = [r["tokens"] for r in good]
        deltas = [(r["candidate_nll_sum"] - r["teacher_nll_sum"]) / r["tokens"] for r in good]
        losses = [100.0 * (r["teacher_top1_correct"] - r["candidate_top1_correct"]) / r["tokens"] for r in good]
        agree = [r["candidate_top1_agrees_teacher"] / r["tokens"] for r in good]
        total = sum(weights)
        metrics["delta_nll_mean"] = sum(d * w for d, w in zip(deltas, weights)) / total
        metrics["top1_loss_pp_mean"] = sum(l * w for l, w in zip(losses, weights)) / total
        metrics["top1_agreement_mean"] = sum(a * w for a, w in zip(agree, weights)) / total
        metrics["teacher_nll_mean"] = sum(r["teacher_nll_sum"] for r in good) / total
        metrics["candidate_nll_mean"] = sum(r["candidate_nll_sum"] for r in good) / total
        try:
            metrics["delta_nll_upper_95"] = _weighted_upper_95(deltas, weights)[1]
            metrics["top1_loss_pp_upper_95"] = _weighted_upper_95(losses, weights)[1]
        except ValueError as exc:
            metrics["delta_nll_upper_95"] = metrics["top1_loss_pp_upper_95"] = None
            reasons.append(f"upper bound unavailable: {exc}")
        for r, d, l, a in zip(good, deltas, losses, agree):
            bucket = domains.setdefault(str(r["domain"]), {"windows": 0, "tokens": 0, "_d": 0.0, "_l": 0.0, "_a": 0.0})
            bucket["windows"] += 1
            bucket["tokens"] += r["tokens"]
            bucket["_d"] += d * r["tokens"]; bucket["_l"] += l * r["tokens"]; bucket["_a"] += a * r["tokens"]
        for bucket in domains.values():
            bucket["delta_nll_mean"] = bucket.pop("_d") / bucket["tokens"]
            bucket["top1_loss_pp_mean"] = bucket.pop("_l") / bucket["tokens"]
            bucket["top1_agreement_mean"] = bucket.pop("_a") / bucket["tokens"]
    else:
        reasons.append("no window scored")
    checks = {
        "delta_nll_mean_le_gate": metrics.get("delta_nll_mean") is not None and metrics["delta_nll_mean"] <= nll_gate,
        "delta_nll_upper_95_le_gate": metrics.get("delta_nll_upper_95") is not None and metrics["delta_nll_upper_95"] <= nll_gate,
        "top1_loss_pp_mean_le_gate": metrics.get("top1_loss_pp_mean") is not None and metrics["top1_loss_pp_mean"] <= top1_gate_pp,
        "all_windows_scored": not failures,
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"{name} failed (delta_nll gate {nll_gate}, top1 loss gate {top1_gate_pp} pp)")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {"verdict": verdict, "gates": {"nll_gate": nll_gate, "top1_gate_pp": top1_gate_pp}, "metrics": metrics,
            "checks": checks, "windows_scored": len(good), "windows_failed": len(failures), "failures": failures,
            "fail_reasons": reasons if verdict == "FAIL" else [], "per_domain": domains, "formulas": FORMULAS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stack-label", required=True)
    parser.add_argument("--api-key-file")
    parser.add_argument("--windows", help="e.g. 0-24 or 0,3,5 (default: all)")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--profile-id")
    parser.add_argument("--teacher-cache")
    parser.add_argument("--nll-gate", type=float, default=0.01)
    parser.add_argument("--top1-gate-pp", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        return score_windows(args)
    except (DatasetError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"dataset/usage error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
