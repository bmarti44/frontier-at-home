#!/usr/bin/env python3
"""Reusable profile qualification kit: one command, one evidence bundle.

Composes the gate scripts that already exist (speed, tool-call, vision,
teacher-logit fidelity, accuracy, full-context fill, soak) against one
declarative profile (docs/PROFILE-SCHEMA.md) and writes a single bundle:

    <out>/manifest.json      resolved profile, host facts, script digests,
                             exact argv of every cell (also under --dry-run)
    <out>/<cell>/log.txt     captured stdout+stderr of that cell's script
    <out>/<cell>/...         the cell script's own evidence files
    <out>/summary.json       parsed per-cell results, targets, baseline, verdict
    <out>/SUMMARY.md         the human table

Nothing here measures anything itself: every number is parsed from the
composed script's own output file, and a cell whose script fails is
recorded FAIL with the tail of its log. See docs/QUALIFY-PROFILE.md.

Plain Python 3.12 stdlib.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
PROFILES_ROOT = REPO_ROOT / "configs" / "profiles"

CELL_ORDER = ("speed", "toolcall", "vision", "media", "teacher", "accuracy", "context", "soak")
DEFAULT_CELLS = ("speed", "toolcall", "vision", "media", "teacher", "accuracy", "context")
ACCURACY_SUITES = ("gsm8k", "mmlu-pro", "humaneval")
CELL_SCRIPTS = {
    "speed": "30_bench_speed.py",
    "toolcall": "39_bench_toolcall.py",
    "vision": "38_bench_vision.py",
    "media": "51_probe_media_max.py",
    "teacher": "49_score_teacher_windows.py",
    "accuracy": "31_bench_accuracy.py",
    "context": "50_probe_context.py",
    "soak": "35_soak.py",
}
COMPOSED_SCRIPTS = ("92_resolve_profile.py", "93_profile_serve.sh", *CELL_SCRIPTS.values())

# Model slug -> chat encoder registered in scripts/31_bench_accuracy.py
# (ENCODER_PATHS). A model without a row here gets its accuracy cell SKIPPED;
# the kit never writes an encoder. Override with --encoder.
MODEL_ENCODERS = {
    "deepseek-v4-flash": "dsv4",
    "laguna-s-2.1": "laguna",
    "qwen3.8-27b": "qwen38",
    "glm-5.3-flash": "glm53",
}

SPEED_CONTEXT_LEVELS = "0,28672"
SPEED_EXTRA_BODY = '{"min_tokens":320}'
LOG_TAIL_LINES = 40

# Prompt tokens held back from each slot for the chat template and the answer.
CONTEXT_FILL_HEADROOM = 12_016

# Metric -> (target key in qualification_targets, comparison, per-ctx?)
# "min": measured >= target passes; "max": measured <= target passes.


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def tail_text(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


# ----------------------------------------------------------------------------
# Profile resolution
# ----------------------------------------------------------------------------


class KitError(RuntimeError):
    pass


def split_profile_id(profile_id: str) -> tuple[str, str]:
    if "/" not in profile_id:
        raise KitError(f"--profile must be <model>/<file>, got {profile_id!r}")
    model_slug, name = profile_id.split("/", 1)
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return model_slug, name


def run_resolver(verb: str, profile_id: str, host: str | None, lifecycle_verb: str = "start") -> subprocess.CompletedProcess:
    argv = [sys.executable, str(SCRIPTS / "92_resolve_profile.py"), verb, "--profile", profile_id]
    if host:
        argv += ["--host", host]
    if verb == "render":
        argv += ["--verb", lifecycle_verb]
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def argv_value(argv: list[str], flag: str) -> str | None:
    """First value following `flag` (or `flag=value`)."""
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(flag + "="):
            return item[len(flag) + 1:]
    return None


def load_raw_profile(model_slug: str, name: str) -> dict:
    """The leaf profile JSON exactly as committed (the resolver's allowlist
    rejects keys it does not know, so optional kit blocks are read raw)."""
    path = PROFILES_ROOT / model_slug / f"{name}.json"
    document = read_json(path)
    if "extends" in document:
        base = read_json(PROFILES_ROOT / model_slug / document["extends"])
        merged = dict(base)
        merged.update(document)
        document = merged
    return document


def load_raw_model(model_slug: str) -> dict:
    return read_json(PROFILES_ROOT / model_slug / "model.json")


def substitute_paths(value: str, host: dict) -> str:
    paths = host.get("paths", {})
    for key in ("repo", "model_root", "cache_root", "state_root"):
        if key in paths:
            value = value.replace("{" + key + "}", str(paths[key]))
    return value


def load_host_document(host: str | None) -> dict:
    sys.path.insert(0, str(SCRIPTS / "lib"))
    try:
        import profile_resolver  # type: ignore
        return profile_resolver.load_host(host)
    except Exception:  # noqa: BLE001 - host facts are optional for the kit
        if host:
            try:
                return read_json(Path(host))
            except OSError:
                pass
        return {}


def resolve_profile(args: argparse.Namespace) -> dict:
    model_slug, name = split_profile_id(args.profile)
    rendered = run_resolver("render", args.profile, args.host)
    if rendered.returncode != 0:
        raise KitError(f"profile render failed: {rendered.stderr.strip() or rendered.stdout.strip()}")
    snapshot = json.loads(rendered.stdout)
    check = run_resolver("check", args.profile, args.host)
    check_result = {
        "exit_code": check.returncode,
        "stdout": check.stdout.strip(),
        "stderr": check.stderr.strip(),
    }
    raw_profile = load_raw_profile(model_slug, name)
    raw_model = load_raw_model(model_slug)
    host_doc = load_host_document(args.host)

    argv = list(snapshot.get("argv", []))
    served = args.served_model
    if served is None:
        served = argv_value(argv, "--served-model-name") or argv_value(argv, "--alias")
    model_path = argv_value(argv, "--model") or argv_value(argv, "-m")
    tokenizer_path = None
    tokenizer_sha = None
    if model_path:
        candidate = Path(model_path)
        if candidate.is_dir():
            candidate = candidate / "tokenizer.json"
        else:
            candidate = candidate.parent / "tokenizer.json"
        tokenizer_path = str(candidate)
        tokenizer_sha = sha256_file(candidate)
    # Expected tokenizer digest from model.json if the file itself is absent.
    expected_tokenizer_sha = None
    for check_entry in snapshot.get("digest_checks", []):
        if str(check_entry.get("path", "")).endswith("tokenizer.json"):
            expected_tokenizer_sha = check_entry.get("sha256")
            if tokenizer_path is None:
                tokenizer_path = check_entry["path"]

    serving = snapshot.get("serving") or raw_profile.get("serving") or {}
    slots = serving.get("parallel_slots")
    tokens_per_slot = serving.get("tokens_per_slot") or serving.get("request_context_cap")
    context_cap = snapshot.get("context_cap") or raw_profile.get("context_cap")
    if slots is None:
        parallel = argv_value(argv, "-np") or argv_value(argv, "--parallel") or argv_value(argv, "--max-num-seqs")
        slots = int(parallel) if parallel else 1
    if tokens_per_slot is None and context_cap:
        tokens_per_slot = int(context_cap) // int(slots)

    safety = snapshot.get("safety") or raw_profile.get("safety") or {}
    binary = snapshot.get("binary")
    binary_sha = sha256_file(Path(binary)) if binary else None
    expected_binary_sha = None
    for check_entry in snapshot.get("digest_checks", []):
        if binary and check_entry.get("path") == binary:
            expected_binary_sha = check_entry.get("sha256")

    reference_logits = raw_model.get("reference_logits")
    if args.reference_logits_dir:
        reference_logits = {"dataset_dir": args.reference_logits_dir, "note": "--reference-logits-dir override"}
    if isinstance(reference_logits, dict) and isinstance(reference_logits.get("dataset_dir"), str):
        reference_logits = dict(reference_logits)
        reference_logits["dataset_dir"] = substitute_paths(reference_logits["dataset_dir"], host_doc)
        if not Path(reference_logits["dataset_dir"]).is_absolute():
            reference_logits["dataset_dir"] = str(REPO_ROOT / reference_logits["dataset_dir"])

    targets = raw_profile.get("qualification_targets")
    options = raw_profile.get("qualification_options") or {}
    vision_thinking_mode = args.vision_thinking_mode or options.get("vision_thinking_mode") or "chat"
    if args.targets:
        targets = read_json(Path(args.targets))

    encoder = args.encoder or MODEL_ENCODERS.get(model_slug)

    config_material = json.dumps({"argv": argv, "env": snapshot.get("env", {})}, sort_keys=True).encode()
    config_hash = f"{snapshot.get('stack_label') or name}-{hashlib.sha256(config_material).hexdigest()[:12]}"

    return {
        "profile_id": snapshot.get("profile_id", args.profile),
        "model_slug": model_slug,
        "profile_file": str(PROFILES_ROOT / model_slug / f"{name}.json"),
        "profile_status": snapshot.get("status"),
        "profile_evidence": (raw_profile.get("status") or {}).get("evidence"),
        "mechanism": snapshot.get("mechanism"),
        "stack_label": snapshot.get("stack_label") or name,
        "switch_alias": snapshot.get("switch_alias"),
        "rendered_port": snapshot.get("port"),
        "served_model": served,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_sha256_expected": expected_tokenizer_sha,
        "slots": int(slots) if slots is not None else None,
        "tokens_per_slot": int(tokens_per_slot) if tokens_per_slot is not None else None,
        "context_cap": context_cap,
        "kill_floor_gib": safety.get("kill_floor_gib"),
        "minimum_start_gib": safety.get("minimum_start_gib"),
        "startup_timeout_seconds": safety.get("startup_timeout_seconds", 1800),
        "safety": safety,
        "serving": serving,
        "containment": snapshot.get("containment") or raw_profile.get("containment"),
        "digest_checks": snapshot.get("digest_checks", []),
        "binary": binary,
        "binary_sha256": binary_sha,
        "binary_sha256_expected": expected_binary_sha,
        "argv": argv,
        "env": snapshot.get("env", {}),
        "check": check_result,
        "reference_logits": reference_logits,
        "qualification_targets": targets,
        "qualification_options": options or None,
        "vision_thinking_mode": vision_thinking_mode,
        "encoder": encoder,
        "config_hash": config_hash,
    }


# ----------------------------------------------------------------------------
# Host facts
# ----------------------------------------------------------------------------


def host_facts() -> dict:
    facts: dict[str, Any] = {
        "uname": " ".join(platform.uname()),
        "hostname": platform.node(),
        "nproc": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                facts["mem_total_kib"] = int(line.split()[1])
                facts["mem_total_gib"] = round(int(line.split()[1]) / 2**20, 2)
                break
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                name, _, driver = out.stdout.strip().splitlines()[0].partition(",")
                facts["gpu_name"] = name.strip()
                facts["gpu_driver"] = driver.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return facts


def git_head() -> dict:
    result: dict[str, Any] = {}
    for key, cmd in (("head", ["git", "rev-parse", "HEAD"]), ("branch", ["git", "branch", "--show-current"])):
        try:
            out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=20, check=False)
            result[key] = out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            result[key] = None
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20, check=False)
        result["dirty"] = bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        result["dirty"] = None
    return result


# ----------------------------------------------------------------------------
# Cell planning
# ----------------------------------------------------------------------------


def parse_media_limits(argv: list[str]) -> dict | None:
    """vLLM-style --limit-mm-per-prompt JSON -> {images, image_size, video_frames, video_size}; None if absent."""
    raw = argv_value(argv, "--limit-mm-per-prompt")
    if not raw:
        return None
    try:
        limits = json.loads(raw)
    except ValueError:
        return None
    image = limits.get("image") if isinstance(limits, dict) else None
    video = limits.get("video") if isinstance(limits, dict) else None
    if not isinstance(image, dict) or not isinstance(video, dict):
        return None
    try:
        return {"images": int(image.get("count", 1)), "image_size": int(min(image.get("width", 512), image.get("height", 512))),
                "video_frames": int(video.get("num_frames", 16)),
                "video_size": int(min(video.get("width", 512), video.get("height", 512)))}
    except (TypeError, ValueError):
        return None


def plan_cells(resolved: dict, args: argparse.Namespace, out: Path, base_url: str) -> dict[str, dict]:
    """Return {cell: {"argv": [...] | None, "skip_reason": str|None, "evidence": path}}."""
    py = sys.executable
    label = resolved["stack_label"]
    served = resolved["served_model"]
    tok_path = resolved["tokenizer_path"]
    tok_sha = resolved["tokenizer_sha256"] or resolved["tokenizer_sha256_expected"]
    plans: dict[str, dict] = {}

    def cell_dir(name: str) -> Path:
        return out / name

    # speed -----------------------------------------------------------------
    speed_argv = [
        py, str(SCRIPTS / CELL_SCRIPTS["speed"]),
        "--base-url", base_url, "--out", str(cell_dir("speed") / "speed.json"),
        "--stack-label", label, "--reps", "2", "--warmup", "1",
        "--max-tokens", "384", "--min-completion-tokens", "200", "--seed", "42",
        "--context-levels", SPEED_CONTEXT_LEVELS, "--ignore-eos-supported",
        "--request-timeout", "900", "--extra-body", SPEED_EXTRA_BODY,
    ]
    if served:
        speed_argv += ["--model-id", served]
    skip = None
    if tok_path and tok_sha:
        speed_argv += ["--output-tokenizer-path", tok_path, "--output-tokenizer-sha256", tok_sha]
    else:
        skip = "tokenizer.json not found next to the rendered --model path; cannot validate output token ids"
    plans["speed"] = {"argv": speed_argv, "skip_reason": skip, "evidence": "speed/speed.json"}

    # toolcall --------------------------------------------------------------
    plans["toolcall"] = {
        "argv": [py, str(SCRIPTS / CELL_SCRIPTS["toolcall"]), "--base-url", base_url + "/v1",
                 "--model", served or "", "--out", str(cell_dir("toolcall") / "toolcall.json"),
                 "--temperature", "0"],
        "skip_reason": None if served else "served model name unknown (pass --served-model)",
        "evidence": "toolcall/toolcall.json",
    }

    # vision ----------------------------------------------------------------
    # Template mode for the answer: chat (non-thinking) by default, the mode of
    # the README rows (thinking models spend the 256-token answer budget
    # reasoning otherwise). A profile whose model answers unparseably in chat
    # mode declares qualification_options.vision_thinking_mode; --vision-thinking-mode
    # overrides. The mode is part of the recorded argv either way.
    vision_argv = [py, str(SCRIPTS / CELL_SCRIPTS["vision"]), "--base-url", base_url,
                   "--out", str(cell_dir("vision")), "--thinking-mode", resolved["vision_thinking_mode"]]
    if args.vision_limit:
        vision_argv += ["--limit", str(args.vision_limit)]
    if served:
        vision_argv += ["--model-id", served]
    plans["vision"] = {"argv": vision_argv, "skip_reason": None, "evidence": "vision/summary.json"}

    # media (declared maximum) -----------------------------------------------
    media_limits = parse_media_limits(resolved.get("argv") or [])
    if media_limits is None:
        plans["media"] = {"argv": None, "evidence": "media/summary.json",
                          "skip_reason": "profile declares no --limit-mm-per-prompt image/video maximum"}
    else:
        media_argv = [py, str(SCRIPTS / CELL_SCRIPTS["media"]), "--base-url", base_url, "--model", served or "",
                      "--out", str(cell_dir("media")), "--stack-label", label,
                      "--images", str(media_limits["images"]), "--image-size", str(media_limits["image_size"]),
                      "--video-frames", str(media_limits["video_frames"]), "--video-size", str(media_limits["video_size"])]
        plans["media"] = {"argv": media_argv, "skip_reason": None, "evidence": "media/summary.json"}

    # teacher ---------------------------------------------------------------
    ref = resolved.get("reference_logits")
    if isinstance(ref, dict) and ref.get("dataset_dir"):
        teacher_argv = [py, str(SCRIPTS / CELL_SCRIPTS["teacher"]), "--base-url", base_url,
                        "--model", served or "", "--dataset-dir", str(ref["dataset_dir"]),
                        "--out", str(cell_dir("teacher")), "--stack-label", label,
                        "--profile-id", resolved["profile_id"]]
        if args.teacher_windows:
            teacher_argv += ["--windows", args.teacher_windows]
        plans["teacher"] = {"argv": teacher_argv, "skip_reason": None, "evidence": "teacher/summary.json"}
    else:
        plans["teacher"] = {"argv": None, "evidence": "teacher/summary.json",
                            "skip_reason": "model.json declares no reference_logits block (BF16 teacher logits dataset)"}

    # accuracy --------------------------------------------------------------
    encoder = resolved.get("encoder")
    if encoder:
        suites = {}
        for suite in ACCURACY_SUITES:
            # HumanEval has no dev/holdout split (31_bench_accuracy.py accepts
            # --split all only); GSM8K and MMLU-Pro take the requested split.
            split = "all" if suite == "humaneval" else args.accuracy_split
            suite_argv = [py, str(SCRIPTS / CELL_SCRIPTS["accuracy"]), "--base-url", base_url,
                          "--out", str(cell_dir("accuracy") / f"acc-{suite}.json"),
                          "--stack-label", label, "--suite", suite, "--split", split,
                          "--transcripts-dir", str(cell_dir("accuracy") / "transcripts" / suite),
                          "--encoder", encoder, "--max-tokens", "16384",
                          "--request-timeout", "2700",
                          "--profile-id", resolved["profile_id"],
                          "--config-hash", resolved["config_hash"]]
            if args.reasoning_effort:
                suite_argv += ["--reasoning-effort", args.reasoning_effort]
            if args.accuracy_split == "holdout":
                suite_argv += ["--config-evidence", str(cell_dir("accuracy") / "config-evidence.json")]
            suites[suite] = suite_argv
        plans["accuracy"] = {"argv": None, "suites": suites, "skip_reason": None, "evidence": "accuracy/acc-<suite>.json"}
    else:
        plans["accuracy"] = {"argv": None, "suites": {}, "evidence": "accuracy/",
                             "skip_reason": ("requires an encoder registered for the model in "
                                             "scripts/31_bench_accuracy.py ENCODER_PATHS (none declared for "
                                             f"{resolved['model_slug']!r}; pass --encoder if one exists)")}

    # context ---------------------------------------------------------------
    context_skip = None
    if not (tok_path and tok_sha):
        context_skip = "tokenizer.json missing; the fill probe needs the model tokenizer"
    elif not (resolved.get("slots") and resolved.get("tokens_per_slot")):
        context_skip = "profile declares no serving topology (slots x tokens_per_slot)"
    # Each slot's cap covers prompt + chat template + generated answer. Fill
    # tokens_per_slot - CONTEXT_FILL_HEADROOM so the request is admitted at the
    # cap (matches the 250,128-token fills of the historical 262,144 gates).
    fill_tokens = (int(resolved["tokens_per_slot"]) - CONTEXT_FILL_HEADROOM) if resolved.get("tokens_per_slot") else None
    if fill_tokens is not None and fill_tokens <= 0:
        context_skip = context_skip or f"tokens_per_slot {resolved['tokens_per_slot']} leaves no room below the fill headroom"
    context_argv = [py, str(SCRIPTS / CELL_SCRIPTS["context"]), "--base-url", base_url,
                    "--model", served or "", "--out", str(cell_dir("context")), "--stack-label", label,
                    "--slots", str(resolved.get("slots")), "--tokens-per-slot", str(fill_tokens),
                    "--tokenizer-path", str(tok_path), "--tokenizer-sha256", str(tok_sha),
                    "--phase", "both", "--profile-id", resolved["profile_id"]]
    if resolved.get("kill_floor_gib") is not None:
        context_argv += ["--mem-floor-gib", str(resolved["kill_floor_gib"])]
    plans["context"] = {"argv": context_argv, "skip_reason": context_skip, "evidence": "context/summary.json"}

    # soak ------------------------------------------------------------------
    soak_argv = [py, str(SCRIPTS / CELL_SCRIPTS["soak"]), "--base-url", base_url, "--stack-label", label,
                 "--config-hash", resolved["config_hash"], "--out", str(cell_dir("soak") / "soak.json")]
    if served:
        soak_argv += ["--model", served]
    plans["soak"] = {"argv": soak_argv, "skip_reason": None if args.with_soak else "off by default (--with-soak)",
                     "evidence": "soak/soak.json"}
    return plans


def select_cells(args: argparse.Namespace) -> list[str]:
    if getattr(args, "resummarize", False):
        return []  # rebuild summary.json / SUMMARY.md from the bundle's existing evidence only
    wanted = list(DEFAULT_CELLS)
    if args.cells:
        wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    if args.with_soak and "soak" not in wanted:
        wanted.append("soak")
    skipped = {c.strip() for c in (args.skip or "").split(",") if c.strip()}
    unknown = [c for c in wanted + sorted(skipped) if c not in CELL_ORDER]
    if unknown:
        raise KitError(f"unknown cell(s) {unknown}; known: {', '.join(CELL_ORDER)}")
    return [c for c in CELL_ORDER if c in wanted and c not in skipped]


# ----------------------------------------------------------------------------
# Parsers (one per cell output format)
# ----------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def parse_speed(document: dict) -> dict:
    """scripts/30_bench_speed.py output -> per ctx level medians."""
    levels: dict[str, dict] = {}
    for cell in document.get("cells", []):
        reps = cell.get("reps", [])
        levels[str(cell.get("ctx_tokens"))] = {
            "median_decode_tok_s": cell.get("median_decode"),
            "median_ttft_s": cell.get("median_ttft"),
            "median_prefill_tok_s": _median([r.get("prefill_tok_s") for r in reps]),
            "valid": bool(cell.get("valid")),
            "reps": len(reps),
        }
    return {"levels": levels, "suite_valid": bool(document.get("suite_valid")),
            "stack_label": (document.get("metadata") or {}).get("stack_label")}


def parse_toolcall(document: dict) -> dict:
    """scripts/39_bench_toolcall.py output."""
    per_case = document.get("per_case") or []
    passed = document.get("passed")
    total = document.get("total")
    if passed is None:
        passed = sum(1 for c in per_case if c.get("passed"))
    if total is None:
        total = len(per_case)
    return {"passed": passed, "total": total, "score": document.get("score"), "suite": document.get("suite")}


def parse_vision(document: dict) -> dict:
    """scripts/38_bench_vision.py summary.json."""
    return {"accuracy": document.get("accuracy"), "n": document.get("n"), "correct": document.get("correct"),
            "invalid_count": document.get("invalid_count"), "error_count": document.get("error_count"),
            "ok": document.get("ok"), "suite": document.get("suite")}


def parse_media(document: dict) -> dict:
    """scripts/51_probe_media_max.py summary.json."""
    return {"verdict": document.get("verdict"), "checks": document.get("checks"), "declared": document.get("declared")}


def parse_teacher(document: dict) -> dict:
    """scripts/49_score_teacher_windows.py summary.json."""
    metrics = document.get("metrics") or {}
    return {"verdict": document.get("verdict"),
            "delta_nll_mean": metrics.get("delta_nll_mean"),
            "delta_nll_upper_95": metrics.get("delta_nll_upper_95"),
            "top1_loss_pp_mean": metrics.get("top1_loss_pp_mean"),
            "top1_loss_pp_upper_95": metrics.get("top1_loss_pp_upper_95"),
            "top1_agreement_mean": metrics.get("top1_agreement_mean"),
            "windows_scored": document.get("windows_scored"),
            "windows_failed": document.get("windows_failed"),
            "fail_reasons": document.get("fail_reasons")}


def parse_accuracy(documents: dict[str, dict]) -> dict:
    """{suite: scripts/31_bench_accuracy.py output} -> per suite accuracy."""
    suites = {}
    for suite, document in documents.items():
        suites[suite] = {"accuracy": document.get("accuracy"), "n": document.get("n"),
                         "correct": document.get("correct"), "invalid_count": document.get("invalid_count"),
                         "split": document.get("split"), "wilson95": document.get("wilson95")}
    return {"suites": suites}


def parse_context(document: dict) -> dict:
    """scripts/50_probe_context.py summary.json."""
    return {"verdict": document.get("verdict"), "total_tokens": document.get("total_tokens"),
            "min_mem_available_gib": document.get("min_mem_available_gib"),
            "memory_floor_gib": document.get("memory_floor_gib"),
            "needles_correct": document.get("needles_correct"), "needles_total": document.get("needles_total"),
            "max_ttft_s": document.get("max_ttft_s"), "checks": document.get("checks")}


def parse_soak(document: dict) -> dict:
    """scripts/35_soak.py output."""
    return {"pass": bool(document.get("pass")), "failed_gates": document.get("failed_gates"),
            "n_requests": document.get("n_requests"), "n_errors": document.get("n_errors"),
            "decode_overall_median_tok_s": document.get("decode_overall_median_tok_s"),
            "mem_available_min_gib": document.get("mem_available_min_gib"),
            "duration_seconds_actual": document.get("duration_seconds_actual")}


def parse_cell_evidence(cell: str, cell_dir: Path, plan: dict) -> dict | None:
    """Locate and parse the cell's own evidence; None when absent."""
    if cell == "accuracy":
        documents = {}
        for suite in plan.get("suites", {}):
            path = cell_dir / f"acc-{suite}.json"
            if path.is_file():
                documents[suite] = read_json(path)
        return parse_accuracy(documents) if documents else None
    paths = {
        "speed": cell_dir / "speed.json",
        "toolcall": cell_dir / "toolcall.json",
        "vision": cell_dir / "summary.json",
        "media": cell_dir / "summary.json",
        "teacher": cell_dir / "summary.json",
        "context": cell_dir / "summary.json",
        "soak": cell_dir / "soak.json",
    }
    path = paths[cell]
    if not path.is_file():
        return None
    document = read_json(path)
    parser = {"speed": parse_speed, "toolcall": parse_toolcall, "vision": parse_vision, "media": parse_media,
              "teacher": parse_teacher, "context": parse_context, "soak": parse_soak}[cell]
    return parser(document)


# ----------------------------------------------------------------------------
# Rows, targets, baseline, verdict
# ----------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _compare(measured: Any, target: Any, mode: str) -> str:
    if measured is None:
        return "FAIL"
    if mode == "min":
        return "PASS" if measured >= target else "FAIL"
    if mode == "max":
        return "PASS" if measured <= target else "FAIL"
    if mode == "eq":
        return "PASS" if measured == target else "FAIL"
    raise ValueError(mode)


def build_rows(cell: str, result: dict | None, targets: dict | None, baseline: dict | None) -> list[dict]:
    """Flatten a parsed cell result into (metric, measured, target, status) rows."""
    targets = targets or {}
    rows: list[dict] = []

    def row(metric: str, measured: Any, target: Any = None, mode: str | None = None) -> None:
        status = "measured"
        if target is not None and mode is not None:
            status = _compare(measured, target, mode)
        rows.append({"cell": cell, "metric": metric, "measured": measured, "target": target,
                     "target_mode": mode, "baseline": lookup_baseline(baseline, cell, metric), "status": status})

    if result is None:
        return rows
    if cell == "speed":
        for ctx, level in sorted(result.get("levels", {}).items(), key=lambda kv: int(kv[0])):
            row(f"decode_tok_s@{ctx}", level.get("median_decode_tok_s"),
                (targets.get("decode_tok_s_min") or {}).get(ctx), "min")
            row(f"ttft_s@{ctx}", level.get("median_ttft_s"), (targets.get("ttft_s_max") or {}).get(ctx), "max")
            row(f"prefill_tok_s@{ctx}", level.get("median_prefill_tok_s"),
                (targets.get("prefill_tok_s_min") or {}).get(ctx), "min")
            row(f"valid@{ctx}", level.get("valid"), True if targets else None, "eq")
    elif cell == "toolcall":
        row("passed", result.get("passed"), targets.get("toolcall_min"), "min")
        row("total", result.get("total"))
    elif cell == "vision":
        row("accuracy", result.get("accuracy"), targets.get("vision_min"), "min")
        row("n", result.get("n"))
    elif cell == "media":
        row("verdict", result.get("verdict"), "PASS" if targets.get("media_pass") else None, "eq")
        for name, ok in sorted((result.get("checks") or {}).items()):
            row(name, ok)
    elif cell == "teacher":
        row("delta_nll_mean", result.get("delta_nll_mean"), targets.get("delta_nll_max"), "max")
        row("delta_nll_upper_95", result.get("delta_nll_upper_95"), targets.get("delta_nll_max"), "max")
        row("top1_loss_pp_mean", result.get("top1_loss_pp_mean"), targets.get("top1_loss_pp_max"), "max")
        row("top1_loss_pp_upper_95", result.get("top1_loss_pp_upper_95"), targets.get("top1_loss_pp_max"), "max")
        row("verdict", result.get("verdict"), "PASS" if targets.get("teacher_pass") else None, "eq")
    elif cell == "accuracy":
        minimums = targets.get("accuracy_min") or {}
        for suite, values in sorted(result.get("suites", {}).items()):
            row(f"{suite}", values.get("accuracy"), minimums.get(suite), "min")
    elif cell == "context":
        row("verdict", result.get("verdict"), "PASS" if targets.get("context_pass") else None, "eq")
        row("total_tokens", result.get("total_tokens"), targets.get("context_tokens_min"), "min")
        row("min_mem_available_gib", result.get("min_mem_available_gib"), targets.get("context_mem_floor_gib"), "min")
    elif cell == "soak":
        row("pass", result.get("pass"), True if targets.get("soak_pass") else None, "eq")
        row("decode_overall_median_tok_s", result.get("decode_overall_median_tok_s"))
    return rows


def lookup_baseline(baseline: dict | None, cell: str, metric: str) -> Any:
    if not baseline:
        return None
    for row in baseline.get("rows", []):
        if row.get("cell") == cell and row.get("metric") == metric:
            return row.get("measured")
    return None


def load_baseline(args: argparse.Namespace) -> dict | None:
    """A prior bundle of this kit (summary.json), from --baseline-results or a profile's status.evidence."""
    if args.baseline_results:
        path = Path(args.baseline_results)
        if path.is_dir():
            path = path / "summary.json"
        document = read_json(path)
        document["_source"] = str(path)
        return document
    if args.baseline:
        model_slug, name = split_profile_id(args.baseline)
        raw = load_raw_profile(model_slug, name)
        evidence = (raw.get("status") or {}).get("evidence")
        if not evidence:
            return {"_source": None, "_error": f"{args.baseline}: status.evidence not set", "rows": []}
        evidence_dir = REPO_ROOT / evidence
        candidates = sorted(list(evidence_dir.glob("qualify-*/summary.json")) + list(evidence_dir.glob("summary.json")))
        candidates = [c for c in candidates if _is_kit_summary(c)]
        if not candidates:
            return {"_source": str(evidence_dir), "_error": "no qualify-*/summary.json of this kit under status.evidence "
                                                          "(use --baseline-results <dir> for a manual bundle)", "rows": []}
        document = read_json(candidates[-1])
        document["_source"] = str(candidates[-1])
        return document
    return None


def _is_kit_summary(path: Path) -> bool:
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and document.get("kind") == "qualify-profile"


def overall_verdict(cells: dict[str, dict], rows: list[dict]) -> str:
    errored = any(c.get("status") == "FAIL" for c in cells.values())
    targeted = [r for r in rows if r.get("target") is not None]
    if errored:
        return "FAIL"
    if any(r["status"] == "FAIL" for r in targeted):
        return "FAIL"
    if not targeted:
        return "MEASURED"
    return "PASS"


def render_summary_md(summary: dict) -> str:
    lines = [f"# Qualification: {summary['profile_id']}", ""]
    lines.append(f"- verdict: **{summary['verdict']}**")
    lines.append(f"- bundle: `{summary['out']}`")
    lines.append(f"- started/ended: {summary.get('start_iso')} / {summary.get('end_iso')}")
    lines.append(f"- stack label: `{summary.get('stack_label')}`; served model: `{summary.get('served_model')}`")
    if summary.get("baseline_source"):
        lines.append(f"- baseline: `{summary['baseline_source']}`")
    if summary.get("baseline_error"):
        lines.append(f"- baseline: {summary['baseline_error']}")
    lines.append(f"- targets: {'declared' if summary.get('targets') else 'none (measured only)'}")
    lines.append("")
    has_baseline = bool(summary.get("baseline_source"))
    header = "| cell | metric | measured | target |" + (" baseline |" if has_baseline else "") + " status | evidence |"
    sep = "|---|---|---|---|" + ("---|" if has_baseline else "") + "---|---|"
    lines += [header, sep]
    for name in CELL_ORDER:
        cell = summary["cells"].get(name)
        if cell is None:
            continue
        evidence = cell.get("evidence") or f"{name}/"
        rows = [r for r in summary["rows"] if r["cell"] == name]
        if cell["status"] == "SKIPPED":
            lines.append(f"| {name} | - | - | - |" + (" - |" if has_baseline else "") +
                         f" SKIPPED | {cell.get('skip_reason', '')} |")
            continue
        if not rows:
            lines.append(f"| {name} | - | - | - |" + (" - |" if has_baseline else "") +
                         f" {cell['status']} | {evidence} (exit {cell.get('exit_code')}) |")
        for row in rows:
            target = _fmt(row["target"])
            if row["target"] is not None and row.get("target_mode") in ("min", "max"):
                target = (">= " if row["target_mode"] == "min" else "<= ") + target
            status = row["status"] if cell["status"] != "FAIL" else f"FAIL ({cell.get('status_reason') or 'see log'})"
            lines.append(f"| {name} | {row['metric']} | {_fmt(row['measured'])} | {target} |" +
                         (f" {_fmt(row['baseline'])} |" if has_baseline else "") + f" {status} | {evidence} |")
    lines.append("")
    failed = [(n, c) for n, c in summary["cells"].items() if c["status"] == "FAIL"]
    if failed:
        lines.append("## Failed cells")
        for name, cell in failed:
            lines.append(f"- **{name}**: {cell.get('status_reason')} (exit {cell.get('exit_code')}, log `{name}/log.txt`)")
            tail = cell.get("log_tail") or ""
            if tail:
                lines += ["", "```", tail[-2000:], "```", ""]
    lines.append("Every number above is parsed from the composed script's own output; "
                 "see manifest.json for the exact argv of each cell.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------------


def serve(verb: str, args: argparse.Namespace, log_path: Path) -> subprocess.CompletedProcess:
    argv = ["bash", str(SCRIPTS / "93_profile_serve.sh"), "--profile", args.profile]
    if args.host:
        argv += ["--host", args.host]
    if args.port:
        argv += ["--port", str(args.port)]
    argv.append(verb)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n")
        log.flush()
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(proc.stdout)
    return proc


def list_models(base_url: str, timeout: float = 5.0) -> list[str]:
    request = urllib.request.Request(base_url + "/v1/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback only
        document = json.loads(response.read().decode("utf-8"))
    return [str(item.get("id")) for item in document.get("data", [])]


def wait_for_model(base_url: str, served: str | None, timeout_s: float) -> list[str]:
    deadline = time.monotonic() + timeout_s
    last_error: str = ""
    while time.monotonic() < deadline:
        try:
            ids = list_models(base_url)
            if served is None or served in ids:
                return ids
            last_error = f"/v1/models lists {ids}, not {served!r}"
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = str(error)
        time.sleep(3)
    raise KitError(f"server on {base_url} not ready within {timeout_s:.0f}s: {last_error}")


# ----------------------------------------------------------------------------
# Cell execution
# ----------------------------------------------------------------------------


def run_subprocess(argv: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n")
        log.flush()
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT, check=False)
    return proc.returncode, time.monotonic() - started


def write_config_evidence(cell_dir: Path, resolved: dict) -> None:
    """31_bench_accuracy.py --config-evidence: needs one server_binary_sha256 and a weights manifest."""
    document = {
        "kind": "qualify-profile-config-evidence",
        "profile_id": resolved["profile_id"],
        "server_binary_sha256": resolved.get("binary_sha256") or resolved.get("binary_sha256_expected"),
        "files": resolved.get("digest_checks", []),
        "argv": resolved.get("argv"),
    }
    write_json(cell_dir / "config-evidence.json", document)


def run_cell(name: str, plan: dict, out: Path, resolved: dict) -> dict:
    cell_dir = out / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"status": None, "evidence": plan.get("evidence"), "start_unix": time.time(),
                              "argv": plan.get("argv"), "skip_reason": plan.get("skip_reason")}
    if plan.get("skip_reason"):
        record["status"] = "SKIPPED"
        record["end_unix"] = time.time()
        return record
    log_path = cell_dir / "log.txt"
    if name == "accuracy":
        record["argv"] = plan["suites"]
        write_config_evidence(cell_dir, resolved)
        exits = {}
        wall = 0.0
        for suite, argv in plan["suites"].items():
            code, seconds = run_subprocess(argv, log_path)
            exits[suite] = code
            wall += seconds
        record["exit_code"] = exits
        record["wall_s"] = wall
        failed = [s for s, c in exits.items() if c != 0]
    else:
        code, wall = run_subprocess(plan["argv"], log_path)
        record["exit_code"] = code
        record["wall_s"] = wall
        failed = [name] if code != 0 else []
    try:
        record["result"] = parse_cell_evidence(name, cell_dir, plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["result"] = None
        record["parse_error"] = str(error)
    record["end_unix"] = time.time()
    if failed or record.get("result") is None:
        record["status"] = "FAIL"
        if failed:
            record["status_reason"] = f"script exit {record['exit_code']}"
        else:
            record["status_reason"] = record.get("parse_error") or "no evidence file written"
        record["log_tail"] = tail_text(log_path)
    else:
        record["status"] = "OK"
    # A gate script that exits 1 with a FAIL verdict still leaves parseable evidence;
    # keep the verdict row visible but mark the cell failed.
    return record


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, help="<model>/<file> under configs/profiles/")
    parser.add_argument("--host", help="explicit host file for the resolver (default: auto-select)")
    parser.add_argument("--port", type=int, default=8015, help="dev port to serve/measure on (default 8015)")
    parser.add_argument("--baseline", help="profile id whose status.evidence holds a prior kit bundle")
    parser.add_argument("--baseline-results", help="directory (or summary.json) of any prior kit bundle")
    parser.add_argument("--out", help="bundle directory (default results/<model>-gates/qualify-<date>)")
    parser.add_argument("--cells", help=f"comma list; default {','.join(DEFAULT_CELLS)}")
    parser.add_argument("--skip", help="comma list of cells to skip")
    parser.add_argument("--with-soak", action="store_true", help="also run the 30-minute soak cell")
    parser.add_argument("--no-launch", action="store_true", help="a server is already up on --port; do not start/stop")
    parser.add_argument("--resummarize", action="store_true",
                        help="run no cell and launch nothing; rebuild summary.json/SUMMARY.md of --out from its manifest and evidence")
    parser.add_argument("--served-model", help="override the served model name derived from the profile argv")
    parser.add_argument("--dry-run", action="store_true", help="resolve, plan, write manifest.json; launch nothing")
    parser.add_argument("--reasoning-effort", default="low", help="accuracy cell --reasoning-effort (default low; '' to omit)")
    parser.add_argument("--accuracy-split", default="holdout", choices=("dev", "holdout"),
                        help="accuracy split (default holdout; docs/PROFILE-SCHEMA.md keeps holdout owner-run)")
    parser.add_argument("--encoder", help="chat encoder name for the accuracy cell (overrides the model map)")
    parser.add_argument("--targets", help="JSON file with qualification_targets (overrides the profile block)")
    parser.add_argument("--reference-logits-dir", help="teacher dataset dir (overrides model.json reference_logits)")
    parser.add_argument("--teacher-windows", help="49_score_teacher_windows.py --windows, e.g. 0-24")
    parser.add_argument("--vision-limit", type=int, help="38_bench_vision.py --limit")
    parser.add_argument("--vision-thinking-mode", choices=("chat", "thinking"),
                        help="38_bench_vision.py --thinking-mode (overrides the profile's qualification_options; default chat)")
    args = parser.parse_args(argv)
    if args.baseline and args.baseline_results:
        parser.error("--baseline and --baseline-results are mutually exclusive")
    if args.resummarize:
        if not args.out:
            parser.error("--resummarize needs --out (the bundle to rebuild)")
        args.no_launch = True
    if args.reasoning_effort == "":
        args.reasoning_effort = None
    return args


def load_previous_bundle(out: Path, cells: list[str]) -> dict[str, dict]:
    """Cells an earlier run of this bundle recorded and this run does not re-run.

    A re-run of one cell (`--cells vision --out <same bundle>`) keeps the other
    cells' records and rows; their evidence is re-parsed from disk, the status
    comes from the previous manifest. Returns {cell: {"manifest": ..., "record": ...}}.
    """
    path = out / "manifest.json"
    if not path.is_file():
        return {}
    try:
        old = read_json(path)
    except (OSError, ValueError):
        return {}
    kept: dict[str, dict] = {}
    for name, entry in (old.get("cells") or {}).items():
        if name in cells or name not in CELL_ORDER or not isinstance(entry, dict) or entry.get("status") is None:
            continue
        plan = {"argv": entry.get("argv"), "skip_reason": entry.get("skip_reason"), "evidence": entry.get("evidence")}
        if name == "accuracy":
            plan["suites"] = entry.get("argv") or {}
        record = {k: entry.get(k) for k in ("status", "exit_code", "wall_s", "start_unix", "end_unix")}
        record["evidence"] = entry.get("evidence")
        if record["status"] == "FAIL":
            record["status_reason"] = f"script exit {entry.get('exit_code')} in the previous run of this bundle"
        if entry.get("skip_reason"):
            record["status"] = "SKIPPED"
            record["status_reason"] = entry["skip_reason"]
        else:
            record["result"] = parse_cell_evidence(name, out / name, plan)
            if record["result"] is None and record["status"] == "OK":
                record["status"], record["status_reason"] = "FAIL", "evidence missing on re-summarize"
        kept[name] = {"manifest": entry, "record": record}
    # Evidence on disk from a run whose manifest was overwritten (an older kit
    # re-summarized only its selected cells): keep it, marked as recovered.
    evidence_paths = {"speed": "speed/speed.json", "toolcall": "toolcall/toolcall.json", "vision": "vision/summary.json",
                      "media": "media/summary.json", "teacher": "teacher/summary.json",
                      "context": "context/summary.json", "soak": "soak/soak.json"}
    for name, rel in evidence_paths.items():
        if name in kept or name in cells or not (out / rel).is_file():
            continue
        plan = {"argv": None, "skip_reason": None, "evidence": rel}
        try:
            result = parse_cell_evidence(name, out / name, plan)
        except (OSError, ValueError, KeyError, TypeError):
            result = None
        if result is None:
            continue
        record = {"status": "OK", "status_reason": "recovered from evidence on disk (not in the previous manifest)",
                  "exit_code": None, "wall_s": None, "evidence": rel, "result": result}
        kept[name] = {"manifest": {"argv": None, "skip_reason": None, "evidence": rel, "status": "OK",
                                   "recovered_from_evidence": True}, "record": record}
    return kept


def default_out(model_slug: str) -> Path:
    date = _dt.date.today().isoformat()
    slug = {"qwen3.8-27b": "qwen38", "glm-5.3-flash": "glm53-flash", "deepseek-v4-flash": "dsv4",
            "glm-5.2": "glm52", "laguna-s-2.1": "laguna"}.get(model_slug, model_slug)
    return REPO_ROOT / "results" / f"{slug}-gates" / f"qualify-{date}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_unix = time.time()
    try:
        resolved = resolve_profile(args)
    except KitError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if not args.dry_run and resolved["check"]["exit_code"] != 0:
        print(f"ERROR: profile check failed: {resolved['check']['stderr'] or resolved['check']['stdout']}", file=sys.stderr)
        return 2

    out = Path(args.out).resolve() if args.out else default_out(resolved["model_slug"])
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        cells = select_cells(args)
    except KitError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    plans = plan_cells(resolved, args, out, base_url)
    previous = load_previous_bundle(out, cells) if not args.dry_run else {}

    manifest: dict[str, Any] = {
        "kind": "qualify-profile-manifest",
        "schema_version": 1,
        "kit": {"script": "scripts/94_qualify_profile.py", "sha256": sha256_file(Path(__file__)),
                "argv": sys.argv if argv is None else argv},
        "dry_run": bool(args.dry_run),
        "no_launch": bool(args.no_launch),
        "start_unix": start_unix,
        "start_iso": utc_now(),
        "profile": resolved,
        "port": args.port,
        "base_url": base_url,
        "out": str(out),
        "host": host_facts(),
        "git": git_head(),
        "scripts": {name: sha256_file(SCRIPTS / name) for name in COMPOSED_SCRIPTS},
        "cells_selected": cells,
        "cells_kept_from_previous_run": sorted(previous),
        "cells": {**{name: entry["manifest"] for name, entry in previous.items()},
                  **{name: {"argv": plans[name].get("argv") if name != "accuracy" else plans[name].get("suites"),
                            "skip_reason": plans[name].get("skip_reason"), "evidence": plans[name].get("evidence")}
                     for name in cells}},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)

    if args.dry_run:
        print(f"dry-run: {resolved['profile_id']} -> {out}")
        print(f"  served model: {resolved['served_model']}  stack label: {resolved['stack_label']}  port: {args.port}")
        print(f"  slots x tokens/slot: {resolved['slots']} x {resolved['tokens_per_slot']} (cap {resolved['context_cap']}); "
              f"kill floor {resolved['kill_floor_gib']} GiB")
        print(f"  tokenizer: {resolved['tokenizer_path']} sha256={resolved['tokenizer_sha256'] or 'absent'}")
        print(f"  check: exit {resolved['check']['exit_code']}")
        print(f"  targets: {'declared' if resolved['qualification_targets'] else 'none'}; encoder: {resolved['encoder']}; vision mode: {resolved['vision_thinking_mode']}")
        for name in cells:
            plan = plans[name]
            if plan.get("skip_reason"):
                print(f"  [{name}] SKIPPED: {plan['skip_reason']}")
            elif name == "accuracy":
                for suite, suite_argv in plan["suites"].items():
                    print(f"  [{name}/{suite}] {' '.join(suite_argv)}")
            else:
                print(f"  [{name}] {' '.join(plan['argv'])}")
        print(f"manifest: {out / 'manifest.json'}")
        return 0

    baseline = load_baseline(args)
    records: dict[str, dict] = {}
    server_log = out / "server" / "lifecycle.txt"
    launched = False
    interrupted = False
    try:
        if not args.no_launch:
            proc = serve("start", args, server_log)
            launched = True  # a failed start may still have left state behind; stop is idempotent enough
            if proc.returncode != 0:
                raise KitError(f"93_profile_serve.sh start failed (exit {proc.returncode}); see {server_log}")
        if cells:
            ids = wait_for_model(base_url, resolved["served_model"], float(resolved.get("startup_timeout_seconds") or 1800))
            manifest["served_models_listed"] = ids
        write_json(out / "manifest.json", manifest)
        for name in cells:
            print(f"[{name}] starting", flush=True)
            records[name] = run_cell(name, plans[name], out, resolved)
            print(f"[{name}] {records[name]['status']}", flush=True)
            manifest["cells"][name].update({k: records[name].get(k) for k in ("status", "exit_code", "wall_s",
                                                                              "start_unix", "end_unix")})
            write_json(out / "manifest.json", manifest)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; stopping the profile", file=sys.stderr)
    except KitError as error:
        records.setdefault("_launch", {"status": "FAIL", "status_reason": str(error)})
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        if launched:
            try:
                serve("stop", args, server_log)
            except (OSError, subprocess.SubprocessError) as error:
                print(f"WARNING: stop failed: {error}", file=sys.stderr)
        manifest["end_unix"] = time.time()
        manifest["end_iso"] = utc_now()
        write_json(out / "manifest.json", manifest)

    for name in cells:
        records.setdefault(name, {"status": "FAIL", "status_reason": "not run (aborted)", "evidence": plans[name].get("evidence")})
    for name, entry in previous.items():
        records[name] = entry["record"]
    all_cells = [name for name in CELL_ORDER if name in records]
    summary = finalize_summary(resolved, records, all_cells, out, baseline, manifest, interrupted)
    # Render before writing anything so a renderer fault can never leave a
    # fresh summary.json beside a stale SUMMARY.md from an earlier run.
    summary_md = render_summary_md(summary)
    write_json(out / "summary.json", summary)
    (out / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    print(f"{summary['verdict']}: {out / 'SUMMARY.md'}")
    return 0 if summary["verdict"] in ("PASS", "MEASURED") else 1


def finalize_summary(resolved: dict, records: dict[str, dict], cells: list[str], out: Path,
                     baseline: dict | None, manifest: dict, interrupted: bool = False) -> dict:
    targets = resolved.get("qualification_targets") or {}
    rows: list[dict] = []
    cell_records = {name: records[name] for name in cells}
    for name in cells:
        record = cell_records[name]
        if record["status"] == "SKIPPED":
            continue
        rows.extend(build_rows(name, record.get("result"), targets, baseline))
    launch_failure = records.get("_launch")
    verdict = overall_verdict(cell_records, rows)
    if launch_failure or interrupted:
        verdict = "FAIL"
    return {
        "kind": "qualify-profile",
        "schema_version": 1,
        "profile_id": resolved["profile_id"],
        "stack_label": resolved["stack_label"],
        "served_model": resolved["served_model"],
        "out": str(out),
        "start_unix": manifest.get("start_unix"),
        "end_unix": manifest.get("end_unix"),
        "start_iso": manifest.get("start_iso"),
        "end_iso": manifest.get("end_iso"),
        "targets": targets or None,
        "baseline_source": (baseline or {}).get("_source"),
        "baseline_error": (baseline or {}).get("_error"),
        "launch_failure": launch_failure,
        "interrupted": interrupted,
        "cells": cell_records,
        "rows": rows,
        "verdict": verdict,
        "verdict_rule": ("PASS iff every non-skipped cell with a target passes and no cell errored; "
                         "FAIL otherwise; MEASURED when no target is declared at all"),
    }


if __name__ == "__main__":
    sys.exit(main())
