#!/usr/bin/env python3
"""Fixed keep-rule scorer for the Flash-Next lossless lever screen (PREREG.md).

Usage: screen_verdict.py <screen dir>              writes <screen dir>/summary.json
       screen_verdict.py --may-run <screen dir> <arm>  exit 0 only if <arm> is the next arm due
       screen_verdict.py --self-test               synthetic cases; exit 0 only if all match
DIAGNOSTIC: the result chooses c10's levers and is never headline evidence.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

KERNEL_KILL = re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid")
SERVER_KILL = re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid")
MEMWATCH_KILL = re.compile(r"BREACH|EMERGENCY_STOP")
MARKERS = {"server.log": "session start", "memwatch.log": "ARMED"}
# The engine's arena guard refuses cleanly and names NVRM as what it prevents;
# its refusal is a FAILED_ATTEMPT, not a driver kill line (ATTEMPTS.md).
GUARD_REFUSAL = "refusing to allocate the KV arena"
ARGS_LINE = re.compile(r"non-default args: (.*)")
MEMINFO = re.compile(r"(\w+):=(\d+)")
KIB_PER_GIB = 1024 * 1024
MIN_KV = 1_048_576
MAX_PEAK_GIB = 106.0
L4_BASE_PEAK_GIB = 104.7
# Projected 1M peak = arm peak - a0 peak + candidate 9's measured 1M-fill peak
# (gate-summary.json peak_used_gib). c10's admission clause ceil(peak + 8) <= 115
# allows a 1M peak of at most 107 GiB.
C9_1M_PEAK_GIB = 105.6740493774414
MAX_PROJECTED_1M_PEAK_GIB = 107.0
MAX_SWAP_GIB = 0.25
MIN_TOOLCALL = 19
MAX_TTFT0_S = 0.5
C5_PROBE = Path(__file__).resolve().parent.parent / "qualify-2026-10-11-c4/probe-start1.json"


class Missing(Exception):
    """Evidence absent or malformed: the arm is NO_RESULT."""


def _ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> str:
    if not path.is_file():
        raise Missing(f"missing {path.name}")
    return path.read_text(errors="replace")


def _json(path: Path) -> dict:
    try:
        return json.loads(_read(path))
    except json.JSONDecodeError as error:
        raise Missing(f"malformed {path.name}: {error}") from None


def _finite(value, what: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise Missing(f"{what} is not a finite number: {value!r}")
    return float(value)


def _spec_counters(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in _read(path).splitlines():
        m = re.match(r"^(vllm:spec_decode_num_(?:drafts|accepted_tokens)_total)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            totals[m.group(1)] = totals.get(m.group(1), 0.0) + float(m.group(2))
    return totals


def lever_chain(arms: dict, name: str) -> list[str]:
    """Levers active in an arm: its base chain's levers plus its own."""
    chain, seen = [], set()
    while name is not None:
        if name in seen or name not in arms:
            raise Missing(f"arms.json base chain is broken at {name}")
        seen.add(name)
        if arms[name]["lever"] is not None:
            chain.append(arms[name]["lever"])
        name = arms[name]["base"]
    return chain[::-1]


def switch_problems(server: str, levers: list[str]) -> list[str]:
    """Startup lines must show exactly the arm's levers taking effect."""
    args = ARGS_LINE.search(server)
    if args is None:
        return ["no 'non-default args' line"]
    args = args.group(1)
    want = {
        "checkpoint_mapped PLE: QWEN38FN_PLE_WILLNEED=1": "L1" in levers,
        "checkpoint_mapped PLE: QWEN38FN_PLE_WILLNEED=0": "L0" in levers and "L1" not in levers,
        "Qwen4Exp MTP draft vocabulary: token ids [0, 65536)": "L2" in levers,
        "Qwen4Exp MTP draft vocabulary: full": "L0" in levers and "L2" not in levers,
        "Quantized a nvfp4 copy of the lm_head": "L2" in levers,
    }
    want_args = {
        "'draft_lm_head_quantization': 'nvfp4'": "L2" in levers,
        "'num_speculative_tokens_per_batch_size'": "L3" in levers,
        "'kv_cache_dtype_skip_layers'": "L4" in levers,
    }
    out = [f"{'missing' if w else 'unexpected'} startup line {t!r}"
           for t, w in want.items() if (t in server) != w]
    out += [f"{'missing' if w else 'unexpected'} non-default arg {t}"
            for t, w in want_args.items() if (t in args) != w]
    return out


def measure(arm_dir: Path, levers: list[str]) -> dict:
    """One arm's measurements; raises Missing on absent or malformed evidence.

    An arm whose driver logged FAILED_ATTEMPT (readiness or launch) is measured
    for memory and kill lines only; it is unsafe whatever else it lacks.
    """
    driver = _read(arm_dir / "driver.log")
    failed = [line for line in driver.splitlines() if "FAILED_ATTEMPT" in line]
    launch = re.search(r"^(\S+) \S+: launching in scope", driver, flags=re.M)
    if launch is None:
        raise Missing("no launching line")
    kv_m = re.search(r"KV capacity (\d+) tokens at 262,144", driver)
    kv = int(kv_m.group(1)) if kv_m else None

    samples = []
    for line in _read(arm_dir / "meminfo.log").splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        values = {k: int(v) for k, v in MEMINFO.findall(parts[1])}
        if "MemAvailable" in values and "SwapFree" in values:
            samples.append((_ts(parts[0]), values))
    t_launch = _ts(launch.group(1))
    before = [v for t, v in samples if t <= t_launch]
    after = [v for t, v in samples if t >= t_launch]
    if not before or not after:
        raise Missing("no memory samples around the launch")
    baseline = before[-1]["MemAvailable"]
    low = min(v["MemAvailable"] for v in after)
    swap_growth = (before[-1]["SwapFree"] - min(v["SwapFree"] for v in after)) / KIB_PER_GIB

    kills = []
    for name, pattern in (("kernel.log", KERNEL_KILL), ("server.log", SERVER_KILL),
                          ("memwatch.log", MEMWATCH_KILL)):
        kills += [f"{name}: {line[:200]}" for line in _read(arm_dir / name).splitlines()
                  if pattern.search(line) and not (name == "server.log" and GUARD_REFUSAL in line)]
    server = _read(arm_dir / "server.log")
    base = {
        "failed_attempt": failed, "kv_capacity": kv, "kills": kills,
        "baseline_gib": baseline / KIB_PER_GIB, "low_gib": low / KIB_PER_GIB,
        "peak_gib": (baseline - low) / KIB_PER_GIB, "swap_growth_gib": swap_growth,
        "fused_draft_fallback_logged": "Fused multi-step draft decode is not supported" in server,
    }
    if failed:
        return {**base, "decode_0": None, "decode_28672": None, "prefill_28672": None,
                "ttft_0": None, "speed_valid": False, "toolcall": None, "acceptance_length": None,
                "probe_hashes": None, "probe_hashes_equal_c5": None}
    for name, marker in MARKERS.items():
        if marker not in _read(arm_dir / name):
            raise Missing(f"{name} lacks {marker!r}")
    problems = switch_problems(server, levers)
    if problems:
        raise Missing("lever switches did not match the arm: " + "; ".join(problems))

    speed = _json(arm_dir / "speed.json")
    cells = {str(c.get("ctx_tokens")): c for c in speed.get("cells", [])}
    if set(cells) != {"0", "28672"}:
        raise Missing(f"speed cells {sorted(cells)}")
    decode = {k: _finite(c.get("median_decode"), f"decode@{k}") for k, c in cells.items()}
    ttft0 = _finite(cells["0"].get("median_ttft"), "ttft@0")
    prefill_reps = [r.get("prefill_tok_s") for r in cells["28672"].get("reps", [])]
    finite_reps = [v for v in prefill_reps if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and v > 0]
    valid = (all(bool(c.get("valid")) for c in cells.values()) and bool(speed.get("suite_valid"))
             and len(finite_reps) == len(prefill_reps) > 0)
    prefill = statistics.median(finite_reps) if valid else None

    tool = _json(arm_dir / "toolcall.json")
    passed, total = tool.get("passed"), tool.get("total")
    if not isinstance(passed, int) or total != 20:
        raise Missing(f"toolcall {passed}/{total}")

    b, a = _spec_counters(arm_dir / "metrics-before.txt"), _spec_counters(arm_dir / "metrics-after.txt")
    drafts = a.get("vllm:spec_decode_num_drafts_total", 0) - b.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = (a.get("vllm:spec_decode_num_accepted_tokens_total", 0)
                - b.get("vllm:spec_decode_num_accepted_tokens_total", 0))

    probe = _json(arm_dir / "probe.json")
    hashes = [g.get("sha256") for g in probe.get("greedy", [])]
    c5 = [g.get("sha256") for g in json.loads(C5_PROBE.read_text()).get("greedy", [])]

    return {
        **base,
        "decode_0": decode["0"], "decode_28672": decode["28672"], "prefill_28672": prefill,
        "ttft_0": ttft0, "speed_valid": valid, "toolcall": passed,
        "acceptance_length": (1 + accepted / drafts) if drafts > 0 else None,
        "probe_hashes": hashes, "probe_hashes_equal_c5": hashes == c5 and len(hashes) == 3,
    }


def void_check(screen: Path, arms: dict, name: str) -> tuple[bool, str]:
    """May this arm's attempt be voided and retried? (ATTEMPTS.md, five conditions)"""
    spec, arm_dir = arms[name], screen / name
    retry = spec.get("voided_by")
    if "retry_of" in spec:
        return False, "a retry arm cannot itself be voided"
    other = arms.get(retry)
    if other is None or other.get("retry_of") != name or "voided_by" in other or any(
            other.get(k) != spec.get(k) for k in ("profile", "base", "lever")):
        return False, f"no matching retry arm {retry!r}"
    if (arm_dir / "speed.json").exists():
        return False, "the attempt reached measurement"
    try:
        m = measure(arm_dir, [])
    except Missing as error:
        return False, f"attempt evidence unreadable: {error}"
    if not m["failed_attempt"]:
        return False, "no FAILED_ATTEMPT line"
    if m["kills"]:
        return False, f"kill lines: {m['kills'][:3]}"
    return True, f"readiness failure without kill lines; retried as {retry}"


def safety_failures(m: dict, a0_peak: float | None) -> list[str]:
    out = []
    if m["failed_attempt"]:
        out.append("FAILED_ATTEMPT")
    if m["kills"]:
        out.append(f"kill lines ({len(m['kills'])})")
    if m["swap_growth_gib"] > MAX_SWAP_GIB:
        out.append(f"swap growth {m['swap_growth_gib']:.2f} GiB")
    if m["peak_gib"] > MAX_PEAK_GIB:
        out.append(f"peak {m['peak_gib']:.2f} GiB")
    if a0_peak is not None and m["peak_gib"] - a0_peak + C9_1M_PEAK_GIB > MAX_PROJECTED_1M_PEAK_GIB:
        out.append(f"projected 1M peak {m['peak_gib'] - a0_peak + C9_1M_PEAK_GIB:.2f} GiB")
    if m["kv_capacity"] is None or m["kv_capacity"] < MIN_KV:
        out.append(f"KV capacity {m['kv_capacity']}")
    if m["failed_attempt"]:
        return out
    if m["toolcall"] < MIN_TOOLCALL:
        out.append(f"tool-call {m['toolcall']}/20")
    if not m["speed_valid"]:
        out.append("speed cell invalid")
    if m["ttft_0"] is None or m["ttft_0"] > MAX_TTFT0_S:
        out.append(f"TTFT@0 {m['ttft_0']} s")
    return out


def score(screen: Path) -> dict:
    arms = json.loads((screen / "arms.json").read_text())
    results: dict[str, dict] = {}
    state = {"last_kept": None, "a0_peak": None}

    def finish(verdict: str | None, reason: str | None = None) -> dict:
        kept = [n for n, e in results.items() if e.get("decision") == "KEPT"]
        decided = {e["lever"] for e in results.values()
                   if e.get("decision") not in (None, "SUPERSEDED", "VOID")}
        untested = sorted({e["lever"] for e in arms.values() if e["lever"] is not None} - decided)
        if verdict is None:
            if any(e.get("decision") == "NO_RESULT" for e in results.values()):
                verdict = "NO_RESULT"
            elif untested or any(e.get("decision") == "NOT_RUN" for e in results.values()):
                verdict = "INCOMPLETE"
            else:
                verdict = "COMPLETE"
        out = {"verdict": verdict,
               "carrier_arm": state["last_kept"] if verdict in ("COMPLETE", "INCOMPLETE") else None,
               "kept_levers": [arms[n]["lever"] for n in kept], "untested_levers": untested,
               "arms": results}
        if reason:
            out["reason"] = reason
        return out

    for name, spec in arms.items():
        arm_dir = screen / name
        ran = arm_dir.is_dir()
        entry: dict = {"profile": spec["profile"], "base": spec["base"], "lever": spec["lever"]}
        results[name] = entry
        if "voided_by" in spec:
            ok, why = void_check(screen, arms, name)
            if ok:
                entry.update(decision="VOID", reason=why)
                continue
            entry["void_refused"] = why
        if spec["base"] != state["last_kept"]:
            if ran:
                entry["decision"] = "INVALID_BASE"
                return finish("NO_RESULT", f"{name} ran on base {spec['base']}, "
                                           f"but the last kept arm is {state['last_kept']}")
            entry["decision"] = "SUPERSEDED"
            continue
        if spec["lever"] == "L4":
            base_peak = results[spec["base"]].get("measured", {}).get("peak_gib")
            if base_peak is None or base_peak > L4_BASE_PEAK_GIB:
                entry["decision"] = "SKIPPED"
                entry["reason"] = f"base peak {base_peak} GiB > {L4_BASE_PEAK_GIB}"
                if ran:
                    return finish("NO_RESULT", f"{name} ran although its skip rule applied")
                continue
        if not ran:
            entry["decision"] = "NOT_RUN"
            continue
        try:
            m = measure(arm_dir, lever_chain(arms, name))
        except Missing as error:
            entry["decision"] = "NO_RESULT"
            entry["reason"] = str(error)
            if spec["lever"] in (None, "L0"):
                return finish("NO_RESULT", f"{name}: {error}")
            continue
        entry["measured"] = m
        unsafe = safety_failures(m, state["a0_peak"])
        entry["safety_failures"] = unsafe
        if spec["lever"] is None:
            if unsafe:
                entry["decision"] = "UNSAFE_BASELINE"
                return finish("NO_RESULT", f"baseline unsafe: {unsafe}")
            entry["decision"] = "BASELINE"
            state["last_kept"], state["a0_peak"] = name, m["peak_gib"]
            continue
        base = results[spec["base"]]["measured"]
        entry["projected_1m_peak_gib"] = m["peak_gib"] - state["a0_peak"] + C9_1M_PEAK_GIB
        entry["probe_hashes_equal_base"] = (m["probe_hashes"] is not None
                                            and m["probe_hashes"] == base["probe_hashes"])
        if not unsafe:
            r0 = m["decode_0"] / base["decode_0"]
            r28 = m["decode_28672"] / base["decode_28672"]
            rp = m["prefill_28672"] / base["prefill_28672"]
            entry["ratios"] = {"decode_0": r0, "decode_28672": r28, "prefill_28672": rp,
                               "decode_geomean": math.sqrt(r0 * r28)}
        if spec["lever"] == "L0":
            ok = not unsafe and min(r0, r28) >= 0.97 and rp >= 0.95
            entry["decision"] = "PASS" if ok else "FAIL"
            if not ok:
                return finish("NO_GO (runtime-002)")
            state["last_kept"] = name
            continue
        keep = (not unsafe and math.sqrt(r0 * r28) >= 1.03 and min(r0, r28) >= 0.98 and rp >= 0.95)
        entry["decision"] = "KEPT" if keep else "DROPPED"
        if keep:
            state["last_kept"] = name
    return finish(None)


def may_run(screen: Path, arm: str) -> tuple[bool, str]:
    """The driver's preflight: only the next due arm may start."""
    arms = json.loads((screen / "arms.json").read_text())
    if arm not in arms:
        return False, f"{arm} is not in arms.json"
    if (screen / arm).exists():
        return False, f"{arm} already ran"
    result = score(screen)
    entry = result["arms"].get(arm)
    if result["verdict"] not in ("INCOMPLETE",) or entry is None or entry.get("decision") != "NOT_RUN":
        return False, (f"screen is {result['verdict']}; {arm} is "
                       f"{(entry or {}).get('decision', 'not reached')} {(entry or {}).get('reason', '')}")
    earlier = list(arms)[:list(arms).index(arm)]
    pending = [n for n in earlier if result["arms"].get(n, {}).get("decision") == "NOT_RUN"]
    if pending:
        return False, f"{arm} must wait for {pending}"
    return True, f"{arm} is due"


# --- self-test --------------------------------------------------------------

def _server_log(levers: list[str]) -> str:
    lines = ["===== p session start x ====="]
    args = "{'kv_cache_dtype': 'fp8', 'speculative_config': {'method': 'mtp'"
    if "L2" in levers:
        args += ", 'draft_lm_head_quantization': 'nvfp4'"
    if "L3" in levers:
        args += ", 'num_speculative_tokens_per_batch_size': [[1, 1, 4], [2, 4, 3]]"
    args += "}"
    if "L4" in levers:
        args += ", 'kv_cache_dtype_skip_layers': [48]"
    lines.append(f"INFO non-default args: {args}}}")
    if "L0" in levers:
        lines.append(f"INFO checkpoint_mapped PLE: QWEN38FN_PLE_WILLNEED={int('L1' in levers)} (x)")
        lines.append("INFO Qwen4Exp MTP draft vocabulary: "
                     + ("token ids [0, 65536)" if "L2" in levers else "full"))
    if "L2" in levers:
        lines.append("INFO Quantized a nvfp4 copy of the lm_head (65536 x 2048 per rank).")
    return "\n".join(lines) + "\n"


def _write_arm(root: Path, name: str, levers: list[str], *, decode0=25.8, decode28=34.9,
               prefill=2680.0, peak_kib=100 * KIB_PER_GIB, swap_kib=0, kv=1079117, tool=20,
               kernel="", failed=False, drop=(), valid=True, ttft0=0.27, server=None,
               memwatch="ARMED target_pid=1\n", prefill_reps=None) -> None:
    d = root / name
    d.mkdir(parents=True)
    base = 115 * KIB_PER_GIB
    driver = ["2026-10-12T00:00:00Z arm x", "2026-10-12T00:00:01Z x: launching in scope u",
              f"2026-10-12T00:10:00Z x: KV capacity {kv} tokens at 262,144 per request"]
    if failed:
        driver.append("2026-10-12T00:10:01Z FAILED_ATTEMPT x")
    reps = prefill_reps if prefill_reps is not None else [{"prefill_tok_s": prefill}] * 3
    files = {
        "driver.log": "\n".join(driver) + "\n",
        "meminfo.log": "".join(
            f"2026-10-12T00:00:0{i}Z MemAvailable:={v} SwapFree:={sf} \n"
            for i, (v, sf) in enumerate([(base, 16_000_000), (base, 16_000_000),
                                         (base - peak_kib, 16_000_000 - swap_kib)])),
        "kernel.log": kernel, "server.log": _server_log(levers) if server is None else server,
        "memwatch.log": memwatch,
        "speed.json": json.dumps({"suite_valid": valid, "cells": [
            {"ctx_tokens": 0, "median_decode": decode0, "median_ttft": ttft0, "valid": valid,
             "reps": [{"prefill_tok_s": 350.0}]},
            {"ctx_tokens": 28672, "median_decode": decode28, "median_ttft": 12.9, "valid": valid,
             "reps": reps}]}),
        "toolcall.json": json.dumps({"passed": tool, "total": 20}),
        "metrics-before.txt": "vllm:spec_decode_num_drafts_total{a=\"b\"} 10\n"
                              "vllm:spec_decode_num_accepted_tokens_total{a=\"b\"} 20\n",
        "metrics-after.txt": "vllm:spec_decode_num_drafts_total{a=\"b\"} 110\n"
                             "vllm:spec_decode_num_accepted_tokens_total{a=\"b\"} 220\n",
        "probe.json": C5_PROBE.read_text(),
    }
    drops = (drop,) if isinstance(drop, str) else drop
    for fname, text in files.items():
        if fname not in drops:
            (d / fname).write_text(text)


def _screen(root: Path, arms: dict, runs: dict) -> dict:
    root.mkdir(parents=True)
    (root / "arms.json").write_text(json.dumps(arms))
    for name, kwargs in runs.items():
        _write_arm(root, name, lever_chain(arms, name), **kwargs)
    return score(root)


ARMS = {
    "a0": {"profile": "p0", "base": None, "lever": None},
    "a1": {"profile": "p1", "base": "a0", "lever": "L0"},
    "a2": {"profile": "p2", "base": "a1", "lever": "L1"},
    "a3": {"profile": "p3", "base": "a2", "lever": "L2"},
    "a4": {"profile": "p4", "base": "a3", "lever": "L3"},
    "a5": {"profile": "p5", "base": "a4", "lever": "L4"},
}
FAST = [dict(decode0=25.8 * f, decode28=34.9 * f) for f in (1.0, 1.0, 1.2, 1.44, 1.73, 1.9)]
UNRUN = ("speed.json", "toolcall.json", "probe.json", "metrics-before.txt", "metrics-after.txt")


def self_test() -> int:
    cases = []

    def run(label, arms, runs, check):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = _screen(Path(tmp) / "s", arms, runs)
                ok = check(result)
            except Exception as error:  # a crash is a failed case, not a pass
                result, ok = {"verdict": f"crash {type(error).__name__}: {error}"}, False
        cases.append((label, ok, result.get("verdict"), result.get("kept_levers")))

    all_runs = {f"a{i}": dict(FAST[i]) for i in range(6)}
    run("clean keeps every lever", ARMS, all_runs,
        lambda r: r["verdict"] == "COMPLETE" and r["kept_levers"] == ["L1", "L2", "L3", "L4"]
        and r["carrier_arm"] == "a5" and r["untested_levers"] == [])
    for label, kwargs in [("kernel NVRM", {"kernel": "kernel: NVRM: Xid 31\n"}),
                          ("swap growth", {"swap_kib": int(0.5 * KIB_PER_GIB)}),
                          ("peak", {"peak_kib": int(106.5 * KIB_PER_GIB)}),
                          ("projected 1M peak", {"peak_kib": int(101.5 * KIB_PER_GIB)}),
                          ("kv short", {"kv": 1_000_000}),
                          ("tool-call", {"tool": 18}),
                          ("failed attempt", {"failed": True}),
                          ("readiness failure without later files", {"failed": True, "drop": UNRUN}),
                          ("invalid speed", {"valid": False}),
                          ("a null prefill rep", {"prefill_reps": [{"prefill_tok_s": 2680.0},
                                                                   {"prefill_tok_s": None}]}),
                          ("ttft", {"ttft0": 0.6}),
                          ("too slow", {"decode0": 25.8 * 1.2, "decode28": 34.9 * 1.2})]:
        runs = {k: dict(v) for k, v in all_runs.items() if k in ("a0", "a1", "a2", "a3")}
        runs["a3"].update(kwargs)
        arms = {k: v for k, v in ARMS.items() if k in ("a0", "a1", "a2", "a3")}
        run(f"{label} drops L2", arms, runs,
            lambda r: r["arms"]["a3"]["decision"] == "DROPPED" and r["kept_levers"] == ["L1"]
            and r["verdict"] == "COMPLETE")
    run("a1 regression is NO_GO", ARMS, {"a0": FAST[0], "a1": dict(decode0=24.0, decode28=33.0)},
        lambda r: r["verdict"] == "NO_GO (runtime-002)" and r["carrier_arm"] is None)
    run("a1 readiness failure is NO_GO", ARMS, {"a0": FAST[0], "a1": dict(failed=True, drop=UNRUN)},
        lambda r: r["verdict"] == "NO_GO (runtime-002)")
    run("unsafe baseline is NO_RESULT", ARMS, {"a0": dict(kernel="NVRM: Xid 13\n")},
        lambda r: r["verdict"] == "NO_RESULT" and r["carrier_arm"] is None)
    hot = {k: dict(v) for k, v in all_runs.items()}
    hot["a4"]["peak_kib"] = int(104.8 * KIB_PER_GIB)
    hot["a0"]["peak_kib"] = hot["a1"]["peak_kib"] = int(104.8 * KIB_PER_GIB)
    hot["a2"]["peak_kib"] = hot["a3"]["peak_kib"] = int(104.8 * KIB_PER_GIB)
    hot.pop("a5")
    run("a5 skipped above 104.7", ARMS, hot,
        lambda r: r["arms"]["a5"]["decision"] == "SKIPPED" and r["verdict"] == "COMPLETE")
    hot["a5"] = dict(FAST[5])
    run("a5 that ran despite its skip is NO_RESULT", ARMS, hot,
        lambda r: r["verdict"] == "NO_RESULT" and "skip rule" in r["reason"])
    miss = {k: dict(v) for k, v in all_runs.items()}
    miss["a2"]["drop"] = "toolcall.json"
    run("missing file is NO_RESULT and not kept", ARMS, miss,
        lambda r: r["arms"]["a2"]["decision"] == "NO_RESULT" and "L1" not in r["kept_levers"]
        and r["verdict"] == "NO_RESULT")
    run("baseline missing file", ARMS, {"a0": dict(drop="speed.json")},
        lambda r: r["verdict"] == "NO_RESULT")
    run("memwatch never armed", ARMS, {"a0": dict(memwatch="")},
        lambda r: r["verdict"] == "NO_RESULT" and "ARMED" in r["reason"])
    run("switch line absent", ARMS, {"a0": FAST[0], "a1": dict(server=_server_log([]))},
        lambda r: r["verdict"] == "NO_RESULT" and "QWEN38FN_PLE_WILLNEED=0" in r["reason"])
    leak = {k: dict(v) for k, v in all_runs.items() if k in ("a0", "a1", "a2", "a3")}
    leak["a2"]["server"] = _server_log(["L0", "L1", "L2"])
    run("a lever switch on in the wrong arm", {k: ARMS[k] for k in leak}, leak,
        lambda r: r["arms"]["a2"]["decision"] == "NO_RESULT" and "unexpected" in r["arms"]["a2"]["reason"])
    bad = {k: dict(v) for k, v in all_runs.items()}
    bad["a2"].update(decode0=25.8, decode28=34.9)   # L1 dropped, a3 still built on a2
    run("arms.json base disagreeing with decisions", ARMS, bad,
        lambda r: r["verdict"] == "NO_RESULT" and "a3 ran on base a2" in r["reason"])
    run("dropped lever without amended arms is INCOMPLETE", ARMS,
        {"a0": FAST[0], "a1": FAST[1], "a2": dict(decode0=25.8, decode28=34.9)},
        lambda r: r["verdict"] == "INCOMPLETE" and r["untested_levers"] == ["L2", "L3", "L4"])
    amended = {k: v for k, v in ARMS.items() if k in ("a0", "a1", "a2", "a3")}
    amended["a3b"] = {"profile": "p3b", "base": "a1", "lever": "L2"}
    runs = {"a0": FAST[0], "a1": FAST[1], "a2": dict(decode0=25.8, decode28=34.9), "a3b": FAST[2]}
    run("amended arm on the last kept base", amended, runs,
        lambda r: r["arms"]["a3"]["decision"] == "SUPERSEDED" and r["kept_levers"] == ["L2"]
        and r["verdict"] == "COMPLETE")
    run("unrun arm leaves the screen incomplete", ARMS, {k: all_runs[k] for k in ("a0", "a1", "a2")},
        lambda r: r["verdict"] == "INCOMPLETE")

    retry_arms = {"a0": ARMS["a0"], "a1": {**ARMS["a1"], "voided_by": "a1r"},
                  "a1r": {**ARMS["a1"], "retry_of": "a1"}, "a2": {**ARMS["a2"], "base": "a1r"},
                  "a3": ARMS["a3"], "a4": ARMS["a4"], "a5": ARMS["a5"]}
    refusal = _server_log(["L0"]) + ("RuntimeError: host page cache released under /m; MemFree 19.85 GiB: "
                                     f"{GUARD_REFUSAL}; page cache outside /m would leave the driver "
                                     "short of free memory (NVRM)\n")
    guard_fail = dict(failed=True, drop=UNRUN, server=refusal)
    retried = {"a0": FAST[0], "a1": guard_fail, **{f"a{i}": dict(FAST[i]) for i in range(2, 6)},
               "a1r": dict(FAST[1])}
    run("guard refusal voided and retried", retry_arms, retried,
        lambda r: r["verdict"] == "COMPLETE" and r["arms"]["a1"]["decision"] == "VOID"
        and r["arms"]["a1r"]["decision"] == "PASS" and r["kept_levers"] == ["L1", "L2", "L3", "L4"])
    nvrm = {**retried, "a1": {**guard_fail, "kernel": "NVRM: Xid 31\n"}}
    run("void refused for a kernel NVRM line", retry_arms, nvrm,
        lambda r: r["verdict"] == "NO_GO (runtime-002)" and "kill lines" in r["arms"]["a1"]["void_refused"])
    done = {**retried, "a1": dict(FAST[1])}
    run("void refused for a completed arm", retry_arms, done,
        lambda r: r["verdict"] == "NO_RESULT" and "measurement" in r["arms"]["a1"]["void_refused"])
    orphan = {"a0": ARMS["a0"], "a1": {**ARMS["a1"], "voided_by": "a1x"}, "a2": ARMS["a2"]}
    run("void refused without a retry arm", orphan, {"a0": FAST[0], "a1": guard_fail},
        lambda r: r["verdict"] == "NO_GO (runtime-002)" and "retry arm" in r["arms"]["a1"]["void_refused"])
    run("guard refusal alone is not a kill line", ARMS, {"a0": FAST[0], "a1": guard_fail},
        lambda r: r["arms"]["a1"]["measured"]["kills"] == [] and r["verdict"] == "NO_GO (runtime-002)")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "s"
        root.mkdir()
        (root / "arms.json").write_text(json.dumps(ARMS))
        gates = [may_run(root, "a0")[0], not may_run(root, "a1")[0]]
        _write_arm(root, "a0", [], **FAST[0])
        gates += [not may_run(root, "a0")[0], may_run(root, "a1")[0], not may_run(root, "a2")[0]]
        _write_arm(root, "a1", ["L0"], **FAST[1])
        _write_arm(root, "a2", ["L0", "L1"], decode0=25.8, decode28=34.9)
        gates.append(not may_run(root, "a3")[0])      # a3 is built on dropped a2
        cases.append(("may-run admits only the next due arm", all(gates), str(gates), None))

    for label, ok, verdict, kept in cases:
        print(f"{'ok ' if ok else 'BAD'} {label}: {verdict} {kept}")
    bad_cases = [c for c in cases if not c[1]]
    print(f"{len(cases) - len(bad_cases)}/{len(cases)} self-test cases match")
    return 0 if not bad_cases else 1


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) == 4 and argv[1] == "--may-run":
        ok, why = may_run(Path(argv[2]), argv[3])
        print(why, file=sys.stderr)
        return 0 if ok else 1
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    screen = Path(argv[1])
    result = score(screen)
    # The summary reports probe identity, not digests: 12-character prefixes
    # keep 64-hex values out of the evidence (secret lint).
    for entry in result["arms"].values():
        hashes = (entry.get("measured") or {}).get("probe_hashes")
        if hashes is not None:
            entry["measured"]["probe_hashes"] = [h[:12] if isinstance(h, str) else h for h in hashes]
    (screen / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
