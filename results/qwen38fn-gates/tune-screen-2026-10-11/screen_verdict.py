#!/usr/bin/env python3
"""Fixed keep-rule scorer for the Flash-Next lossless lever screen (PREREG.md).

Usage: screen_verdict.py <screen dir>     writes <screen dir>/summary.json
       screen_verdict.py --self-test      synthetic cases; exit 0 only if all match
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
MEMINFO = re.compile(r"(\w+):=(\d+)")
KIB_PER_GIB = 1024 * 1024
MIN_KV = 1_048_576
MAX_PEAK_GIB = 106.0
L4_BASE_PEAK_GIB = 104.7
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


def measure(arm_dir: Path) -> dict:
    """One arm's measurements; raises Missing on absent or malformed evidence."""
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
                  if pattern.search(line)]
    server = _read(arm_dir / "server.log")

    speed = _json(arm_dir / "speed.json")
    cells = {str(c.get("ctx_tokens")): c for c in speed.get("cells", [])}
    if set(cells) != {"0", "28672"}:
        raise Missing(f"speed cells {sorted(cells)}")
    decode = {k: _finite(c.get("median_decode"), f"decode@{k}") for k, c in cells.items()}
    ttft0 = _finite(cells["0"].get("median_ttft"), "ttft@0")
    prefill_reps = [r.get("prefill_tok_s") for r in cells["28672"].get("reps", [])]
    prefill = _finite(statistics.median(prefill_reps) if prefill_reps else None, "prefill@28672")
    valid = all(bool(c.get("valid")) for c in cells.values()) and bool(speed.get("suite_valid"))

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
        "failed_attempt": failed, "kv_capacity": kv, "kills": kills,
        "baseline_gib": baseline / KIB_PER_GIB, "low_gib": low / KIB_PER_GIB,
        "peak_gib": (baseline - low) / KIB_PER_GIB, "swap_growth_gib": swap_growth,
        "decode_0": decode["0"], "decode_28672": decode["28672"], "prefill_28672": prefill,
        "ttft_0": ttft0, "speed_valid": valid, "toolcall": passed,
        "acceptance_length": (1 + accepted / drafts) if drafts > 0 else None,
        "fused_draft_fallback_logged": "Fused multi-step draft decode is not supported" in server,
        "probe_hashes_equal_c5": hashes == c5 and len(hashes) == 3,
    }


def safety_failures(m: dict) -> list[str]:
    out = []
    if m["failed_attempt"]:
        out.append("FAILED_ATTEMPT")
    if m["kills"]:
        out.append(f"kill lines ({len(m['kills'])})")
    if m["swap_growth_gib"] > MAX_SWAP_GIB:
        out.append(f"swap growth {m['swap_growth_gib']:.2f} GiB")
    if m["peak_gib"] > MAX_PEAK_GIB:
        out.append(f"peak {m['peak_gib']:.2f} GiB")
    if m["kv_capacity"] is None or m["kv_capacity"] < MIN_KV:
        out.append(f"KV capacity {m['kv_capacity']}")
    if m["toolcall"] < MIN_TOOLCALL:
        out.append(f"tool-call {m['toolcall']}/20")
    if not m["speed_valid"]:
        out.append("speed cell invalid")
    if m["ttft_0"] > MAX_TTFT0_S:
        out.append(f"TTFT@0 {m['ttft_0']:.3f} s")
    return out


def score(screen: Path) -> dict:
    arms = json.loads((screen / "arms.json").read_text())
    results: dict[str, dict] = {}
    state = {"last_kept": None}

    def finish(verdict: str | None, reason: str | None = None) -> dict:
        kept = [n for n, e in results.items() if e.get("decision") == "KEPT"]
        if verdict is None:
            if any(e.get("decision") == "NO_RESULT" for e in results.values()):
                verdict = "NO_RESULT"
            elif any(e.get("decision") == "NOT_RUN" for e in results.values()):
                verdict = "INCOMPLETE"
            else:
                verdict = "COMPLETE"
        out = {"verdict": verdict, "carrier_arm": state["last_kept"],
               "kept_levers": [arms[n]["lever"] for n in kept], "arms": results}
        if reason:
            out["reason"] = reason
        return out

    for name, spec in arms.items():
        arm_dir = screen / name
        ran = arm_dir.is_dir()
        entry: dict = {"profile": spec["profile"], "base": spec["base"], "lever": spec["lever"]}
        results[name] = entry
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
                continue
        if not ran:
            entry["decision"] = "NOT_RUN"
            continue
        try:
            m = measure(arm_dir)
        except Missing as error:
            entry["decision"] = "NO_RESULT"
            entry["reason"] = str(error)
            if spec["lever"] in (None, "L0"):
                return finish("NO_RESULT", f"{name}: {error}")
            continue
        entry["measured"] = m
        unsafe = safety_failures(m)
        entry["safety_failures"] = unsafe
        if spec["lever"] is None:
            if unsafe:
                entry["decision"] = "UNSAFE_BASELINE"
                return finish("NO_RESULT", f"baseline unsafe: {unsafe}")
            entry["decision"] = "BASELINE"
            state["last_kept"] = name
            continue
        base = results[spec["base"]]["measured"]
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


# --- self-test --------------------------------------------------------------

def _write_arm(root: Path, name: str, *, decode0=25.8, decode28=34.9, prefill=2680.0,
               peak_kib=100 * KIB_PER_GIB, swap_kib=0, kv=1079117, tool=20, kernel="",
               failed=False, drop=None, valid=True, ttft0=0.27) -> None:
    d = root / name
    d.mkdir(parents=True)
    base = 115 * KIB_PER_GIB
    driver = ["2026-10-12T00:00:00Z arm x", "2026-10-12T00:00:01Z x: launching in scope u",
              f"2026-10-12T00:10:00Z x: KV capacity {kv} tokens at 262,144 per request"]
    if failed:
        driver.append("2026-10-12T00:10:01Z FAILED_ATTEMPT x")
    files = {
        "driver.log": "\n".join(driver) + "\n",
        "meminfo.log": "".join(
            f"2026-10-12T00:00:0{i}Z MemAvailable:={v} SwapFree:={sf} \n"
            for i, (v, sf) in enumerate([(base, 16_000_000), (base, 16_000_000),
                                         (base - peak_kib, 16_000_000 - swap_kib)])),
        "kernel.log": kernel, "server.log": "ok\n", "memwatch.log": "ARMED\n",
        "speed.json": json.dumps({"suite_valid": valid, "cells": [
            {"ctx_tokens": 0, "median_decode": decode0, "median_ttft": ttft0, "valid": valid,
             "reps": [{"prefill_tok_s": 350.0}]},
            {"ctx_tokens": 28672, "median_decode": decode28, "median_ttft": 12.9, "valid": valid,
             "reps": [{"prefill_tok_s": prefill}] * 3}]}),
        "toolcall.json": json.dumps({"passed": tool, "total": 20}),
        "metrics-before.txt": "vllm:spec_decode_num_drafts_total{a=\"b\"} 10\n"
                              "vllm:spec_decode_num_accepted_tokens_total{a=\"b\"} 20\n",
        "metrics-after.txt": "vllm:spec_decode_num_drafts_total{a=\"b\"} 110\n"
                             "vllm:spec_decode_num_accepted_tokens_total{a=\"b\"} 220\n",
        "probe.json": C5_PROBE.read_text(),
    }
    for fname, text in files.items():
        if fname != drop:
            (d / fname).write_text(text)


def _screen(root: Path, arms: dict, runs: dict) -> dict:
    root.mkdir(parents=True)
    (root / "arms.json").write_text(json.dumps(arms))
    for name, kwargs in runs.items():
        _write_arm(root, name, **kwargs)
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


def self_test() -> int:
    cases = []

    def run(label, arms, runs, check):
        with tempfile.TemporaryDirectory() as tmp:
            result = _screen(Path(tmp) / "s", arms, runs)
        ok = check(result)
        cases.append((label, ok, result.get("verdict"), result.get("kept_levers")))

    all_runs = {f"a{i}": dict(FAST[i]) for i in range(6)}
    run("clean keeps every lever", ARMS, all_runs,
        lambda r: r["verdict"] == "COMPLETE" and r["kept_levers"] == ["L1", "L2", "L3", "L4"]
        and r["carrier_arm"] == "a5")
    for label, kwargs in [("kernel NVRM", {"kernel": "kernel: NVRM: Xid 31\n"}),
                          ("swap growth", {"swap_kib": int(0.5 * KIB_PER_GIB)}),
                          ("peak", {"peak_kib": int(106.5 * KIB_PER_GIB)}),
                          ("kv short", {"kv": 1_000_000}),
                          ("tool-call", {"tool": 18}),
                          ("failed attempt", {"failed": True}),
                          ("invalid speed", {"valid": False}),
                          ("ttft", {"ttft0": 0.6}),
                          ("too slow", {"decode0": 25.8 * 1.2, "decode28": 34.9 * 1.2})]:
        runs = {k: dict(v) for k, v in all_runs.items()}
        runs["a3"].update(kwargs)
        runs.pop("a4"), runs.pop("a5")
        arms = dict(ARMS)
        arms = {k: v for k, v in arms.items() if k not in ("a4", "a5")}
        run(f"{label} drops L2", arms, runs,
            lambda r: r["arms"]["a3"]["decision"] in ("DROPPED", "NO_RESULT") and r["kept_levers"] == ["L1"])
    run("a1 regression is NO_GO", ARMS, {"a0": FAST[0], "a1": dict(decode0=24.0, decode28=33.0)},
        lambda r: r["verdict"] == "NO_GO (runtime-002)")
    hot = {k: dict(v) for k, v in all_runs.items()}
    hot["a4"]["peak_kib"] = int(105.0 * KIB_PER_GIB)
    hot.pop("a5")
    run("a5 skipped above 104.7", ARMS, hot,
        lambda r: r["arms"]["a5"]["decision"] == "SKIPPED" and r["verdict"] == "COMPLETE")
    miss = {k: dict(v) for k, v in all_runs.items()}
    miss["a2"]["drop"] = "toolcall.json"
    run("missing file is NO_RESULT and not kept", ARMS, miss,
        lambda r: r["arms"]["a2"]["decision"] == "NO_RESULT" and "L1" not in r["kept_levers"]
        and r["verdict"] == "NO_RESULT")
    run("baseline missing file", ARMS, {"a0": dict(drop="speed.json")},
        lambda r: r["verdict"] == "NO_RESULT")
    bad = {k: dict(v) for k, v in all_runs.items()}
    bad["a2"].update(decode0=25.8, decode28=34.9)   # L1 dropped, a3 still built on a2
    run("arms.json base disagreeing with decisions", ARMS, bad,
        lambda r: r["verdict"] == "NO_RESULT" and "a3 ran on base a2" in r["reason"])
    amended = dict(ARMS)
    amended = {k: v for k, v in amended.items() if k in ("a0", "a1", "a2", "a3")}
    amended["a3b"] = {"profile": "p3b", "base": "a1", "lever": "L2"}
    runs = {"a0": FAST[0], "a1": FAST[1], "a2": dict(decode0=25.8, decode28=34.9), "a3b": FAST[2]}
    run("amended arm on the last kept base", amended, runs,
        lambda r: r["arms"]["a3"]["decision"] == "SUPERSEDED" and r["kept_levers"] == ["L2"])
    run("unrun arm leaves the screen incomplete", ARMS, {k: all_runs[k] for k in ("a0", "a1", "a2")},
        lambda r: r["verdict"] == "INCOMPLETE")

    for label, ok, verdict, kept in cases:
        print(f"{'ok ' if ok else 'BAD'} {label}: {verdict} {kept}")
    bad_cases = [c for c in cases if not c[1]]
    print(f"{len(cases) - len(bad_cases)}/{len(cases)} self-test cases match")
    return 0 if not bad_cases else 1


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    screen = Path(argv[1])
    result = score(screen)
    (screen / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
