#!/usr/bin/env python3
"""Inkling 2 Mini lever screen scorer (DIAGNOSTIC; the keep rule is PREREG.md's, fixed by its commit).

  screen_verdict.py <screen dir>                 score every arm, write summary.json, print the verdict
  screen_verdict.py --may-run <screen dir> <arm> exit 0 only if <arm> is due
  screen_verdict.py --effort-due <screen dir> <arm> exit 0 only if the screen is COMPLETE with <arm> as carrier
  screen_verdict.py --self-test                  synthetic cases; all must match

Each lever arm (a1-a5) is compared with a0 alone. The combined arm (a6, named
in arms.json with "levers") carries every kept lever and is scored against a0
too. Missing or malformed evidence makes an arm NO_RESULT, and its lever is
not kept.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

KERNEL_KILL = re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I)
SERVER_KILL = re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid")
MEMWATCH_KILL = re.compile(r"BREACH|EMERGENCY_STOP")
GIB_KIB = 1024 * 1024
SWAP_GROWTH_MAX_GIB = 0.25
FILL_MARGIN_GIB = 3.0      # a 28K screen fill vs the 1M direct fill: activations only (the arena is preallocated)
PROJECTED_LOW_MIN_GIB = 12.0
TOOLCALL_MIN = 19
BATCH_FLAGS = {"batch4096": 4096, "batch8192": 8192}
LEVER_LINES = {
    # lever: (server-log needle, must appear only when the lever is on)
    "max_ctas_pow2": "INKLING2_MAX_CTAS_POW2=1",
    "dspark7": "'method': 'dspark'",
    "moe_humming": "'moe_backend': 'humming'",
}


class Bad(Exception):
    """Missing or malformed evidence."""


def ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Bad(f"{path.name}: {exc}") from exc


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def arm_levers(arms: dict, arm: str) -> list[str]:
    entry = arms[arm]
    if "levers" in entry:
        return list(entry["levers"])
    return [] if entry["lever"] is None else [entry["lever"]]


def memory(arm_dir: Path) -> dict:
    driver = (arm_dir / "driver.log").read_text()
    m = re.search(r"^(\S+) \S+: launching in scope", driver, re.M)
    if not m:
        raise Bad("driver.log has no launching line")
    launch = ts(m.group(1))
    samples = []
    for line in (arm_dir / "meminfo.log").read_text().splitlines():
        a = re.search(r"MemAvailable:=(\d+)", line)
        s = re.search(r"SwapFree:=(\d+)", line)
        if not (a and s):
            continue
        samples.append((ts(line.split()[0]), int(a.group(1)) / GIB_KIB, int(s.group(1)) / GIB_KIB))
    before = [x for x in samples if x[0] <= launch]
    after = [x for x in samples if x[0] >= launch]
    if not before or len(after) < 10:
        raise Bad("meminfo.log does not cover the launch")
    base_avail, base_swap = before[-1][1], before[-1][2]
    low = min(x[1] for x in after)
    return {"launch_avail_gib": base_avail, "low_avail_gib": low, "peak_use_gib": base_avail - low,
            "swap_growth_gib": max(0.0, base_swap - min(x[2] for x in after)),
            "projected_1m_low_gib": low - FILL_MARGIN_GIB}


def score_arm(arm_dir: Path, levers: list[str]) -> dict:
    """Facts and safety for one arm; raises Bad on missing/malformed evidence."""
    out: dict = {"levers": levers, "unsafe": []}
    driver = (arm_dir / "driver.log").read_text() if (arm_dir / "driver.log").exists() else None
    if driver is None:
        raise Bad("driver.log missing")
    out["failed_attempt"] = "FAILED_ATTEMPT" in driver
    out["memory"] = memory(arm_dir)
    kernel = (arm_dir / "kernel.log").read_text()
    server = (arm_dir / "server.log").read_text()
    memwatch = (arm_dir / "memwatch.log").read_text()
    if "ARMED" not in memwatch:
        raise Bad("memwatch.log has no ARMED line")
    kills = [l for l in kernel.splitlines() if KERNEL_KILL.search(l)]
    kills += [l for l in server.splitlines() if SERVER_KILL.search(l)]
    kills += [l for l in memwatch.splitlines() if MEMWATCH_KILL.search(l)]
    out["kill_lines"] = kills[:5]
    mem = out["memory"]
    for cond, why in ((out["failed_attempt"], "FAILED_ATTEMPT"), (bool(kills), "kill line"),
                      (mem["swap_growth_gib"] > SWAP_GROWTH_MAX_GIB, "swap growth"),
                      (mem["projected_1m_low_gib"] < PROJECTED_LOW_MIN_GIB, "projected 1M low")):
        if cond:
            out["unsafe"].append(why)
    if out["failed_attempt"]:
        return out   # nothing after readiness is required
    c = re.search(r"KV concurrency ([0-9.]+)x at 262,144", driver)
    if not c:
        raise Bad("driver.log has no KV concurrency line")
    out["kv_concurrency"] = float(c.group(1))
    if out["kv_concurrency"] < 4.0:
        out["unsafe"].append("KV concurrency < 4.0")
    # The startup lines show exactly the arm's levers.
    nd = re.search(r"non-default args: (\{.*\})", server)
    if not nd:
        raise Bad("server.log has no non-default args line")
    batch = re.search(r"'max_num_batched_tokens': (\d+)", nd.group(1))
    want_batch = next((BATCH_FLAGS[l] for l in levers if l in BATCH_FLAGS), 2048)
    if not batch or int(batch.group(1)) != want_batch:
        raise Bad(f"max_num_batched_tokens {batch and batch.group(1)} != {want_batch}")
    for lever, needle in LEVER_LINES.items():
        if (needle in server) != (lever in levers):
            raise Bad(f"lever line {needle!r} {'missing' if lever in levers else 'present'}")
    if "INKLING2_MAX_IMAGE_SIDE=4096" not in server:
        raise Bad("patch 0004 startup line missing")
    probe = load_json(arm_dir / "probe.json")
    hashes = probe.get("greedy")
    if not (isinstance(hashes, list) and len(hashes) == 3 and all(isinstance(h, str) and h for h in hashes)):
        raise Bad("probe.json greedy hashes malformed")
    out["probe"] = hashes
    cold = load_json(arm_dir / "cold.json")
    rows = cold.get("rows") or []
    if len(rows) != 6 or not all(finite(r.get("prompt_tokens")) for r in rows) \
            or not finite(cold.get("aggregate_prefill_tok_per_s")):
        raise Bad("cold.json incomplete")
    out["cold_prefill"] = cold["aggregate_prefill_tok_per_s"]
    speed = load_json(arm_dir / "speed.json")
    cells = {c.get("ctx_tokens"): c for c in speed.get("cells") or []}
    if speed.get("suite_valid") is not True or set(cells) != {0, 28672}:
        raise Bad("speed.json invalid or missing a context")
    for ctx, cell in cells.items():
        if cell.get("valid") is not True or not finite(cell.get("median_decode")):
            raise Bad(f"speed cell {ctx} invalid")
    prefill = [r.get("prefill_tok_s") for r in cells[28672].get("reps") or [] if r.get("valid")]
    if len(prefill) < 3 or not all(finite(p) for p in prefill):
        raise Bad("prefill reps at 28,672 missing")
    out["decode0"] = cells[0]["median_decode"]
    out["decode28"] = cells[28672]["median_decode"]
    out["prefill28"] = statistics.median(prefill)
    out["ttft0"] = cells[0].get("median_ttft")
    code = load_json(arm_dir / "code.json")
    out["code_decode"] = code.get("median_decode_tok_s") if code.get("valid") is True else None
    tool = load_json(arm_dir / "toolcall.json")
    if not isinstance(tool.get("passed"), int) or tool.get("total") != 20:
        raise Bad("toolcall.json malformed")
    out["toolcall"] = tool["passed"]
    if tool["passed"] < TOOLCALL_MIN:
        out["unsafe"].append("tool-call < 19/20")
    if "scope empty after stop" not in driver:
        out["unsafe"].append("scope not empty after stop")
    return out


def decide(lever: str, a: dict, base: dict) -> tuple[bool, str]:
    """PREREG keep rule for one lever arm against a0."""
    if a["unsafe"]:
        return False, "unsafe: " + ", ".join(a["unsafe"])
    r0, r28 = a["decode0"] / base["decode0"], a["decode28"] / base["decode28"]
    rp, rc = a["prefill28"] / base["prefill28"], a["cold_prefill"] / base["cold_prefill"]
    geo, worst = math.sqrt(r0 * r28), min(r0, r28)
    same = a["probe"] == base["probe"]
    rt0 = (a["ttft0"] / base["ttft0"]) if finite(a.get("ttft0")) and finite(base.get("ttft0")) else None
    rcode = (a["code_decode"] / base["code_decode"]) if finite(a.get("code_decode")) and finite(base.get("code_decode")) else None
    a["ratios"] = {"r0": r0, "r28": r28, "rp": rp, "rc": rc, "rt0": rt0, "rcode": rcode, "geo": geo,
                   "probe_identical": same}
    if lever in BATCH_FLAGS:
        ok = rp >= 1.03 and rc >= 1.0 and worst >= 0.98
        return ok, f"rp {rp:.4f} (>=1.03), rc {rc:.4f} (>=1.0), min decode {worst:.4f} (>=0.98)"
    if lever == "max_ctas_pow2":
        if not same:
            return False, "greedy probes differ from a0 (must be byte-identical)"
        ok = (rc >= 1.03 or (rt0 is not None and rt0 <= 0.97)) and worst >= 0.98 and rp >= 0.97
        return ok, f"rc {rc:.4f} (>=1.03) or rt0 {rt0} (<=0.97); min decode {worst:.4f}; rp {rp:.4f}"
    if lever == "dspark7":
        prose = geo >= 1.03 and worst >= 0.98 and rp >= 0.95
        code = rcode is not None and rcode >= 1.25 and worst >= 0.98 and rp >= 0.95
        return prose or code, f"geo {geo:.4f} (>=1.03) or code {rcode} (>=1.25); min decode {worst:.4f}; rp {rp:.4f}"
    if lever == "moe_humming":
        if not same:
            return False, "greedy probes differ from a0 (not byte-identical; the fidelity gate is not spent)"
        ok = (geo >= 1.03 or rp >= 1.03) and worst >= 0.98 and rp >= 0.95
        return ok, f"geo {geo:.4f} or rp {rp:.4f} (>=1.03); min decode {worst:.4f}"
    raise ValueError(lever)


def score(ev: Path) -> dict:
    arms = load_json(ev / "arms.json")
    result: dict = {"arms": {}, "kept": [], "dropped": {}, "no_result": {}}
    try:
        base = score_arm(ev / "a0", [])
    except (Bad, OSError) as exc:
        return {**result, "verdict": "NO_RESULT", "why": f"a0: {exc}"}
    result["arms"]["a0"] = base
    if base["unsafe"]:
        return {**result, "verdict": "NO_RESULT", "why": "a0 unsafe: " + ", ".join(base["unsafe"])}
    for arm in sorted(a for a in arms if a != "a0" and "levers" not in arms[a]):
        lever = arms[arm]["lever"]
        if not (ev / arm).exists():
            result["no_result"][lever] = "not run"
            continue
        try:
            facts = score_arm(ev / arm, [lever])
        except (Bad, OSError) as exc:
            result["no_result"][lever] = str(exc)
            continue
        keep, why = decide(lever, facts, base)
        facts["keep"], facts["why"] = keep, why
        result["arms"][arm] = facts
        (result["kept"].append(lever) if keep else result["dropped"].__setitem__(lever, why))
    batches = [l for l in result["kept"] if l in BATCH_FLAGS]
    if len(batches) > 1:   # one batch size: the larger prefill gain
        best = max(batches, key=lambda l: next(f["ratios"]["rp"] for f in result["arms"].values()
                                                if f["levers"] == [l]))
        for l in batches:
            if l != best:
                result["kept"].remove(l)
                result["dropped"][l] = f"kept, but {best} gained more prefill"
    combined = [a for a in arms if "levers" in arms[a]]
    if len(result["kept"]) <= 1:
        result["carrier"] = next((a for a, f in result["arms"].items() if f["levers"] == result["kept"]), "a0")
        verdict = "COMPLETE"
    elif not combined:
        result["carrier"] = None
        verdict = "INCOMPLETE"   # the combined arm still has to be authored and run
    else:
        arm = combined[-1]
        if sorted(arms[arm]["levers"]) != sorted(result["kept"]):
            return {**result, "verdict": "NO_RESULT", "why": f"{arm} levers {arms[arm]['levers']} != kept {result['kept']}"}
        if not (ev / arm).exists():
            result["carrier"], verdict = None, "INCOMPLETE"
        else:
            try:
                facts = score_arm(ev / arm, arms[arm]["levers"])
            except (Bad, OSError) as exc:
                return {**result, "verdict": "NO_RESULT", "why": f"{arm}: {exc}"}
            ok = not facts["unsafe"]
            if ok:
                r0, r28 = facts["decode0"] / base["decode0"], facts["decode28"] / base["decode28"]
                rp = facts["prefill28"] / base["prefill28"]
                facts["ratios"] = {"r0": r0, "r28": r28, "rp": rp, "geo": math.sqrt(r0 * r28)}
                ok = math.sqrt(r0 * r28) >= 1.0 and rp >= 0.97
            facts["keep"] = ok
            result["arms"][arm] = facts
            if ok:
                result["carrier"] = arm
            else:   # fall back to the single kept lever with the best decode geomean
                best = max((a for a, f in result["arms"].items() if len(f["levers"]) == 1 and f.get("keep")),
                           key=lambda a: result["arms"][a]["ratios"]["geo"])
                result["carrier"] = best
            verdict = "COMPLETE"
    untested = [arms[a]["lever"] for a in arms if a != "a0" and "levers" not in arms[a]
                and arms[a]["lever"] in result["no_result"]]
    if untested and verdict == "COMPLETE":
        verdict = "INCOMPLETE"
    result["untested_levers"] = untested
    result["verdict"] = verdict
    return result


def may_run(ev: Path, arm: str) -> bool:
    arms = load_json(ev / "arms.json")
    if arm not in arms or (ev / arm).exists():
        return False
    if arm == "a0":
        return True
    if not (ev / "a0").exists():
        return False
    if "levers" in arms[arm]:
        result = score(ev)
        singles = [a for a in arms if a != "a0" and "levers" not in arms[a]]
        return all((ev / a).exists() for a in singles) and sorted(arms[arm]["levers"]) == sorted(result["kept"]) \
            and len(result["kept"]) > 1
    return True


def effort_due(ev: Path, arm: str) -> bool:
    if (ev / f"effort-{arm}").exists():
        return False
    result = score(ev)
    return result["verdict"] == "COMPLETE" and result.get("carrier") == arm


# --------------------------------------------------------------------------- self-test

def _write_arm(root: Path, arm: str, levers: list[str], **kw) -> None:
    d = root / arm
    d.mkdir(parents=True)
    batch = next((BATCH_FLAGS[l] for l in levers if l in BATCH_FLAGS), 2048)
    driver = ["2026-11-23T05:00:00Z arm launching", f"2026-11-23T05:00:01Z {arm}: launching in scope u"]
    if kw.get("failed"):
        driver.append(f"2026-11-23T05:03:00Z FAILED_ATTEMPT {arm}: launch did not reach ready")
    else:
        driver.append(f"2026-11-23T05:04:00Z {arm}: KV concurrency {kw.get('conc', 4.07)}x at 262,144 per request")
        driver.append(f"2026-11-23T05:30:00Z {arm}: scope empty after stop")
    (d / "driver.log").write_text("\n".join(driver) + "\n")
    lines = []
    low = kw.get("low", 25.0)
    for i in range(40):
        avail = 115.5 if i < 2 else (low if i == 20 else 40.0)
        swap = 16.0 - (kw.get("swap", 0.0) if i == 30 else 0.0)
        lines.append(f"2026-11-23T05:00:{i:02d}Z MemAvailable:={int(avail * GIB_KIB)} SwapFree:={int(swap * GIB_KIB)} ")
    (d / "meminfo.log").write_text("\n".join(lines) + "\n")
    (d / "kernel.log").write_text(kw.get("kernel", ""))
    (d / "memwatch.log").write_text("ts=x ARMED target_pid=1\n" + kw.get("memwatch", ""))
    nd = {"max_num_batched_tokens": batch}
    if "moe_humming" in levers:
        nd["moe_backend"] = "humming"
    if "dspark7" in levers:
        nd["speculative_config"] = {"method": "dspark"}
    server = [f"INFO non-default args: {nd!r}", "INFO INKLING2_MAX_IMAGE_SIDE=4096: encoder budget"]
    if "max_ctas_pow2" in levers:
        server.append("INFO INKLING2_MAX_CTAS_POW2=1: max_ctas bucketed")
    if kw.get("drop_line"):
        server = [l for l in server if kw["drop_line"] not in l]
    (d / "server.log").write_text("\n".join(server) + "\n" + kw.get("server", ""))
    if kw.get("failed"):
        return
    s = kw.get("s", {})
    (d / "probe.json").write_text(json.dumps({"greedy": kw.get("probe", ["a", "b", "c"])}))
    if not kw.get("no_cold"):
        (d / "cold.json").write_text(json.dumps({"rows": [{"prompt_tokens": 1000}] * 6,
                                                 "aggregate_prefill_tok_per_s": 2000 * s.get("rc", 1.0)}))
    reps = [{"valid": True, "prefill_tok_s": 2700 * s.get("rp", 1.0)} for _ in range(3)]
    if kw.get("null_prefill"):
        reps[0]["prefill_tok_s"] = None
    (d / "speed.json").write_text(json.dumps({"suite_valid": True, "cells": [
        {"ctx_tokens": 0, "valid": True, "median_decode": 34.0 * s.get("r0", 1.0), "median_ttft": 0.5 * s.get("rt0", 1.0), "reps": reps},
        {"ctx_tokens": 28672, "valid": True, "median_decode": 29.0 * s.get("r28", 1.0), "median_ttft": 11.0, "reps": reps}]}))
    (d / "code.json").write_text(json.dumps({"valid": True, "median_decode_tok_s": 34.0 * s.get("rcode", 1.0)}))
    (d / "toolcall.json").write_text(json.dumps({"passed": kw.get("tool", 20), "total": 20}))


SINGLE = {"a1": "batch4096", "a2": "batch8192", "a3": "max_ctas_pow2", "a4": "dspark7", "a5": "moe_humming"}


def _screen(root: Path, arms_extra=None, **per_arm) -> Path:
    arms = {"a0": {"profile": "p0", "lever": None, "base": None}}
    arms.update({a: {"profile": f"p{a}", "lever": l, "base": "a0"} for a, l in SINGLE.items()})
    arms.update(arms_extra or {})
    root.mkdir()
    (root / "arms.json").write_text(json.dumps(arms))
    _write_arm(root, "a0", [], **per_arm.get("a0", {}))
    for a, l in SINGLE.items():
        if per_arm.get(a) is not None and per_arm[a].get("skip"):
            continue
        _write_arm(root, a, [l], **per_arm.get(a, {}))
    for a, e in (arms_extra or {}).items():
        if per_arm.get(a, {}).get("skip"):
            continue
        _write_arm(root, a, e["levers"], **per_arm.get(a, {}))
    return root


def self_test() -> int:
    gain = {"s": {"r0": 1.05, "r28": 1.05, "rp": 1.05, "rc": 1.05}}
    cases = []

    def case(name, check, **kw):
        cases.append((name, check, kw))

    case("nothing gains: COMPLETE, a0 carrier, nothing kept",
         lambda r: r["verdict"] == "COMPLETE" and r["kept"] == [] and r["carrier"] == "a0")
    case("one lever gains: carrier is that arm",
         lambda r: r["kept"] == ["dspark7"] and r["carrier"] == "a4", a4=gain)
    case("two levers kept without a combined arm: INCOMPLETE",
         lambda r: r["verdict"] == "INCOMPLETE" and set(r["kept"]) == {"dspark7", "max_ctas_pow2"},
         a3={"s": {"rc": 1.05}}, a4=gain)
    case("combined arm scored: carrier a6",
         lambda r: r["verdict"] == "COMPLETE" and r["carrier"] == "a6",
         extra={"a6": {"profile": "p6", "levers": ["max_ctas_pow2", "dspark7"]}},
         a3={"s": {"rc": 1.05}}, a4=gain, a6=gain)
    case("combined arm slower than a0: best single lever carries",
         lambda r: r["carrier"] == "a4",
         extra={"a6": {"profile": "p6", "levers": ["max_ctas_pow2", "dspark7"]}},
         a3={"s": {"rc": 1.05}}, a4=gain, a6={"s": {"r0": 0.9, "r28": 0.9}})
    case("combined arm levers disagree with decisions: NO_RESULT",
         lambda r: r["verdict"] == "NO_RESULT",
         extra={"a6": {"profile": "p6", "levers": ["moe_humming", "dspark7"]}},
         a3={"s": {"rc": 1.05}}, a4=gain, a6=gain)
    case("a0 unsafe: NO_RESULT", lambda r: r["verdict"] == "NO_RESULT", a0={"kernel": "NVRM: Xid 31\n"})
    case("a0 FAILED_ATTEMPT: NO_RESULT", lambda r: r["verdict"] == "NO_RESULT", a0={"failed": True})
    case("kill line drops a gaining lever", lambda r: "dspark7" in r["dropped"],
         a4={**gain, "kernel": "Out of memory: Killed process\n"})
    case("server OOM drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "server": "torch.OutOfMemoryError\n"})
    case("memwatch breach drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "memwatch": "BREACH\n"})
    case("swap growth drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "swap": 0.5})
    case("projected 1M low drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "low": 14.0})
    case("KV concurrency < 4 drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "conc": 3.9})
    case("tool-call 18/20 drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "tool": 18})
    case("readiness failure drops (no later files needed)", lambda r: "dspark7" in r["dropped"],
         a4={"failed": True})
    case("pow2 with different probes drops", lambda r: "max_ctas_pow2" in r["dropped"],
         a3={"s": {"rc": 1.2}, "probe": ["x", "b", "c"]})
    case("pow2 kept on TTFT@0 alone", lambda r: "max_ctas_pow2" in r["kept"], a3={"s": {"rt0": 0.9}})
    case("humming with different probes drops", lambda r: "moe_humming" in r["dropped"],
         a5={**gain, "probe": ["x", "b", "c"]})
    case("dspark kept on code alone", lambda r: "dspark7" in r["kept"], a4={"s": {"rcode": 1.5}})
    case("dspark prose regression drops even with code gain", lambda r: "dspark7" in r["dropped"],
         a4={"s": {"rcode": 2.0, "r28": 0.95}})
    case("batch needs 3% prefill", lambda r: "batch4096" in r["dropped"], a1={"s": {"rp": 1.02}})
    case("two batches kept: the larger prefill gain stays", lambda r: "batch8192" in r["kept"]
         and "batch4096" not in r["kept"], a1={"s": {"rp": 1.04}}, a2={"s": {"rp": 1.08}})
    case("missing cold.json: NO_RESULT lever, INCOMPLETE", lambda r: "dspark7" in r["no_result"]
         and r["verdict"] == "INCOMPLETE", a4={**gain, "no_cold": True})
    case("null prefill rep: NO_RESULT lever", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "null_prefill": True})
    case("missing lever line: NO_RESULT lever", lambda r: "max_ctas_pow2" in r["no_result"],
         a3={"s": {"rc": 1.05}, "drop_line": "INKLING2_MAX_CTAS_POW2"})
    case("missing patch 0004 line: NO_RESULT", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "drop_line": "INKLING2_MAX_IMAGE_SIDE"})
    case("arm not run: INCOMPLETE", lambda r: r["verdict"] == "INCOMPLETE" and "moe_humming" in r["untested_levers"],
         a5={"skip": True})

    failures = 0
    for name, check, kw in cases:
        tmp = Path(tempfile.mkdtemp(prefix="inkling2-screen-"))
        try:
            extra = kw.pop("extra", None)
            ev = _screen(tmp / "s", extra, **kw)
            result = score(ev)
            ok = bool(check(result))
        except Exception as exc:  # a crash is a failed case
            ok, result = False, {"crash": repr(exc)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": {json.dumps(result, default=str)[:400]}"))
    # --may-run
    tmp = Path(tempfile.mkdtemp(prefix="inkling2-screen-"))
    try:
        root = tmp / "m"
        root.mkdir()
        (root / "arms.json").write_text(json.dumps({"a0": {"profile": "p", "lever": None, "base": None},
                                                    "a4": {"profile": "q", "lever": "dspark7", "base": "a0"}}))
        checks = [may_run(root, "a0"), not may_run(root, "a4"), not may_run(root, "zz")]
        _write_arm(root, "a0", [])
        checks += [not may_run(root, "a0"), may_run(root, "a4")]
        _write_arm(root, "a4", ["dspark7"], s={"r0": 1.05, "r28": 1.05})
        checks += [effort_due(root, "a4"), not effort_due(root, "a0")]
        (root / "effort-a4").mkdir()
        checks += [not effort_due(root, "a4")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(checks)
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} --may-run and --effort-due admit only due runs {checks}")
    print(f"{len(cases) + 1 - failures}/{len(cases) + 1} self-test cases match")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return self_test()
    if argv[:1] == ["--may-run"] and len(argv) == 3:
        return 0 if may_run(Path(argv[1]), argv[2]) else 1
    if argv[:1] == ["--effort-due"] and len(argv) == 3:
        return 0 if effort_due(Path(argv[1]), argv[2]) else 1
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    ev = Path(argv[0])
    if self_test() != 0:
        print("self-test failed: refusing to score", file=sys.stderr)
        return 1
    result = score(ev)
    (ev / "summary.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({k: result.get(k) for k in ("verdict", "why", "kept", "dropped", "no_result", "carrier",
                                                 "untested_levers")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
