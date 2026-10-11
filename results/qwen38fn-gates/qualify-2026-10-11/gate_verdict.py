#!/usr/bin/env python3
"""Fixed Phase-2 gate verdict per PREREG.md in this directory.

Usage: gate_verdict.py <bundle dir>  -> writes <dir>/gate-summary.json, prints the verdict.
Fails closed: missing or malformed evidence is NO_RESULT, never PASS.
The kill-line patterns are imported from the smoke scorer so both runs use
exactly the same ones.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SWAP_MAX_GIB, FLOOR_GIB, TOOLCALL_MIN = 0.25, 10.0, 19
MIN_CONTEXT_TOKENS, CONTEXT_CAP = 1_000_000, 1_048_576
STARTS = ("start1", "start2")
AGENT_GATE_TESTS = {"prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"}

_spec = importlib.util.spec_from_file_location(
    "smoke_verdict", Path(__file__).resolve().parent.parent / "smoke-2026-10-11" / "verdict.py")
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)
KILL = _smoke.KILL


def load(path, missing):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        missing.append(f"{path.name} ({type(error).__name__})")
        return None


def main(ev):
    missing, fail, checks = [], [], {}

    hits, failed_attempts, low = [], [], None
    for start in STARTS:
        obs = ev / "observe" / start
        for name, rx in KILL.items():
            files = [obs / "server.log"] if name == "server" else [obs / name]
            for f in files:
                if not f.is_file():
                    missing.append(f"observe/{start}/{f.name}")
                    continue
                hits += [f"{start}/{f.name}: {l.strip()}"
                         for l in f.read_text(errors="replace").splitlines() if rx.search(l)]
        rows = []
        meminfo = obs / "meminfo.log"
        for line in meminfo.read_text().splitlines() if meminfo.is_file() else []:
            kv = dict(re.findall(r"(\w+):=(\d+)", line))
            if {"MemAvailable", "SwapTotal", "SwapFree"} <= kv.keys():
                rows.append(kv)
        if len(rows) < 600:
            missing.append(f"observe/{start}/meminfo.log (fewer than 600 samples)")
        else:
            used = [(int(r["SwapTotal"]) - int(r["SwapFree"])) / 2**20 for r in rows]
            checks[f"swap_growth_gib_{start}"] = max(used) - used[0]
            start_low = min(int(r["MemAvailable"]) for r in rows) / 2**20
            checks[f"mem_available_low_gib_{start}"] = start_low
            low = start_low if low is None else min(low, start_low)
            if checks[f"swap_growth_gib_{start}"] > SWAP_MAX_GIB:
                fail.append(f"clause 2: swap growth in {start}")
        driver = obs / "driver.log"
        if driver.is_file():
            failed_attempts += [l for l in driver.read_text().splitlines() if "FAILED_ATTEMPT" in l]
        else:
            missing.append(f"observe/{start}/driver.log")
    checks["kill_lines"] = hits
    checks["failed_attempt_lines"] = failed_attempts
    checks["mem_available_low_gib"] = low
    if hits:
        fail.append("clause 1: kill/OOM/NVRM/Xid line")
    if low is not None and low < FLOOR_GIB:
        fail.append("clause 3: external low point below 10 GiB")

    manifest = load(ev / "manifest.json", missing)
    context = load(ev / "context" / "summary.json", missing)
    if manifest is not None and context is not None:
        cap = (manifest.get("profile") or {}).get("context_cap")
        checks["context"] = {"verdict": context.get("verdict"), "total_tokens": context.get("total_tokens"),
                             "min_mem_available_gib": context.get("min_mem_available_gib"),
                             "context_cap": cap}
        tokens = context.get("total_tokens")
        if cap != CONTEXT_CAP or not isinstance(tokens, int):
            missing.append("context: cap or total_tokens malformed")
        elif context.get("verdict") != "PASS" or tokens < MIN_CONTEXT_TOKENS:
            fail.append("clause 4: direct 1M fill")

    probes = [load(ev / f"probe-{start}.json", missing) for start in STARTS]
    if all(p is not None for p in probes):
        try:
            a, b = ([g["sha256"] for g in p["greedy"]] for p in probes)
        except (KeyError, TypeError):
            missing.append("probe greedy rows malformed")
        else:
            checks["restart_identity"] = [x == y for x, y in zip(a, b)]
            if len(a) != 3 or len(b) != 3:
                missing.append("probe: expected 3 greedy rows per start")
            elif a != b:
                fail.append("clause 5: restart greedy identity")

    toolcall = load(ev / "toolcall" / "toolcall.json", missing)
    if toolcall is not None:
        cases = toolcall.get("per_case") or []
        passed = toolcall["passed"] if toolcall.get("passed") is not None else sum(bool(c.get("passed")) for c in cases)
        total = toolcall["total"] if toolcall.get("total") is not None else len(cases)
        checks["toolcall"] = {"passed": passed, "total": total}
        if total != 20:
            missing.append(f"toolcall total {total} != 20")
        elif passed < TOOLCALL_MIN:
            fail.append("clause 6: tool-call")

    agent = load(ev / "agent-gate.json", missing)
    if agent is not None:
        results = {r.get("test"): r.get("pass") for r in agent.get("results") or []}
        checks["agent_gate"] = results
        if set(results) != AGENT_GATE_TESTS:
            missing.append(f"agent-gate tests {sorted(results)}")
        elif not all(v is True for v in results.values()):
            fail.append("clause 7: agent gate")

    soak = load(ev / "soak" / "soak.json", missing)
    if soak is not None:
        checks["soak_pass"] = soak.get("pass")
        if not isinstance(soak.get("pass"), bool):
            missing.append("soak.pass malformed")
        elif not soak["pass"]:
            fail.append("clause 8: soak")

    verdict = "FAIL" if fail else ("NO_RESULT" if missing or failed_attempts else "PASS")
    out = {"formula": "PREREG.md", "checks": checks, "fail": fail, "missing": missing, "verdict": verdict}
    (ev / "gate-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, "fail": fail, "missing": missing,
                      "mem_low_gib": low, "context": checks.get("context")}))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
