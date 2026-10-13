#!/usr/bin/env python3
"""Mutation check for gate_verdict.py: a clean synthetic bundle must PASS and
every broken or failing variant must give its expected verdict.

Usage: gate_mutations.py  -> prints one line per case, exit 0 iff all match.
Each case builds a throwaway git repository (the freeze: PREREG.md plus the
frozen manifests the scorer reads) and a bundle, then runs the scorer with
GATE_REPO and GATE_REFERENCE_PROBE pointing at them.
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCORER = Path(__file__).resolve().parent / "gate_verdict.py"
CANDIDATE = "results/mimo-v26-flash-gates/qualify-2026-10-13-c1"
GIB_KIB = 2**20
T0 = datetime(2026, 10, 14, tzinfo=timezone.utc)
REV, TREE = "1" * 40, "2" * 40


def stamp(seconds):
    return (T0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo(root: Path, mut: str) -> str:
    def git(*args):
        return subprocess.run(["git", "-C", str(root), "-c", "user.name=m", "-c", "user.email=m@m", *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    files = {
        f"{CANDIDATE}/PREREG.md": "frozen\n",
        f"{CANDIDATE}/gate_verdict.py": "v1\n",
        "configs/build-manifests/mimo-v26-flash-exl3-runtime-001.json": json.dumps(
            {"sources": {"tabbyAPI": {"revision": REV, "tree": TREE},
                         "exllamav3": {"revision": REV, "tree": TREE}}}),
        "weights/mimo-v2.6-flash/manifest.json": json.dumps({"artifacts": {
            "pack": {"files": [{"name": "a", "bytes": 100}, {"name": "b", "bytes": 200}]},
            "draft": {"files": [{"name": "c", "bytes": 50}]}}}),
        "configs/profiles/mimo-v2.6-flash/cuda-spark-128g-262k.json": json.dumps(
            {"artifact_roles": {"model": "pack", "draft_model": "draft"}}),
        "evidence.txt": "none\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "freeze")
    if mut in ("headevidence", "headfrozen"):
        target = f"{CANDIDATE}/gate_verdict.py" if mut == "headfrozen" else "evidence.txt"
        (root / target).write_text("changed\n")
        git("commit", "-qam", "after")
    return git("rev-parse", "HEAD")


def build(tmp: Path, mut: str) -> Path:
    head = repo(tmp / "repo", mut)
    ev = tmp / "ev"
    obs = ev / "observe"
    obs.mkdir(parents=True)
    n = 100 if mut == "short" else 800
    rows = []
    for i in range(n):
        if mut == "gap" and 300 <= i < 310:
            continue
        avail = 115.4 if i < 13 else 10.0
        if i == 400:
            avail = {"lowmem": 7.9, "peak": 9.2}.get(mut, 9.6)
        swap_free = 16 * GIB_KIB - (GIB_KIB // 2 if mut == "swap" and i > 200 else 0)
        rows.append(f"{stamp(i)} MemFree:={int(avail * GIB_KIB) - 100} MemAvailable:={int(avail * GIB_KIB)} "
                    f"SwapTotal:={16 * GIB_KIB} SwapFree:={swap_free} cg_anon=1 cg_file=1 \n")
    (obs / "meminfo.log").write_text("".join(rows))
    driver = [f"{stamp(0)} head {head if mut != 'headdiff' else 'b' * 40} dirty={1 if mut == 'dirty' else 0}"]
    if mut == "twoheads":
        driver.append(f"{stamp(1)} head {'c' * 40} dirty=0")
    driver += [
        f"runtime-src tabbyAPI {REV} {TREE} dirty=0 match=True",
        f"runtime-src exllamav3 {REV if mut != 'runtime' else '3' * 40} {TREE} dirty=0 "
        f"match={mut != 'runtime'}",
        f"runtime-wheel exllamav3 sha256_match=True files=600 mismatches={1 if mut == 'wheel' else 0}",
        f"weights model pack files=2 bytes=300 mismatches={1 if mut == 'weights' else 0}",
        f"weights draft_model draft files={2 if mut == 'weightcount' else 1} bytes=50 mismatches=0",
        f"{stamp(10)} MemFree gate pass 118000000 kB (3 samples)",
        f"{stamp(12)} launching in scope mimo26-c1-1",
    ]
    if mut == "attempt":
        driver.append(f"{stamp(20)} FAILED_ATTEMPT surviving descendants")
    driver.append(f"{stamp(790)} scope empty after stop")
    if mut != "nojournal":
        driver.append(f"{stamp(799)} kernel journal alive at exit")
    (obs / "driver.log").write_text("\n".join(driver) + "\n")
    (obs / "kernel.log").write_text("x NVRM: Xid 13\n" if mut == "xid" else "x drop_caches: 3\n")
    memwatch = "ts=1 START\n" if mut == "unarmed" else "ts=1 ARMED\n" + ("ts=2 BREACH\n" if mut == "breach" else "")
    (obs / "memwatch.log").write_text(memwatch)
    if mut != "noserverlog":
        server = "" if mut == "emptyserver" else "===== p session start t =====\n"
        (obs / "server.log").write_text(server + ("CUDA out of memory\n" if mut == "oom" else "ready\n"))

    budget = "256" if mut == "visionbudget" else "4096"
    (ev / "manifest.json").write_text(json.dumps({
        "git": {"head": "d" * 40 if mut == "kithead" else head},
        "profile": {"context_cap": 1_048_576 if mut == "cap" else 262_144},
        "cells": {"vision": {"argv": ["py", "38", "--thinking-mode", "chat", "--max-tokens", budget]}}}))
    (ev / "context").mkdir()
    (ev / "context" / "summary.json").write_text(json.dumps({
        "verdict": "FAIL" if mut == "context" else "PASS",
        "total_tokens": 240_000 if mut == "tokens" else 250_135, "max_ttft_s": 684.0,
        "min_mem_available_gib": 9.6}))
    greedy = [{"sha256": "abc"}, {"sha256": "abd"}, {"sha256": "abe"}]
    (tmp / "reference-probe.json").write_text(json.dumps({"greedy": greedy}))
    if mut != "noprobe":
        mine = [dict(g) for g in greedy]
        if mut == "identity":
            mine[2]["sha256"] = "zzz"
        (ev / "probe-c1.json").write_text(json.dumps({"greedy": mine}))
    (ev / "toolcall").mkdir()
    (ev / "toolcall" / "toolcall.json").write_text(json.dumps(
        {"passed": 18 if mut == "tool" else 20, "total": 19 if mut == "tooltotal" else 20}))
    tests = ["prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"]
    if mut == "agentmissing":
        tests = tests[:3]
    (ev / "agent-gate.json").write_text(json.dumps({
        "engine": "vllm", "model": "mimo-v2.6-flash", "spec_block_drop": False,
        "prefix_block_tokens": 2048 if mut == "agentslack" else 256,
        "results": [{"test": t, "pass": not (mut == "agent" and t == "turn-continuation")} for t in tests]}))
    (ev / "soak").mkdir()
    (ev / "soak" / "soak.json").write_text(json.dumps({
        "pass": mut != "soak", "mem_floor_gib": 12.0 if mut == "soakfloor" else 5.0,
        "mem_floor_source": "default", "mem_available_min_gib": 9.7}))
    (ev / "vision").mkdir()
    correct = {"vision52": 52, "vision53": 53}.get(mut, 60)
    (ev / "vision" / "summary.json").write_text(json.dumps({
        "n": 99 if mut == "visionn" else 100, "correct": correct, "invalid_count": 1,
        "error_count": 1 if mut == "visionerror" else 0}))
    (ev / "speed").mkdir()
    (ev / "speed" / "speed.json").write_text(json.dumps({
        "suite_valid": mut != "speed",
        "cells": [{"ctx_tokens": 0, "median_decode": 21.9, "median_ttft": 0.6, "valid": True},
                  {"ctx_tokens": 28672, "median_decode": 35.7, "median_ttft": 45.0, "valid": True}]}))
    return ev


EXPECT = {
    "clean": "PASS", "headevidence": "PASS", "vision53": "PASS",
    "xid": "FAIL", "breach": "FAIL", "oom": "FAIL", "swap": "FAIL", "lowmem": "FAIL", "peak": "FAIL",
    "context": "FAIL", "tokens": "FAIL", "identity": "FAIL", "tool": "FAIL", "agent": "FAIL", "soak": "FAIL",
    "vision52": "FAIL", "visionerror": "FAIL",
    "cap": "NO_RESULT", "noprobe": "NO_RESULT", "tooltotal": "NO_RESULT", "agentmissing": "NO_RESULT",
    "agentslack": "NO_RESULT", "soakfloor": "NO_RESULT", "visionn": "NO_RESULT", "visionbudget": "NO_RESULT",
    "speed": "NO_RESULT", "attempt": "NO_RESULT", "dirty": "NO_RESULT", "twoheads": "NO_RESULT",
    "headdiff": "NO_RESULT", "headfrozen": "NO_RESULT", "runtime": "NO_RESULT", "wheel": "NO_RESULT",
    "weights": "NO_RESULT", "weightcount": "NO_RESULT", "noserverlog": "NO_RESULT", "unarmed": "NO_RESULT",
    "emptyserver": "NO_RESULT", "gap": "NO_RESULT", "short": "NO_RESULT", "kithead": "NO_RESULT", "nojournal": "NO_RESULT",
}


def main() -> int:
    ok = True
    for mut, expected in EXPECT.items():
        with tempfile.TemporaryDirectory() as tmp:
            ev = build(Path(tmp), mut)
            env = dict(os.environ, GATE_REPO=str(Path(tmp) / "repo"),
                       GATE_REFERENCE_PROBE=str(Path(tmp) / "reference-probe.json"))
            proc = subprocess.run([sys.executable, str(SCORER), str(ev)], capture_output=True, text=True, env=env)
            try:
                summary = json.loads((ev / "gate-summary.json").read_text())
                got = summary["verdict"]
            except (OSError, ValueError, KeyError):
                summary, got = {}, f"CRASH rc={proc.returncode} {proc.stderr.strip()[-200:]}"
            rc_ok = (proc.returncode == 0) == (got == "PASS")
            match = got == expected and rc_ok
            ok &= match
            why = (summary.get("fail") or summary.get("missing") or [""])[0] if summary else ""
            print(f"{'ok ' if match else 'BAD'} {mut:13s} expected={expected:9s} got={got} rc={proc.returncode}"
                  + ("" if match and expected == "PASS" else f"  [{why[:110]}]"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
