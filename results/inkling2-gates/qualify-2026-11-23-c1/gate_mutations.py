#!/usr/bin/env python3
"""Mutation check for gate_verdict.py: a clean synthetic bundle must PASS and
every broken or failing variant must give its expected verdict, for the
expected reason where one is named (each case breaks one check alone).

Usage: gate_mutations.py  -> prints one line per case, exit 0 iff all match.
Each case builds a throwaway git repository (the freeze: PREREG.md plus the
frozen manifests the scorer reads) and a bundle, then runs the scorer with
--mutation and GATE_REPO and GATE_REFERENCE_PROBE pointing at them.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCORER = Path(__file__).resolve().parent / "gate_verdict.py"
CANDIDATE = "results/inkling2-gates/qualify-2026-11-23-c1"
EFFORT = "low"  # the scorer's
GIB_KIB = 2**20
T0 = datetime(2026, 11, 24, tzinfo=timezone.utc)
TREE = "5" * 64
FREEZE = "7" * 64
METHOD = "(completion_tokens - K) / (t_last_event - t_first_event), K = max tokens per event"
FIXTURE = "fixture text\n"


def stamp(seconds):
    return (T0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo(root: Path, mut: str) -> str:
    def git(*args):
        return subprocess.run(["git", "-C", str(root), "-c", "user.name=m", "-c", "user.email=m@m", *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    files = {
        f"{CANDIDATE}/PREREG.md": "frozen\n",
        f"{CANDIDATE}/gate_verdict.py": "v1\n",
        "configs/build-manifests/inkling-2-mini-runtime.json": json.dumps(
            {"vllm_tree": {"sha256": TREE, "files": 6525}, "patches": [{"path": f"p{i}"} for i in range(4)],
             "freeze_sha256": FREEZE}),
        "weights/inkling-2-mini/manifest.json": json.dumps(
            {"files": [{"name": "a", "bytes": 100}, {"name": "b", "bytes": 200}, {"name": "c", "bytes": 50}]}),
        "evidence.txt": "none\n",
        "fixtures/ctx-32k.txt": FIXTURE,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "freeze")
    if mut in ("headevidence", "headfrozen", "wrongcode"):
        target = "evidence.txt" if mut == "headevidence" else f"{CANDIDATE}/gate_verdict.py"
        (root / target).write_text("changed\n")
        git("commit", "-qam", "after")
    if mut == "wrongcode":  # the run's head changed frozen code; a later revert restores the tree
        head = git("rev-parse", "HEAD")
        git("revert", "--no-edit", "HEAD")
        return head
    if mut == "treeonly":  # the head is the freeze; the scoring tree has an uncommitted frozen edit
        (root / f"{CANDIDATE}/gate_verdict.py").write_text("edited\n")
    return git("rev-parse", "HEAD")


def build(tmp: Path, mut: str) -> Path:
    head = repo(tmp / "repo", mut)
    ev = tmp / "ev"
    obs = ev / "observe"
    obs.mkdir(parents=True)
    n = 100 if mut == "short" else 800
    rows = []
    for i in range(n):
        if ((mut == "gap" and 300 <= i < 310) or (mut == "gapend" and 776 <= i < 790)
                or (mut == "gatesecond" and i == 12) or (mut == "stalebase" and 3 <= i <= 12)):
            continue
        avail = (112.0 if mut == "lowonly" else 115.4) if i < 13 else 25.0
        if mut == "stalebase" and i < 3:  # production still loaded, then the sampler is silent to the launch
            avail = 12.0
        if i == 400:
            avail = {"lowmem": 9.9, "lowonly": 9.9, "peak": 10.2}.get(mut, 20.0)
        swap_used = GIB_KIB // 2 if mut == "swap" and i > 200 else 0
        if mut == "swapbase":  # 0.30 GiB before the stop, 0 at the launch, 0.45 GiB in the run
            swap_used = int(0.30 * GIB_KIB) if i < 5 else (int(0.45 * GIB_KIB) if i > 300 else 0)
        swap_free = 16 * GIB_KIB - swap_used
        rows.append(f"{stamp(i)} MemFree:={int(avail * GIB_KIB) - 100} MemAvailable:={int(avail * GIB_KIB)} "
                    f"SwapTotal:={16 * GIB_KIB} SwapFree:={swap_free} cg_anon=1 cg_file=1 \n")
    (obs / "meminfo.log").write_text("".join(rows))
    driver = [f"{stamp(0)} head {head if mut != 'headdiff' else 'b' * 40} dirty={1 if mut == 'dirty' else 0}"]
    if mut == "twoheads":
        driver.append(f"{stamp(1)} head {'c' * 40} dirty=0")
    driver += [
        f"interpreter match={mut != 'interpreter'}",
        f"vllm-tree {TREE[:16] if mut != 'runtime' else '6' * 16} files=6525 after='patches 0001-0004' "
        f"match={mut != 'runtime'}",
        f"patches 4 mismatches={['p2'] if mut == 'patch' else []}",
        f"weights files={4 if mut == 'weightcount' else 3} bytes=350 mismatches={1 if mut == 'weights' else 0}",
        f"packages freeze sha256_16 {FREEZE[:16] if mut != 'freeze' else '8' * 16} match={mut != 'freeze'}",
        f"{stamp(1)} agent-gate fixture sha256_16 "
        f"{hashlib.sha256((FIXTURE if mut != 'fixture' else 'other').encode()).hexdigest()[:16]}",
        f"{stamp(12 if mut == 'gatesecond' else 10)} MemFree gate pass 118000000 kB, "
        f"MemAvailable {121000000 - (12 * GIB_KIB if mut == 'gatelow' else 0)} kB (3 samples)",
        f"{stamp(12)} launching in scope inkling2-c1-1",
    ]
    if mut == "attempt":
        driver.append(f"{stamp(20)} FAILED_ATTEMPT surviving descendants")
    driver.append(f"{stamp(790)} scope empty after stop")
    if mut != "nojournal":
        driver.append(f"{stamp(795)} kernel journal alive after the engine stop")
    (obs / "driver.log").write_text("\n".join(driver) + "\n")
    (obs / "kernel.log").write_text("x NVRM: Xid 13\n" if mut == "xid" else "x drop_caches: 3\n")
    memwatch = {"unarmed": "ts=1 START\n", "disarmedonly": "ts=1 START\nts=2 DISARMED target_pid=1 target_pgid=1\n"}.get(
        mut, "ts=1 ARMED target_pid=1 target_pgid=1\n" + ("ts=2 BREACH\n" if mut == "breach" else ""))
    (obs / "memwatch.log").write_text(memwatch)
    if mut != "noserverlog":
        server = "" if mut == "emptyserver" else "===== p session start t =====\n"
        effort = {"low": "medium", "medium": "low"}[EFFORT] if mut == "effort" else EFFORT
        batch = 4096 if mut == "batchline" else 2048
        humming = "" if mut == "leverline" else "'moe_backend': 'humming', "
        server += (f"INFO non-default args: {{'default_chat_template_kwargs': {{'reasoning_effort': '{effort}'}}, "
                   f"'max_num_batched_tokens': {batch}, {humming}'speculative_config': "
                   f"{{'method': 'dspark', 'num_speculative_tokens': 7}}, 'kv_cache_memory_bytes': 68000000000}}\n")
        server += "INFO INKLING2_MAX_CTAS_POW2=1: trtllm MoE max CTAs rounded to a power of two\n"
        if mut != "patch4line":
            server += "INFO INKLING2_MAX_IMAGE_SIDE=4096: encoder budget sized for 4096 x 4096 px images\n"
        kv = {"kvshort": "1,000,000", "nokv": None}.get(mut, "1,238,000")
        if kv:
            server += f"INFO GPU KV cache size: {kv} tokens, Maximum concurrency for 262,144 tokens per request: 4.72x\n"
        roots = ["/home/bmarti44/.cache/inkling-2-mini", "/home/bmarti44/.cache/qwen38-flash-next"]
        if mut == "runtimerelease":
            roots = roots[:1]
        for root in roots:
            server += f"INFO host page cache released under {root} (77166 files) before the KV arena\n"
        if mut == "refused":
            server += ("ERROR host page cache released under /home/bmarti44/models (919 files); MemFree 60.00 GiB, "
                       "KV arena 63.33 GiB + margin 2.00 GiB: refusing to allocate the KV arena\n")
        elif mut != "norelease":
            server += ("INFO host page cache released under /home/bmarti44/models (919 files); MemFree 90.55 GiB, "
                       "KV arena 63.33 GiB + margin 2.00 GiB\n")
        (obs / "server.log").write_text(server + ("CUDA out of memory\n" if mut == "oom" else "ready\n"))

    budget = "256" if mut == "visionbudget" else "4096"
    (ev / "manifest.json").write_text(json.dumps({
        "git": {"head": "d" * 40 if mut == "kithead" else head},
        "profile": {"context_cap": 262_144 if mut == "cap" else 1_048_576},
        "cells": {"vision": {"argv": ["py", "38", "--thinking-mode", "thinking" if mut == "visionmode" else "chat",
                                      "--max-tokens", budget]}}}))
    (ev / "context").mkdir()
    (ev / "context" / "summary.json").write_text(json.dumps({
        "verdict": "FAIL" if mut == "context" else "PASS",
        "total_tokens": 990_000 if mut == "tokens" else 1_000_540, "max_ttft_s": 684.0,
        "min_mem_available_gib": 20.6}))
    greedy = [{"sha256": "abc"}, {"sha256": "abd"}, {"sha256": "abe"}]
    (tmp / "reference-probe.json").write_text(json.dumps({"greedy": greedy}))
    if mut not in ("noprobe", "failmissing"):
        mine = [dict(g) for g in greedy]
        if mut == "identity":
            mine[2]["sha256"] = "zzz"
        (ev / "probe-c1.json").write_text(json.dumps({"greedy": mine}))
    (ev / "toolcall").mkdir()
    (ev / "toolcall" / "toolcall.json").write_text("[]" if mut == "toptype" else json.dumps(
        {"passed": 18 if mut in ("tool", "failmissing") else 20, "total": 19 if mut == "tooltotal" else 20}))
    tests = ["prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"]
    if mut == "agentmissing":
        tests = tests[:3]
    (ev / "agent-gate.json").write_text(json.dumps({
        "engine": "vllm", "model": "inkling-2-mini", "spec_block_drop": mut != "specdrop",
        "prefix_block_tokens": 2048 if mut == "agentslack" else 64,
        "results": "4/4" if mut == "nestedtype" else
                   [{"test": t, "pass": not (mut == "agent" and t == "turn-continuation")} for t in tests]}))
    (ev / "soak").mkdir()
    (ev / "soak" / "soak.json").write_text(json.dumps({
        "pass": mut != "soak", "mem_floor_gib": 12.0 if mut == "soakfloor" else 5.0,
        "mem_floor_source": "env-override" if mut == "soaksource" else "default", "mem_available_min_gib": 9.7}))
    (ev / "vision").mkdir()
    correct = {"vision52": 52, "vision53": 53}.get(mut, 60)
    (ev / "vision" / "summary.json").write_text(json.dumps({
        "n": 99 if mut == "visionn" else 100, "correct": correct, "invalid_count": 1,
        "error_count": 1 if mut == "visionerror" else 0}))
    (ev / "speed").mkdir()
    (ev / "speed" / "speed.json").write_text(json.dumps({
        "metadata": {**({} if mut == "speedk" else {"max_tokens_per_event": 8}),
                     **({} if mut == "speedmethod" else {"method": METHOD})},
        "suite_valid": mut != "speed",
        "cells": [{"ctx_tokens": 0, "median_decode": 21.9, "median_ttft": 0.6, "valid": True},
                  {"ctx_tokens": 28672, "median_decode": 35.7, "median_ttft": 45.0, "valid": mut != "speedlevel"}]}))
    return ev


EXPECT = {
    "clean": "PASS", "headevidence": "PASS", "vision53": "PASS",
    "xid": "FAIL", "breach": "FAIL", "oom": "FAIL", "swap": "FAIL", "lowmem": "FAIL", "peak": "FAIL",
    "context": "FAIL", "tokens": "FAIL", "identity": "FAIL", "tool": "FAIL", "agent": "FAIL", "soak": "FAIL",
    "vision52": "FAIL", "visionerror": "FAIL",
    "cap": "NO_RESULT", "noprobe": "NO_RESULT", "tooltotal": "NO_RESULT", "agentmissing": "NO_RESULT",
    "agentslack": "NO_RESULT", "soakfloor": "NO_RESULT", "visionn": "NO_RESULT", "visionbudget": "NO_RESULT",
    "speed": "NO_RESULT", "attempt": "NO_RESULT", "dirty": "NO_RESULT", "twoheads": "NO_RESULT",
    "headdiff": "NO_RESULT", "headfrozen": "NO_RESULT", "runtime": "NO_RESULT", "interpreter": "NO_RESULT",
    "patch": "NO_RESULT",
    "weights": "NO_RESULT", "weightcount": "NO_RESULT", "noserverlog": "NO_RESULT", "unarmed": "NO_RESULT",
    "emptyserver": "NO_RESULT", "gap": "NO_RESULT",  # dropped by mistake in candidate 2 (A2-L2)
    "short": "NO_RESULT", "kithead": "NO_RESULT", "nojournal": "NO_RESULT",
    # Each of these breaks one check alone (review round 1, A-M1 and the A lows).
    "lowonly": "FAIL", "failmissing": "FAIL", "swapbase": "FAIL", "gatesecond": "PASS",
    "wrongcode": "NO_RESULT", "treeonly": "NO_RESULT", "speedlevel": "NO_RESULT", "visionmode": "NO_RESULT",
    "soaksource": "NO_RESULT", "gapend": "NO_RESULT", "disarmedonly": "NO_RESULT", "toptype": "NO_RESULT",
    "nestedtype": "NO_RESULT", "fixture": "NO_RESULT", "nomutationflag": "NO_RESULT",
    "gatelow": "NO_RESULT",  # review round 1, B3: MemAvailable 113.4 GiB at the gate
    "stalebase": "NO_RESULT",  # review round 2, A2-L1: the last sample before the launch is 10 s old
    # Inkling's own clauses: the direct 1M capacity, the configuration lines and the worker.
    "kvshort": "FAIL", "nokv": "NO_RESULT", "refused": "FAIL", "runtimerelease": "NO_RESULT",
    "norelease": "NO_RESULT", "effort": "NO_RESULT", "leverline": "NO_RESULT", "patch4line": "NO_RESULT",
    "speedk": "NO_RESULT",
    # Review A-M3-L2 / L4: the runtime's package freeze and the block-aware speed method.
    "freeze": "NO_RESULT", "speedmethod": "NO_RESULT",
    # The carrier's configuration (a6): the batch line and the agent gate's DSpark slack.
    "batchline": "NO_RESULT", "specdrop": "NO_RESULT",
}
# The reason a case must give (a substring of one fail or missing entry), where a
# different check could otherwise produce the same verdict.
REASON = {
    "lowonly": "clause 3", "failmissing": "clause 7", "swapbase": "clause 2", "wrongcode": "code identity",
    "treeonly": "scoring identity", "speedlevel": "clause 11", "visionmode": "vision protocol",
    "soaksource": "clause 9", "gapend": "clauses 2-4", "disarmedonly": "memwatch.log", "toptype": "toolcall.json",
    "nestedtype": "malformed", "fixture": "agent-gate fixture", "nomutationflag": "outside --mutation",
    "headdiff": "code identity", "headfrozen": "scoring identity", "kithead": "kit manifest head",
    "gatelow": "start gate", "nojournal": "kernel journal alive", "emptyserver": "server.log",
    "gap": "clauses 2-4", "stalebase": "clauses 2-4",
    "peak": "clause 4", "identity": "clause 6", "kvshort": "clause 5: KV capacity", "nokv": "KV capacity line", "refused": "clause 12",
    "runtimerelease": "clause 12", "norelease": "clause 12", "effort": "reasoning_effort",
    "leverline": "lever lines", "batchline": "lever lines", "specdrop": "agent gate", "patch4line": "patch 0004", "speedk": "clause 11", "interpreter": "interpreter",
    "patch": "patches", "freeze": "packages freeze", "speedmethod": "clause 11", "runtime": "vllm-tree", "weights": "weights", "weightcount": "weights",
}


def main() -> int:
    ok = True
    for mut, expected in EXPECT.items():
        with tempfile.TemporaryDirectory() as tmp:
            ev = build(Path(tmp), mut)
            env = dict(os.environ, GATE_REPO=str(Path(tmp) / "repo"),
                       GATE_REFERENCE_PROBE=str(Path(tmp) / "reference-probe.json"))
            flags = [] if mut == "nomutationflag" else ["--mutation"]
            proc = subprocess.run([sys.executable, str(SCORER), str(ev), *flags], capture_output=True, text=True, env=env)
            try:
                summary = json.loads((ev / "gate-summary.json").read_text())
                got = summary["verdict"]
            except (OSError, ValueError, KeyError):
                summary, got = {}, f"CRASH rc={proc.returncode} {proc.stderr.strip()[-200:]}"
            rc_ok = (proc.returncode == 0) == (got == "PASS")
            reasons = (summary.get("fail") or []) + (summary.get("missing") or []) if summary else []
            reason_ok = mut not in REASON or any(REASON[mut] in r for r in reasons)
            match = got == expected and rc_ok and reason_ok
            ok &= match
            why = next((r for r in reasons if REASON.get(mut, "") in r), reasons[0] if reasons else "")
            print(f"{'ok ' if match else 'BAD'} {mut:13s} expected={expected:9s} got={got} rc={proc.returncode}"
                  + ("" if match and expected == "PASS" else f"  [{why[:110]}]"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
