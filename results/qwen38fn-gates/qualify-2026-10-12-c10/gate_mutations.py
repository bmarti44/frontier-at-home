#!/usr/bin/env python3
"""Mutation check for gate_verdict.py: a clean synthetic bundle must PASS and
every broken or failing variant must not.

Usage: gate_mutations.py  -> prints one line per case, exit 0 iff all match.
Run 2's drand receipt is a real published round (the W7 verifier fixture);
the synthetic freeze commit is dated so that round is the first after it, so
the bundled BLS verifier runs for real.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORER = HERE / "gate_verdict.py"
CODE = HERE.parents[2]
CANDIDATE = "results/qwen38fn-gates/qualify-2026-10-12-c10"
GIB_KIB = 2**20
LAUNCH = 5
FIXTURE = json.loads((CODE / "results/glm52-gates/W7-resume-production-drand-verifier-fixture.json").read_text())
ROUND = int(FIXTURE["round"])
_spec = importlib.util.spec_from_file_location("matched", CODE / "scripts/97_matched_speed_blocks.py")
MATCHED = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MATCHED)
GATE = importlib.util.spec_from_file_location("gate", SCORER)
_gate = importlib.util.module_from_spec(GATE)
GATE.loader.exec_module(_gate)


def stamp(second: int) -> str:
    return f"2026-10-12T00:{second // 60:02d}:{second % 60:02d}Z"


def git_repo(repo: Path, mut: str) -> tuple[str, str]:
    """A repo whose freeze commit adds PREREG.md; return (freeze, the runs' head)."""
    frozen_at = MATCHED.DRAND_GENESIS + (ROUND - 2) * MATCHED.DRAND_PERIOD + 5
    env = dict(os.environ, GIT_COMMITTER_DATE=f"@{frozen_at} +0000", GIT_AUTHOR_DATE=f"@{frozen_at} +0000")

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=m", "-c", "user.email=m@m", *args],
                              capture_output=True, text=True, check=True, env=env).stdout.strip()
    profile = repo / "configs/profiles/qwen3.8-flash-next/p.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("v1\n")
    toolcall = repo / "evalsets/toolcall/cases.jsonl"
    toolcall.parent.mkdir(parents=True)
    toolcall.write_text("{}\n")
    tool = repo / "scripts/97_matched_speed_blocks.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("v1\n")
    (repo / "evidence.txt").write_text("start1\n")
    git("init", "-q")
    if mut != "noprereg":
        prereg = repo / CANDIDATE / "PREREG.md"
        prereg.parent.mkdir(parents=True)
        prereg.write_text("rule\n")
    git("add", "-A")
    git("commit", "-qm", "freeze")
    freeze = git("rev-parse", "HEAD")
    if mut in ("headevidence", "headfrozen", "headtoolcall", "headtool"):
        changed = {"headfrozen": profile, "headtoolcall": toolcall, "headtool": tool}.get(mut, repo / "evidence.txt")
        changed.write_text("changed\n")
        git("commit", "-qam", "later")
    return freeze, git("rev-parse", "HEAD")


def speed_doc(decode0, decode28, prefill, ttft0=0.21):
    return {"suite_valid": True, "cells": [
        {"ctx_tokens": 0, "median_decode": decode0, "median_ttft": ttft0, "valid": True,
         "reps": [{"prefill_tok_s": 350.0}] * 3},
        {"ctx_tokens": 28672, "median_decode": decode28, "median_ttft": 10.6, "valid": True,
         "reps": [{"prefill_tok_s": prefill}] * 3}]}


def build_run2(ev: Path, mut: str, freeze: str, head: str) -> None:
    if mut == "noplan":
        return
    receipt = {"round": ROUND + (1 if mut == "wronground" else 0), "randomness": FIXTURE["randomness"],
               "signature": FIXTURE["signature"], "previous_signature": FIXTURE["previous_signature"],
               "relays": list(MATCHED.DRAND_RELAYS), "frozen_commit": "0" * 40 if mut == "othercommit" else freeze}
    orders = MATCHED.block_orders(FIXTURE["randomness"], 5, _gate.DOMAIN)
    plan = {"schema_version": 1, "domain": _gate.DOMAIN, "randomness": FIXTURE["randomness"],
            "blocks": [{"ABBA": "BAAB", "BAAB": "ABBA"}[o] for o in orders] if mut == "orders" else orders,
            "a_profile": _gate.A_PROFILE,
            "b_profile": _gate.B_PROFILE + ("-x" if mut == "wrongarm" else ""), "drand": receipt}
    (ev / "plan.json").write_text(json.dumps(plan))
    blocks = ev / "blocks"
    blocks.mkdir()
    (blocks / "plan.json").write_text(json.dumps(plan))
    (blocks / "driver.log").write_text(f"{stamp(0)} head {head} dirty={1 if mut == 'r2dirty' else 0}\n")
    noisy = {0: 1.25, 1: 0.85, 2: 1.30, 3: 0.80, 4: 1.10}
    for block, position, arm in MATCHED.schedule(plan["blocks"]):
        d = blocks / f"block-{block}" / f"{position}-{arm}"
        d.mkdir(parents=True)
        j = 1.0 + 0.002 * position
        if arm == "B":
            d0 = 1.05 if mut == "slow0" else 1.35
            d28 = noisy[block] if mut == "noisy" else 1.16
            pf = 0.90 if mut == "prefill" else 1.01
        else:
            d0 = d28 = pf = 1.0
        start = {"block": block, "position": position, "arm": arm,
                 "profile": plan["a_profile" if arm == "A" else "b_profile"], "failed": None,
                 "kill_lines": ["kernel.log: NVRM: Xid 13"] if mut == "r2kill" and block == 2 else [],
                 "swap_growth_gib": 0.6 if mut == "r2swap" and block == 1 else 0.0,
                 "low_mem_available_gib": 7.5 if mut == "r2low" and block == 3 else 10.4}
        (d / "start.json").write_text(json.dumps(start))
        if not (mut == "r2missing" and block == 4 and position == 1):
            (d / "speed.json").write_text(json.dumps(speed_doc(25.3 * d0 * j, 35.5 * d28 * j, 2675.0 * pf * j)))


def build(ev: Path, mut: str) -> None:
    freeze, head = git_repo(ev / "repo", mut)
    obs = ev / "observe" / "start1"
    obs.mkdir(parents=True)
    lines = []
    for i in range(700):
        if i <= LAUNCH:
            avail = (117 if mut == "admission" or (mut == "admissionlate" and i) else
                     114 if mut == "admissionlate" else 115) * GIB_KIB
        else:
            avail = int(9.5 * GIB_KIB) - i * 10
        if mut == "lowmem" and i == 300:
            avail = int(7.9 * GIB_KIB)
        swap_free = 16 * GIB_KIB - (GIB_KIB // 2 if mut == "swap" and i > 100 else 0)
        lines.append(f"{stamp(i)} MemAvailable:={avail} SwapTotal:={16 * GIB_KIB} SwapFree:={swap_free} \n")
    if mut == "short":
        lines = lines[:100]
    (obs / "meminfo.log").write_text("".join(lines))
    (obs / "kernel.log").write_text("x NVRM: Xid 13\n" if mut == "xid" else "")
    memwatch = "ts=1 ARMED\nts=2 BREACH\n" if mut == "breach" else "ts=1 ARMED\n"
    (obs / "memwatch.log").write_text("ts=1 START\n" if mut == "unarmed" else memwatch)
    tokens = "1,040,000" if mut == "kvcap" else "1,079,117"
    root = "/tmp/models" if mut == "wrongroot" else "/home/bmarti44/models"
    release = "" if mut == "norelease" else (
        f"(EngineCore pid=1) INFO 10-12 13:09:03 [qwen38fn_worker.py:86] host page cache released under {root} "
        f"({0 if mut == 'zerofiles' else 496} files); MemFree 24.10 GiB, KV arena 17.90 GiB + margin 2.00 GiB\n")
    if mut == "refused":
        release = release.replace("INFO", "ERROR").rstrip("\n") + ": refusing to allocate the KV arena\n"
    rroot = "/home/bmarti44/.cache/other" if mut == "runtimewrongroot" else _gate.RUNTIME_ROOT
    runtime_release = "" if mut == "noruntimerelease" else (
        f"(EngineCore pid=1) INFO 10-12 13:09:02 [qwen38fn_worker.py:83] host page cache released under {rroot} "
        f"({0 if mut == 'runtimezero' else 134720} files) before the KV arena\n")
    server = "" if mut == "emptyserver" else (
        f"===== p session start t =====\n{runtime_release}{release}INFO GPU KV cache size: {tokens} tokens, "
        "Maximum concurrency for 262,144 tokens per request: 4.12x\n"
        + ("torch.OutOfMemoryError\n" if mut == "oom" else "INFO ready\n"))
    if mut != "noserverlog":
        (obs / "server.log").write_text(server)
    dirty = 1 if mut == "dirty" else 0
    launch = "" if mut == "nolaunch" else f"{stamp(LAUNCH)} start1: launching in scope u (RuntimeMaxSec=1)\n"
    gguf, mmproj = (("1.00", "0.50") if mut == "coldcache" else ("16.55", "0.86"))
    lib = "0.10" if mut == "coldruntime" else "0.33"
    precondition = "" if mut == "noprecondition" else (
        f"{stamp(LAUNCH)} precondition {gguf} GiB {_gate.GGUF[0]}\n"
        f"{stamp(LAUNCH)} precondition {mmproj} GiB {_gate.GGUF[1]}\n"
        + "".join(f"{stamp(LAUNCH)} precondition {lib} GiB {p}\n"
                  for p in _gate.RUNTIME_LIBS[: 8 if mut == "fewlibs" else 9]))
    (obs / "driver.log").write_text(f"{stamp(0)} head {head} dirty={dirty}\n" + precondition + launch
                                    + ("FAILED_ATTEMPT survivors\n" if mut == "attempt" else "ok\n"))
    (ev / "manifest.json").write_text(json.dumps(
        {"profile": {"context_cap": 1_000_000 if mut == "cap" else 1_048_576}}))
    (ev / "context").mkdir()
    (ev / "context" / "summary.json").write_text(json.dumps(
        {"verdict": "FAIL" if mut == "context" else "PASS",
         "total_tokens": 900_000 if mut == "tokens" else 1_000_512, "min_mem_available_gib": 9.5}))
    reference = [{"sha256": f"r{i}"} for i in range(3)]
    if mut != "noreference":
        (ev / "reference-probe.json").write_text(json.dumps({"greedy": reference}))
    rows = [{"sha256": "x" if (mut == "identity" and i == 2) else f"r{i}"} for i in range(3)]
    if mut != "noprobe":
        (ev / "probe-start1.json").write_text(json.dumps({"greedy": rows}))
    (ev / "toolcall").mkdir()
    (ev / "toolcall" / "toolcall.json").write_text(json.dumps(
        {"passed": 18 if mut == "tool" else 20, "total": 20}))
    tests = ["prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"]
    if mut == "agentmissing":
        tests = tests[:3]
    (ev / "agent-gate.json").write_text(json.dumps({"results": [
        {"test": t, "pass": not (mut == "agent" and t == "turn-continuation")} for t in tests]}))
    (ev / "soak").mkdir()
    floor = {"mem_floor_gib": 12.0, "mem_floor_source": "default"} if mut == "soakfloor" else \
        {"mem_floor_gib": 5.0, "mem_floor_source": "default"}
    (ev / "soak" / "soak.json").write_text(json.dumps({"pass": mut != "soak", "mem_available_min_gib": 9.9, **floor}))
    correct = {"visionworse": 52, "vision53": 53, "vision63": 63}.get(mut, 64)
    if mut != "novision":
        (ev / "vision").mkdir()
        (ev / "vision" / "summary.json").write_text(json.dumps(
            {"ok": mut != "visionerror", "n": 99 if mut == "visionshort" else 100,
             "correct": correct, "invalid_count": 3,
             "error_count": 1 if mut == "visionerror" else 0}))
    if mut != "nospeed":
        (ev / "speed").mkdir()
        (ev / "speed" / "speed.json").write_text(json.dumps(
            speed_doc(34.2, 41.1, 2709.0, ttft0=0.6 if mut == "ttft" else 0.21)))
    build_run2(ev, mut, freeze, head)


EXPECT = {
    # candidate 9's cases (identity is NO_RESULT here: a changed text path needs the holdout)
    "clean": "PASS", "headevidence": "PASS", "vision53": "PASS", "vision63": "PASS",
    "xid": "FAIL", "breach": "FAIL", "oom": "FAIL", "swap": "FAIL", "lowmem": "FAIL", "admission": "FAIL",
    "admissionlate": "FAIL", "nolaunch": "NO_RESULT", "headtoolcall": "NO_RESULT",
    "refused": "FAIL", "norelease": "NO_RESULT", "wrongroot": "NO_RESULT",
    "noprecondition": "NO_RESULT", "coldcache": "NO_RESULT", "zerofiles": "NO_RESULT", "soakfloor": "NO_RESULT",
    "context": "FAIL", "tokens": "FAIL", "identity": "NO_RESULT", "tool": "FAIL", "agent": "FAIL",
    "soak": "FAIL", "kvcap": "FAIL", "visionworse": "FAIL", "visionerror": "FAIL",
    "cap": "NO_RESULT", "short": "NO_RESULT", "noserverlog": "NO_RESULT", "noprobe": "NO_RESULT",
    "noreference": "NO_RESULT", "agentmissing": "NO_RESULT", "attempt": "NO_RESULT", "dirty": "NO_RESULT",
    "emptyserver": "NO_RESULT", "unarmed": "NO_RESULT", "headfrozen": "NO_RESULT", "noprereg": "NO_RESULT",
    "visionshort": "NO_RESULT", "novision": "NO_RESULT",
    # candidate 10: runtime precondition and release (clauses 11, 12)
    "coldruntime": "NO_RESULT", "fewlibs": "NO_RESULT", "noruntimerelease": "NO_RESULT",
    "runtimezero": "NO_RESULT", "runtimewrongroot": "NO_RESULT",
    # run 2 safety (clauses 1-3), matched speed (13), TTFT (14), identity and randomness
    "r2kill": "FAIL", "r2swap": "FAIL", "r2low": "FAIL", "r2missing": "NO_RESULT",
    "slow0": "FAIL", "noisy": "FAIL", "prefill": "FAIL", "ttft": "FAIL", "nospeed": "NO_RESULT",
    "noplan": "NO_RESULT", "wronground": "NO_RESULT", "othercommit": "NO_RESULT", "orders": "NO_RESULT",
    "wrongarm": "NO_RESULT", "r2dirty": "NO_RESULT", "headtool": "NO_RESULT",
}


def main() -> int:
    ok = True
    for mut, expected in EXPECT.items():
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            build(ev, mut)
            env = dict(os.environ, GATE_REPO=str(ev / "repo"),
                       GATE_REFERENCE_PROBE=str(ev / "reference-probe.json"))
            proc = subprocess.run([sys.executable, str(SCORER), str(ev)], capture_output=True, text=True, env=env)
            try:
                got = json.loads((ev / "gate-summary.json").read_text())["verdict"]
            except (OSError, ValueError):
                got = f"crash: {proc.stderr.strip().splitlines()[-1:]}"
            rc_ok = (proc.returncode == 0) == (got == "PASS")
            match = got == expected and rc_ok
            ok &= match
            print(f"{'ok ' if match else 'BAD'} {mut:16s} expected={expected:9s} got={got} rc={proc.returncode}")
    print(f"{sum(1 for _ in EXPECT)} cases")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
