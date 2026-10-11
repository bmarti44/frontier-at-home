#!/usr/bin/env python3
"""Mutation check for gate_verdict.py: a clean synthetic bundle must PASS and
every broken or failing variant must not.

Usage: gate_mutations.py  -> prints one line per case, exit 0 iff all match.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCORER = Path(__file__).resolve().parent / "gate_verdict.py"
CANDIDATE = "results/qwen38fn-gates/qualify-2026-10-11-c7"
GIB_KIB = 2**20


def git_repo(repo: Path, mut: str) -> str:
    """A repo whose freeze commit adds PREREG.md; return the run's head."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=m", "-c", "user.email=m@m", *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    profile = repo / "configs/profiles/qwen3.8-flash-next/p.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("v1\n")
    (repo / "evidence.txt").write_text("start1\n")
    git("init", "-q")
    if mut != "noprereg":
        prereg = repo / CANDIDATE / "PREREG.md"
        prereg.parent.mkdir(parents=True)
        prereg.write_text("rule\n")
    git("add", "-A")
    git("commit", "-qm", "freeze")
    if mut in ("headevidence", "headfrozen"):
        (profile if mut == "headfrozen" else repo / "evidence.txt").write_text("changed\n")
        git("commit", "-qam", "later")
    return git("rev-parse", "HEAD")


def build(ev: Path, mut: str) -> None:
    head = git_repo(ev / "repo", mut)
    obs = ev / "observe" / "start1"
    obs.mkdir(parents=True)
    lines = []
    for i in range(700):
        avail = (117 if mut == "admission" else 115) * GIB_KIB if i == 0 else int(9.5 * GIB_KIB) - i * 10
        if mut == "lowmem" and i == 300:
            avail = int(7.9 * GIB_KIB)
        swap_free = 16 * GIB_KIB - (GIB_KIB // 2 if mut == "swap" and i > 100 else 0)
        lines.append(f"t MemAvailable:={avail} SwapTotal:={16 * GIB_KIB} SwapFree:={swap_free} \n")
    if mut == "short":
        lines = lines[:100]
    (obs / "meminfo.log").write_text("".join(lines))
    (obs / "kernel.log").write_text("x NVRM: Xid 13\n" if mut == "xid" else "")
    memwatch = "ts=1 ARMED\nts=2 BREACH\n" if mut == "breach" else "ts=1 ARMED\n"
    (obs / "memwatch.log").write_text("ts=1 START\n" if mut == "unarmed" else memwatch)
    tokens = "1,040,000" if mut == "kvcap" else "1,079,117"
    server = "" if mut == "emptyserver" else (
        f"===== p session start t =====\nINFO GPU KV cache size: {tokens} tokens, "
        "Maximum concurrency for 262,144 tokens per request: 4.12x\n"
        + ("torch.OutOfMemoryError\n" if mut == "oom" else "INFO ready\n"))
    if mut != "noserverlog":
        (obs / "server.log").write_text(server)
    dirty = 1 if mut == "dirty" else 0
    (obs / "driver.log").write_text(f"2026-10-11T00:00:00Z head {head} dirty={dirty}\n"
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
    (ev / "soak" / "soak.json").write_text(json.dumps({"pass": mut != "soak"}))
    correct = {"visionworse": 52, "vision53": 53, "vision63": 63}.get(mut, 64)
    if mut != "novision":
        (ev / "vision").mkdir()
        (ev / "vision" / "summary.json").write_text(json.dumps(
            {"ok": mut != "visionerror", "n": 99 if mut == "visionshort" else 100,
             "correct": correct, "invalid_count": 3,
             "error_count": 1 if mut == "visionerror" else 0}))


EXPECT = {
    "clean": "PASS", "headevidence": "PASS", "vision53": "PASS", "vision63": "PASS",
    "xid": "FAIL", "breach": "FAIL", "oom": "FAIL", "swap": "FAIL", "lowmem": "FAIL", "admission": "FAIL",
    "context": "FAIL", "tokens": "FAIL", "identity": "FAIL", "tool": "FAIL", "agent": "FAIL",
    "soak": "FAIL", "kvcap": "FAIL", "visionworse": "FAIL", "visionerror": "FAIL",
    "cap": "NO_RESULT", "short": "NO_RESULT", "noserverlog": "NO_RESULT", "noprobe": "NO_RESULT",
    "noreference": "NO_RESULT", "agentmissing": "NO_RESULT", "attempt": "NO_RESULT", "dirty": "NO_RESULT",
    "emptyserver": "NO_RESULT", "unarmed": "NO_RESULT", "headfrozen": "NO_RESULT", "noprereg": "NO_RESULT",
    "visionshort": "NO_RESULT", "novision": "NO_RESULT",
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
            got = json.loads((ev / "gate-summary.json").read_text())["verdict"]
            rc_ok = (proc.returncode == 0) == (got == "PASS")
            match = got == expected and rc_ok
            ok &= match
            print(f"{'ok ' if match else 'BAD'} {mut:13s} expected={expected:9s} got={got} rc={proc.returncode}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
