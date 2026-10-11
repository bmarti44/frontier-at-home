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
GIB_KIB = 2**20


def git_heads(repo: Path, mut: str) -> tuple[str, str]:
    """Two real commits: evidence-only change, or a change to a frozen file."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=m", "-c", "user.email=m@m", *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    frozen = repo / "results/qwen38fn-gates/qualify-2026-10-11/gate_verdict.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("v1\n")
    (repo / "evidence.txt").write_text("start1\n")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "c1")
    first = git("rev-parse", "HEAD")
    (frozen if mut == "headfrozen" else repo / "evidence.txt").write_text("changed\n")
    git("commit", "-qam", "c2")
    return first, git("rev-parse", "HEAD")


def build(ev: Path, mut: str) -> None:
    heads = git_heads(ev / "repo", mut) if mut in ("headevidence", "headfrozen") else None
    for start in ("start1", "start2"):
        obs = ev / "observe" / start
        obs.mkdir(parents=True)
        lines = []
        for i in range(700):
            avail = 15 * GIB_KIB - i * 10
            if mut == "lowmem" and start == "start1" and i == 300:
                avail = 9 * GIB_KIB
            swap_free = 16 * GIB_KIB - (GIB_KIB // 2 if mut == "swap" and start == "start2" and i > 100 else 0)
            lines.append(f"t MemAvailable:={avail} SwapTotal:={16 * GIB_KIB} SwapFree:={swap_free} \n")
        if mut == "short" and start == "start2":
            lines = lines[:100]
        (obs / "meminfo.log").write_text("".join(lines))
        (obs / "kernel.log").write_text("x NVRM: Xid 13\n" if mut == "xid" and start == "start1" else "")
        memwatch = "ts=1 ARMED\nts=2 BREACH\n" if mut == "breach" else "ts=1 ARMED\n"
        (obs / "memwatch.log").write_text("ts=1 START\n" if mut == "unarmed" else memwatch)
        tokens = "1,040,000" if mut == "kvcap" and start == "start2" else "1,050,123"
        server = "" if mut == "emptyserver" and start == "start1" else (
            f"===== p session start t =====\nINFO GPU KV cache size: {tokens} tokens, "
            "Maximum concurrency for 262,144 tokens per request: 4.01x\n"
            + ("torch.OutOfMemoryError\n" if mut == "oom" else "INFO ready\n"))
        if not (mut == "noserverlog" and start == "start2"):
            (obs / "server.log").write_text(server)
        head = "b" * 40 if mut == "headdiff" and start == "start2" else "a" * 40
        if heads:
            head = heads[0] if start == "start1" else heads[1]
        dirty = 1 if mut == "dirty" and start == "start1" else 0
        (obs / "driver.log").write_text(f"2026-10-11T00:00:00Z head {head} dirty={dirty}\n"
                                        + ("FAILED_ATTEMPT survivors\n" if mut == "attempt" else "ok\n"))
    (ev / "manifest.json").write_text(json.dumps(
        {"profile": {"context_cap": 1_000_000 if mut == "cap" else 1_048_576}}))
    (ev / "context").mkdir()
    (ev / "context" / "summary.json").write_text(json.dumps(
        {"verdict": "FAIL" if mut == "context" else "PASS",
         "total_tokens": 900_000 if mut == "tokens" else 1_000_512, "min_mem_available_gib": 13.0}))
    for n, start in enumerate(("start1", "start2")):
        rows = [{"sha256": "abc" if not (mut == "identity" and n == 1 and i == 2) else "abd"}
                for i in range(3)]
        if not (mut == "noprobe" and n == 1):
            (ev / f"probe-{start}.json").write_text(json.dumps({"greedy": rows}))
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


EXPECT = {
    "clean": "PASS", "xid": "FAIL", "breach": "FAIL", "oom": "FAIL", "swap": "FAIL",
    "lowmem": "FAIL", "context": "FAIL", "tokens": "FAIL", "identity": "FAIL", "tool": "FAIL",
    "agent": "FAIL", "soak": "FAIL", "cap": "NO_RESULT", "short": "NO_RESULT",
    "noserverlog": "NO_RESULT", "noprobe": "NO_RESULT", "agentmissing": "NO_RESULT",
    "attempt": "NO_RESULT", "kvcap": "FAIL", "headdiff": "NO_RESULT", "dirty": "NO_RESULT",
    "emptyserver": "NO_RESULT", "unarmed": "NO_RESULT", "headevidence": "PASS", "headfrozen": "NO_RESULT",
}


def main() -> int:
    ok = True
    for mut, expected in EXPECT.items():
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            build(ev, mut)
            env = dict(os.environ, GATE_REPO=str(ev / "repo"))
            proc = subprocess.run([sys.executable, str(SCORER), str(ev)], capture_output=True, text=True, env=env)
            got = json.loads((ev / "gate-summary.json").read_text())["verdict"]
            rc_ok = (proc.returncode == 0) == (got == "PASS")
            match = got == expected and rc_ok
            ok &= match
            print(f"{'ok ' if match else 'BAD'} {mut:13s} expected={expected:9s} got={got} rc={proc.returncode}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
