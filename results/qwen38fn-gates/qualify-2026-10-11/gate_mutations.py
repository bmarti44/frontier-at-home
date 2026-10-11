#!/usr/bin/env python3
"""Mutation check for gate_verdict.py: a clean synthetic bundle must PASS and
every broken or failing variant must not.

Usage: gate_mutations.py  -> prints one line per case, exit 0 iff all match.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCORER = Path(__file__).resolve().parent / "gate_verdict.py"
GIB_KIB = 2**20


def build(ev: Path, mut: str) -> None:
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
        (obs / "memwatch.log").write_text("ts=1 BREACH\n" if mut == "breach" else "ts=1 ARMED\n")
        if not (mut == "noserverlog" and start == "start2"):
            (obs / "server.log").write_text("torch.OutOfMemoryError\n" if mut == "oom" else "INFO ready\n")
        (obs / "driver.log").write_text("FAILED_ATTEMPT survivors\n" if mut == "attempt" else "ok\n")
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
    "attempt": "NO_RESULT",
}


def main() -> int:
    ok = True
    for mut, expected in EXPECT.items():
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            build(ev, mut)
            proc = subprocess.run([sys.executable, str(SCORER), str(ev)], capture_output=True, text=True)
            got = json.loads((ev / "gate-summary.json").read_text())["verdict"]
            rc_ok = (proc.returncode == 0) == (got == "PASS")
            match = got == expected and rc_ok
            ok &= match
            print(f"{'ok ' if match else 'BAD'} {mut:13s} expected={expected:9s} got={got} rc={proc.returncode}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
