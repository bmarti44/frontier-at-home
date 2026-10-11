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
CANDIDATE = "results/qwen38fn-gates/qualify-2026-10-11-c9"
GIB_KIB = 2**20
LAUNCH = 5


def stamp(second: int) -> str:
    return f"2026-10-11T00:{second // 60:02d}:{second % 60:02d}Z"


def git_repo(repo: Path, mut: str) -> str:
    """A repo whose freeze commit adds PREREG.md; return the run's head."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=m", "-c", "user.email=m@m", *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    profile = repo / "configs/profiles/qwen3.8-flash-next/p.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("v1\n")
    toolcall = repo / "evalsets/toolcall/cases.jsonl"
    toolcall.parent.mkdir(parents=True)
    toolcall.write_text("{}\n")
    (repo / "evidence.txt").write_text("start1\n")
    git("init", "-q")
    if mut != "noprereg":
        prereg = repo / CANDIDATE / "PREREG.md"
        prereg.parent.mkdir(parents=True)
        prereg.write_text("rule\n")
    git("add", "-A")
    git("commit", "-qm", "freeze")
    if mut in ("headevidence", "headfrozen", "headtoolcall"):
        changed = {"headfrozen": profile, "headtoolcall": toolcall}.get(mut, repo / "evidence.txt")
        changed.write_text("changed\n")
        git("commit", "-qam", "later")
    return git("rev-parse", "HEAD")


def build(ev: Path, mut: str) -> None:
    head = git_repo(ev / "repo", mut)
    obs = ev / "observe" / "start1"
    obs.mkdir(parents=True)
    lines = []
    for i in range(700):
        # Samples 0-5 are before the launch line (stamped at second 5).
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
        f"(EngineCore pid=1) INFO 10-11 13:09:03 [qwen38fn_worker.py:66] host page cache released under {root} "
        f"({0 if mut == 'zerofiles' else 496} files); MemFree 24.10 GiB, KV arena 17.90 GiB + margin 2.00 GiB\n")
    if mut == "refused":
        release = release.replace("INFO", "ERROR").rstrip("\n") + ": refusing to allocate the KV arena\n"
    server = "" if mut == "emptyserver" else (
        f"===== p session start t =====\n{release}INFO GPU KV cache size: {tokens} tokens, "
        "Maximum concurrency for 262,144 tokens per request: 4.12x\n"
        + ("torch.OutOfMemoryError\n" if mut == "oom" else "INFO ready\n"))
    if mut != "noserverlog":
        (obs / "server.log").write_text(server)
    dirty = 1 if mut == "dirty" else 0
    launch = "" if mut == "nolaunch" else f"{stamp(LAUNCH)} start1: launching in scope u (RuntimeMaxSec=1)\n"
    gguf, mmproj = (("1.00", "0.50") if mut == "coldcache" else ("16.55", "0.86"))
    precondition = "" if mut == "noprecondition" else (
        f"{stamp(LAUNCH)} precondition {gguf} GiB /home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf\n"
        f"{stamp(LAUNCH)} precondition {mmproj} GiB /home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf\n")
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


EXPECT = {
    "clean": "PASS", "headevidence": "PASS", "vision53": "PASS", "vision63": "PASS",
    "xid": "FAIL", "breach": "FAIL", "oom": "FAIL", "swap": "FAIL", "lowmem": "FAIL", "admission": "FAIL",
    "admissionlate": "FAIL", "nolaunch": "NO_RESULT", "headtoolcall": "NO_RESULT",
    "refused": "FAIL", "norelease": "NO_RESULT", "wrongroot": "NO_RESULT",
    "noprecondition": "NO_RESULT", "coldcache": "NO_RESULT", "zerofiles": "NO_RESULT", "soakfloor": "NO_RESULT",
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
