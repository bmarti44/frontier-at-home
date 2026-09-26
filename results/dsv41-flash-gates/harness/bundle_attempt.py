#!/usr/bin/env python3
"""Build the AGENTS.md evidence bundle for one DeepSeek V4.1 smoke attempt.

usage: bundle_attempt.py ATTEMPT_DIR ARM [--derived-post-hoc]

Reads only what the runner and containment wrapper wrote into ATTEMPT_DIR
(command.txt, unit.txt, started_at.txt, finished_at.txt, exit_code.txt,
identity.json when present, containment/{main,cmd,samples,kernel}.log, and
steps.json / cuda.tsv / fidelity-summary.json per arm) and writes:

  manifest.json  inputs, identities and configuration, with hashes
  raw.jsonl      one row per memory sample, kernel event, engine log line,
                 emitted token (text arm) and the exit record
  summary.json   fixed checks, unrounded metrics and the verdict

--derived-post-hoc marks bundles rebuilt later from preserved raw logs of
attempts that ran before this bundler existed; their verdicts are recomputed
from the same logs and never replace the original failure records.

Fixed checks (all arms): exit_code == 0; wrapper reported killed=no; wrapper
kernel.log has no NVRM/Xid/OOM line; the lowest MemAvailable sample is at
least the 40 GiB kill floor; binary and model identity verified before and
after the run (identity.json). Text arm: at least 128 engine-emitted steps in
steps.json. Fidelity arm: fidelity-summary.json verdict PASS.
"""
import hashlib
import json
import os
import re
import sys

KILL_FLOOR_KIB = 40 * 1024 * 1024
MIN_TEXT_STEPS = 128
FAULT_RE = re.compile(r"NVRM|Xid|oom-kill|Out of memory|NV_ERR_NO_MEMORY")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path, default=None):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return default


def parse_samples(text):
    rows = []
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        row = {"type": "memory_sample", "ts": parts[0]}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                row[k] = int(v) if v.isdigit() else v
        rows.append(row)
    return rows


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    d, arm = sys.argv[1], sys.argv[2]
    derived = len(sys.argv) == 4 and sys.argv[3] == "--derived-post-hoc"
    if arm not in ("text", "fidelity", "diag"):
        raise SystemExit(f"unknown arm {arm}")
    c = os.path.join(d, "containment")
    main_log = read(os.path.join(c, "main.log"), "")
    cmd_log = read(os.path.join(c, "cmd.log"), "")
    kernel_log = read(os.path.join(c, "kernel.log"), "")
    samples = parse_samples(read(os.path.join(c, "samples.log"), ""))
    exit_code = read(os.path.join(d, "exit_code.txt"))
    identity = json.loads(read(os.path.join(d, "identity.json"), "null") or "null")

    killed = re.search(r"SAFE_RUN end rc=(-?\d+) killed=(\S+)", main_log)
    cgroup = re.search(r"cgroup_verified (.*)", main_log)
    kernel_events = [l for l in kernel_log.splitlines() if FAULT_RE.search(l)]
    mem = [s["mem_avail_kb"] for s in samples if isinstance(s.get("mem_avail_kb"), int)]

    raw = list(samples)
    raw += [{"type": "kernel_event", "line": l} for l in kernel_events]
    raw += [{"type": "engine_log", "line": l} for l in cmd_log.splitlines() if l.startswith("ds4:")]
    steps = None
    if arm == "text":
        sp = os.path.join(d, "steps.json")
        if os.path.exists(sp):
            steps = json.load(open(sp))["steps"]
            raw += [{"type": "token", "step": s["step"], "id": s["selected"]["id"]} for s in steps]
    raw.append({"type": "exit", "exit_code": None if exit_code is None else int(exit_code),
                "wrapper_rc": int(killed.group(1)) if killed else None,
                "wrapper_killed": killed.group(2) if killed else None})

    files = {}
    for name in ("command.txt", "unit.txt", "started_at.txt", "finished_at.txt", "exit_code.txt",
                 "identity.json", "stdout.txt", "stderr.txt", "steps.json", "cuda.tsv",
                 "fidelity-summary.json", "containment/main.log", "containment/cmd.log",
                 "containment/samples.log", "containment/kernel.log"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            files[name] = sha256_file(p)
    manifest = {
        "schema_version": 1,
        "kind": "fidelity_smoke_not_capability" if arm != "diag" else "diagnostic_not_acceptance",
        "arm": arm,
        "derived_post_hoc": derived,
        "unit": (read(os.path.join(d, "unit.txt"), "") or "").strip(),
        "command": (read(os.path.join(d, "command.txt"), "") or "").strip(),
        "started_at": (read(os.path.join(d, "started_at.txt"), "") or "").strip(),
        "finished_at": (read(os.path.join(d, "finished_at.txt"), "") or "").strip(),
        "cgroup": cgroup.group(1) if cgroup else None,
        "kill_floor_kib": KILL_FLOOR_KIB,
        "identity": identity,
        "artifact_sha256": files,
    }
    checks = {
        "exit_code_zero": exit_code is not None and int(exit_code) == 0,
        "wrapper_not_killed": bool(killed) and killed.group(2) == "no",
        "no_kernel_fault": not kernel_events,
        "memory_samples_present": bool(mem),
        "mem_low_at_or_above_kill_floor": bool(mem) and min(mem) >= KILL_FLOOR_KIB,
        "identity_verified_pre_and_post": bool(identity) and identity.get("verified") is True,
    }
    metrics = {"mem_avail_low_kib": min(mem) if mem else None, "memory_sample_count": len(mem),
               "kernel_fault_lines": len(kernel_events)}
    if arm == "text":
        checks["steps_at_least_128"] = steps is not None and len(steps) >= MIN_TEXT_STEPS
        metrics["emitted_steps"] = None if steps is None else len(steps)
    if arm == "fidelity":
        fs = json.loads(read(os.path.join(d, "fidelity-summary.json"), "null") or "null")
        checks["fidelity_verdict_pass"] = bool(fs) and fs.get("verdict") == "PASS"
        metrics["fidelity"] = fs
    summary = {"schema_version": 1, "arm": arm, "derived_post_hoc": derived,
               "kind": manifest["kind"], "checks": checks, "metrics": metrics,
               "verdict": "PASS" if all(checks.values()) else "FAIL"}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(d, "raw.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps({"dir": d, "verdict": summary["verdict"], "checks": checks}, sort_keys=True))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
