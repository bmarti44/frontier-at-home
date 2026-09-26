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

Candidate 4 (review round 2): required artifacts must exist and be non-empty
where noted (see REQUIRED); any missing one is a FAIL, never an empty default.
The text arm reconstructs output.txt from the engine-emitted token bytes and
stays PENDING_REVIEW until text-review.json records an explicit
coherent_on_topic verdict. The fidelity arm copies every paired case row into
raw.jsonl. Exit status: 0 PASS, 1 FAIL, 2 PENDING_REVIEW.

Fixed checks (all arms): required artifacts present; exit_code == 0; wrapper reported killed=no; wrapper
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
REQUIRED = {
    "all": ["command.txt", "unit.txt", "started_at.txt", "finished_at.txt", "exit_code.txt",
            "identity.json", "containment/main.log", "containment/cmd.log",
            "containment/samples.log", "containment/kernel.log"],
    "text": ["steps.json"],
    "fidelity": ["cuda.tsv", "fidelity-summary.json"],
    "diag": [],
}
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
    required = REQUIRED["all"] + REQUIRED[arm]
    missing = [n for n in required if not os.path.isfile(os.path.join(d, n))]
    # Freshness: every required artifact must be written at or after the run
    # started (the runner also refuses a non-empty output directory).
    stale = []
    start_txt = read(os.path.join(d, "started_at.txt"))
    if start_txt:
        from datetime import datetime
        start = datetime.fromisoformat(start_txt.strip().replace(",", ".")[:26] + "+00:00").timestamp()
        stale = [n for n in required if n not in missing and n != "started_at.txt"
                 and os.path.getmtime(os.path.join(d, n)) < start - 1.0]
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
    review = None
    if arm == "text":
        sp = os.path.join(d, "steps.json")
        if os.path.exists(sp):
            with open(sp) as f:
                steps = json.load(f)["steps"]
            raw += [{"type": "token", "step": s["step"], "id": s["selected"]["id"],
                     "bytes": s["selected"]["bytes"]} for s in steps]
            text = bytes(b for s in steps for b in s["selected"]["bytes"]).decode("utf-8", "replace")
            with open(os.path.join(d, "output.txt"), "w", encoding="utf-8") as f:
                f.write(text)
        rp = os.path.join(d, "text-review.json")
        if os.path.exists(rp):
            with open(rp) as f:
                review = json.load(f)
    if arm == "fidelity" and os.path.exists(os.path.join(d, "cuda.tsv")):
        import csv
        with open(os.path.join(d, "cuda.tsv"), newline="") as f:
            raw += [dict(r, type="fidelity_case") for r in csv.DictReader(f, delimiter="\t")]
    raw.append({"type": "exit", "exit_code": None if exit_code is None else int(exit_code),
                "wrapper_rc": int(killed.group(1)) if killed else None,
                "wrapper_killed": killed.group(2) if killed else None})

    files = {}
    for name in ("command.txt", "unit.txt", "started_at.txt", "finished_at.txt", "exit_code.txt",
                 "identity.json", "stdout.txt", "stderr.txt", "steps.json", "output.txt",
                 "text-review.json", "cuda.tsv",
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
        "required_artifacts_present": not missing,
        "required_artifacts_fresh": bool(start_txt) and not stale,
        "exit_code_zero": exit_code is not None and int(exit_code) == 0,
        "wrapper_not_killed": bool(killed) and killed.group(2) == "no",
        "no_kernel_fault": not kernel_events,
        "memory_samples_present": bool(mem),
        "mem_low_at_or_above_kill_floor": bool(mem) and min(mem) >= KILL_FLOOR_KIB,
        "identity_verified_pre_and_post": bool(identity) and identity.get("verified") is True,
    }
    metrics = {"missing_artifacts": missing, "stale_artifacts": stale, "mem_avail_low_kib": min(mem) if mem else None, "memory_sample_count": len(mem),
               "kernel_fault_lines": len(kernel_events)}
    if arm == "text":
        checks["steps_at_least_128"] = steps is not None and len(steps) >= MIN_TEXT_STEPS
        metrics["emitted_steps"] = None if steps is None else len(steps)
        metrics["text_review"] = review
    if arm == "fidelity":
        fs = json.loads(read(os.path.join(d, "fidelity-summary.json"), "null") or "null")
        checks["fidelity_verdict_pass"] = bool(fs) and fs.get("verdict") == "PASS"
        metrics["fidelity"] = fs
    verdict = "PASS" if all(checks.values()) else "FAIL"
    if arm == "text" and verdict == "PASS":
        if review is None:
            verdict = "PENDING_REVIEW"
        elif review.get("coherent_on_topic") is not True:
            checks["text_review_coherent_on_topic"] = False
            verdict = "FAIL"
        else:
            checks["text_review_coherent_on_topic"] = True
    summary = {"schema_version": 1, "arm": arm, "derived_post_hoc": derived,
               "kind": manifest["kind"], "checks": checks, "metrics": metrics,
               "verdict": verdict}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(d, "raw.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps({"dir": d, "verdict": summary["verdict"], "checks": checks}, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "PENDING_REVIEW": 2}[summary["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
