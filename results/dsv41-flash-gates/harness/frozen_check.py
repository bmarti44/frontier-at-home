#!/usr/bin/env python3
"""Frozen-input check shared by run_smoke.sh (record) and bundle_attempt.py (verify).

usage: frozen_check.py FROZEN_JSON PHASE   (prints one record; exit 0 ok, 22 mismatch)

A record binds one phase of an attempt to the committed candidate: the HEAD
commit, the actual SHA-256 of every frozen component, and one boolean per
expected check. verify() is the bundler's independent re-derivation (review
round 5, finding 3): it trusts nothing in the records it can recompute. The
commit must exist, every tracked component at that commit must hash to the
recorded value, the recorded map must be complete and equal the frozen-inputs
expectations, and the harness, scorer and frozen file on disk must match.

Candidate 8 (review round 6): each record carries its phase (the file name it
is written to) and recorded_at. verify() takes {file name: (record, mtime)}
and requires the phase to equal the file name, mtime to agree with
recorded_at, and the times to follow the run: the launch check within
LAUNCH_WINDOW_S before started_at, the post-run check after finished_at, and
the post-score check strictly after the post-run check. It also re-hashes every
tracked component and external fixture on disk at bundling time.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = "/home/bmarti44/spark-deepseek-v4-flash"
FIX = ("/home/bmarti44/.cache/ds4-v41-0aaea5a2/gguf-tools/quality-testing/"
       "deepseek-v4.1-flash-20260919-router")
HARNESS = "results/dsv41-flash-gates/harness/"
# Component name -> repository-relative path (tracked) or absolute path (fixture).
TRACKED = {
    "run_smoke.sh": HARNESS + "run_smoke.sh",
    "bundle_attempt.py": HARNESS + "bundle_attempt.py",
    "frozen_check.py": HARNESS + "frozen_check.py",
    "test_score_fidelity.py": HARNESS + "test_score_fidelity.py",
    "test_bundle_attempt.py": HARNESS + "test_bundle_attempt.py",
    "score_fidelity.py": HARNESS + "score_fidelity.py",
    "glm_safe_run.sh": "results/glm52-gates/harness/glm_safe_run.sh",
    "03_memory_guard.py": "scripts/03_memory_guard.py",
}
UNTRACKED = {
    "fixture_manifest.tsv": os.path.join(FIX, "manifest.tsv"),
    "base-default.tsv": os.path.join(FIX, "results/base-default.tsv"),
}
CHECK_KEYS = sorted(list(TRACKED) + list(UNTRACKED) +
                    ["frozen_inputs", "frozen_inputs_committed", "frozen_paths_clean"])
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LAUNCH = "frozen-check.json"
POST_RUN = "frozen-check-post-run.json"
POST_SCORE = "frozen-check-post-score.json"
LAUNCH_WINDOW_S = 60.0   # launch check -> started_at (identity check in between)
POST_WINDOW_S = 600.0    # finished_at -> post-run, post-run -> post-score (scoring)
MTIME_SLACK_S = 5.0


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    try:
        with open(path, "rb") as f:
            return sha(f.read())
    except FileNotFoundError:
        return None


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True)


def expected(frozen_path):
    """Component -> expected hash, from the frozen-inputs file."""
    f = json.load(open(frozen_path))
    want = dict(f["component_sha256"])
    want["score_fidelity.py"] = f["fixed_scorer_sha256"]
    want["fixture_manifest.tsv"] = f["fixture_sha256"]
    want["base-default.tsv"] = f["raw_artifact_sha256"]
    return want


def record(frozen_path, phase):
    rel = os.path.relpath(os.path.realpath(frozen_path), REPO)
    want = expected(frozen_path)
    actual = {n: sha_file(os.path.join(REPO, p)) for n, p in TRACKED.items()}
    actual.update({n: sha_file(p) for n, p in UNTRACKED.items()})
    checks = {n: actual[n] is not None and actual[n] == want.get(n) for n in list(TRACKED) + list(UNTRACKED)}
    actual["frozen_inputs"] = sha_file(frozen_path)
    blob = git("show", "HEAD:" + rel)
    checks["frozen_inputs"] = actual["frozen_inputs"] is not None
    checks["frozen_inputs_committed"] = blob.returncode == 0 and sha(blob.stdout) == actual["frozen_inputs"]
    status = git("status", "--porcelain", "--", rel, *TRACKED.values())
    checks["frozen_paths_clean"] = status.returncode == 0 and status.stdout == b""
    head = git("rev-parse", "HEAD")
    return {"phase": phase, "recorded_at": time.time(), "ok": all(checks.values()), "checks": checks,
            "commit": head.stdout.decode().strip() if head.returncode == 0 else None,
            "frozen_inputs_path": rel, "component_sha256": actual}


def phase_timing(by_name, start, finish):
    """Reasons the records' phases and times do not fit this run."""
    reasons = []
    if start is None or finish is None:
        return ["run start/finish unknown"]
    at = {}
    for name, (r, mtime) in by_name.items():
        t = r.get("recorded_at")
        if r.get("phase") != name:
            reasons.append(f"{name}: phase {r.get('phase')!r} is not its file name")
        if not isinstance(t, (int, float)) or isinstance(t, bool):
            reasons.append(f"{name}: recorded_at missing")
            continue
        if mtime is None or not (t - 1.0 <= mtime <= t + MTIME_SLACK_S):
            reasons.append(f"{name}: file mtime disagrees with recorded_at")
        at[name] = t
    if LAUNCH in at and not (start - LAUNCH_WINDOW_S <= at[LAUNCH] <= start + 1.0):
        reasons.append("launch check is not just before started_at")
    if POST_RUN in at and not (finish - 1.0 <= at[POST_RUN] <= finish + POST_WINDOW_S):
        reasons.append("post-run check is not just after finished_at")
    if POST_SCORE in at and not (POST_RUN in at and at[POST_RUN] < at[POST_SCORE] <= at[POST_RUN] + POST_WINDOW_S):
        reasons.append("post-score check is not just after the post-run check")
    return reasons


def verify(by_name, frozen_path, bundler_path, start=None, finish=None):
    """Return (ok, reasons) for {file name: (record, mtime)} of one attempt."""
    if not by_name or not all(isinstance(v, tuple) and isinstance(v[0], dict) for v in by_name.values()):
        return False, ["missing or malformed record"]
    records = [v[0] for v in by_name.values()]
    reasons = phase_timing(by_name, start, finish)
    for r in records:
        c = r.get("checks")
        if r.get("ok") is not True or not isinstance(c, dict) or sorted(c) != CHECK_KEYS \
                or not all(v is True for v in c.values()):
            reasons.append(f"phase {r.get('phase')!r}: incomplete or failing checks")
    first = records[0]
    commit, amap = first.get("commit"), first.get("component_sha256")
    if any(r.get("commit") != commit or r.get("component_sha256") != amap
           or r.get("frozen_inputs_path") != first.get("frozen_inputs_path") for r in records):
        reasons.append("phases disagree on commit, hashes or frozen-inputs path")
    if not isinstance(amap, dict) or sorted(amap) != sorted(list(TRACKED) + list(UNTRACKED) + ["frozen_inputs"]):
        return False, reasons + ["component map incomplete"]
    want = expected(frozen_path)
    want["frozen_inputs"] = sha_file(frozen_path)
    if amap != want:
        reasons.append("component map differs from the frozen-inputs expectations on disk")
    if not isinstance(commit, str) or not COMMIT_RE.match(commit) \
            or git("cat-file", "-e", commit + "^{commit}").returncode != 0:
        return False, reasons + ["commit does not exist"]
    rel = os.path.relpath(os.path.realpath(frozen_path), REPO)
    if first.get("frozen_inputs_path") != rel:
        reasons.append("frozen-inputs path differs")
    for name, path in list(TRACKED.items()) + [("frozen_inputs", rel)]:
        blob = git("show", f"{commit}:{path}")
        if blob.returncode != 0 or sha(blob.stdout) != amap.get(name):
            reasons.append(f"{name} at {commit} does not match the recorded hash")
    on_disk = {n: os.path.join(REPO, p) for n, p in TRACKED.items()}
    on_disk.update(UNTRACKED)
    on_disk["bundle_attempt.py"] = bundler_path
    on_disk["frozen_check.py"] = os.path.abspath(__file__)
    for name, path in on_disk.items():
        if sha_file(path) != amap.get(name):
            reasons.append(f"{name} on disk differs from the recorded hash")
    return not reasons, reasons


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    rec = record(sys.argv[1], sys.argv[2])
    print(json.dumps(rec, sort_keys=True))
    sys.exit(0 if rec["ok"] else 22)
