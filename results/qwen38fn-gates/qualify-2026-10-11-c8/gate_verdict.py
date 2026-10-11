#!/usr/bin/env python3
"""Fixed candidate-8 gate verdict per PREREG.md in this directory.

Usage: gate_verdict.py <bundle dir>  -> writes <dir>/gate-summary.json, prints the verdict.
Fails closed: missing or malformed evidence is NO_RESULT, never PASS.
The kill-line patterns are imported from the smoke scorer so every run uses
exactly the same ones.
"""
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

# Owner decision 2026-10-11: images on at 1M, kill floor 8 GiB. Candidate 8
# adds the host page-cache release that candidate 7 lacked.
SWAP_MAX_GIB, FLOOR_GIB, TOOLCALL_MIN = 0.25, 8.0, 19
# The profile's safety.minimum_start_gib: the switch's only admission.
ADMISSION_GIB = 115.0
# qwen38-1m MMMU-val-100 (results/qwen38-gates/qualify-2026-09-11-kit-check/vision): 64/100.
DEFAULT_VISION_CORRECT, VISION_N, WORSE_Z = 64, 100, -1.645
MIN_CONTEXT_TOKENS, CONTEXT_CAP = 1_000_000, 1_048_576
KV_CAPACITY = re.compile(r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for 262,144 tokens per request")
HEAD = re.compile(r"^\S+ head ([0-9a-f]{40}) dirty=(\d+)$")
LAUNCH = re.compile(r"^(\S+) start1: launching in scope ")
# The driver primes qwen38-1m's GGUF and mmproj (17.4 GiB) before launch; the
# run tests the fix only if that cache was still resident at launch.
PRECONDITION = re.compile(r"^\S+ precondition ([0-9.]+) GiB (\S+)$")
PRIMED = ("/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
          "/home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf")
PRECONDITION_MIN_GIB = 15.0
# qwen38fn_worker.HostCacheReleaseWorker logs one line before the KV arena.
RELEASE = re.compile(r"host page cache released under (\S+) \((\d+) files\); MemFree ([0-9.]+) GiB, "
                     r"KV arena ([0-9.]+) GiB \+ margin ([0-9.]+) GiB")
REFUSED = "refusing to allocate the KV arena"
MODEL_ROOT = "/home/bmarti44/models"
# A log that exists but never saw the run proves nothing.
MARKERS = {"server.log": "session start", "memwatch.log": "ARMED"}
START = "start1"
AGENT_GATE_TESTS = {"prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"}
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("GATE_REPO") or HERE.parents[2])
CANDIDATE = "results/qwen38fn-gates/qualify-2026-10-11-c8"
# The freeze is the last commit that touched PREREG.md. The run may sit on a
# later commit only if nothing that shapes the measurement differs from it.
FROZEN = (
    "configs/profiles/qwen3.8-flash-next", "configs/build-manifests/qwen38-flash-next-runtime.json",
    "scripts/lib", "scripts/dev/regression-suite.py", "scripts/01_memwatch.sh", "scripts/02_membudget.py",
    "scripts/03_memory_guard.py", "scripts/30_bench_speed.py", "scripts/30_bench_speed_blocks.py",
    "scripts/35_soak.py", "scripts/38_bench_vision.py", "scripts/39_bench_toolcall.py",
    "scripts/50_probe_context.py", "scripts/92_resolve_profile.py", "scripts/93_profile_serve.sh",
    "scripts/94_qualify_profile.py", "fixtures", "evalsets/mmmu-val-100", "evalsets/toolcall",
    "results/qwen38fn-gates/smoke-2026-10-11/probe.py", "results/qwen38fn-gates/smoke-2026-10-11/verdict.py",
    "results/qwen38fn-gates/qualify-2026-10-11-c4/probe-start1.json",
    "results/qwen38fn-gates/qualify-2026-10-11-c7/pagecache_residency.py",
    f"{CANDIDATE}/PREREG.md", f"{CANDIDATE}/run-phase2.sh", f"{CANDIDATE}/gate_verdict.py",
    f"{CANDIDATE}/host-worktree.json",
)
# Candidate 5's text-only greedy probes: the holdout accuracy transfers only
# if turning images on leaves the text path byte-identical.
REFERENCE_PROBE = Path(os.environ.get("GATE_REFERENCE_PROBE")
                       or HERE.parent / "qualify-2026-10-11-c4" / "probe-start1.json")

_spec = importlib.util.spec_from_file_location("smoke_verdict", HERE.parent / "smoke-2026-10-11" / "verdict.py")
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)
KILL = _smoke.KILL


def load(path, missing):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        missing.append(f"{path.name} ({type(error).__name__})")
        return None


def worse_z(correct: int, n: int = VISION_N, reference: int = DEFAULT_VISION_CORRECT) -> float:
    """One-sided two-proportion z of correct/n against reference/n (pooled)."""
    pooled = (correct + reference) / (2 * n)
    se = math.sqrt(pooled * (1 - pooled) * 2 / n)
    return (correct / n - reference / n) / se if se else 0.0


def main(ev):
    missing, fail, checks = [], [], {}
    obs = ev / "observe" / START

    server = obs / "server.log"
    if server.is_file():
        capacity = [int(m.replace(",", "")) for m in KV_CAPACITY.findall(server.read_text(errors="replace"))]
        checks["kv_capacity_tokens"] = capacity[-1] if capacity else None
        if not capacity:
            missing.append("observe/start1/server.log: no KV capacity line at 262,144")
        elif capacity[-1] < CONTEXT_CAP:
            fail.append(f"clause 4: KV capacity {capacity[-1]} < {CONTEXT_CAP}")
        text = server.read_text(errors="replace")
        releases = RELEASE.findall(text)
        checks["host_cache_release"] = [
            {"root": r, "files": int(n), "mem_free_gib": float(f), "arena_gib": float(a), "margin_gib": float(m)}
            for r, n, f, a, m in releases]
        if REFUSED in text:
            fail.append("clause 12: the worker refused the KV arena (MemFree short after the release)")
        elif (len(releases) != 1 or releases[0][0] != MODEL_ROOT or int(releases[0][1]) < 1
              or float(releases[0][3]) < 17.0):
            missing.append(f"clause 12: want one release line under {MODEL_ROOT}, at least one file, "
                           f"for the 17.9 GiB arena, got {releases}")
    for name, marker in MARKERS.items():
        f = obs / name
        if f.is_file() and marker not in f.read_text(errors="replace"):
            missing.append(f"observe/start1/{name}: no '{marker}' line")
    hits = []
    for name, rx in KILL.items():
        f = obs / ("server.log" if name == "server" else name)
        if not f.is_file():
            missing.append(f"observe/start1/{f.name}")
            continue
        hits += [f"{f.name}: {l.strip()}" for l in f.read_text(errors="replace").splitlines() if rx.search(l)]
    checks["kill_lines"] = hits
    if hits:
        fail.append("clause 1: kill/OOM/NVRM/Xid line")

    driver = obs / "driver.log"
    driver_lines = driver.read_text().splitlines() if driver.is_file() else []
    launches = [m.group(1) for m in map(LAUNCH.match, driver_lines) if m]
    checks["launch_stamps"] = launches
    primed = {path: float(gib) for gib, path in (m.groups() for m in map(PRECONDITION.match, driver_lines) if m)}
    checks["precondition_resident_gib"] = primed
    if set(primed) != set(PRIMED) or sum(primed.values()) < PRECONDITION_MIN_GIB:
        missing.append(f"clause 11: qwen38-1m's files not resident at launch ({primed}); the fix was not tested")
    rows = []
    meminfo = obs / "meminfo.log"
    for line in meminfo.read_text().splitlines() if meminfo.is_file() else []:
        kv = dict(re.findall(r"(\w+):=(\d+)", line))
        if {"MemAvailable", "SwapTotal", "SwapFree"} <= kv.keys():
            rows.append((line.split(" ", 1)[0], kv))
    if len(rows) < 600:
        missing.append("observe/start1/meminfo.log (fewer than 600 samples)")
    else:
        used = [(int(r["SwapTotal"]) - int(r["SwapFree"])) / 2**20 for _, r in rows]
        available = [int(r["MemAvailable"]) / 2**20 for _, r in rows]
        checks["swap_growth_gib"] = max(used) - used[0]
        checks["mem_available_low_gib"] = low = min(available)
        if checks["swap_growth_gib"] > SWAP_MAX_GIB:
            fail.append("clause 2: swap growth")
        if low < FLOOR_GIB:
            fail.append("clause 3: external low point below 8 GiB")
        # Peak use counts from the highest reading at or before the launch
        # line: the sampler starts before the start guard waits, so its first
        # row can sit below the memory the engine actually started with.
        before = [a for (t, _), a in zip(rows, available) if len(launches) == 1 and t <= launches[0]]
        if not before:
            missing.append(f"clause 10: no meminfo sample at or before one launch line (launches {launches})")
        else:
            checks["launch_mem_available_gib"] = max(before)
            checks["peak_used_gib"] = peak = max(before) - low
            if peak + FLOOR_GIB > ADMISSION_GIB:
                fail.append("clause 10: peak use + 8 GiB exceeds the 115 GiB switch admission")

    if driver.is_file():
        lines = driver_lines
        checks["failed_attempt_lines"] = [l for l in lines if "FAILED_ATTEMPT" in l]
        heads = [m.groups() for m in map(HEAD.match, lines) if m]
        checks["heads"] = heads
        freeze = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%H", "--",
                                 f"{CANDIDATE}/PREREG.md"], capture_output=True, text=True).stdout.strip()
        checks["freeze"] = freeze
        if len(heads) != 1 or heads[0][1] != "0" or not freeze:
            missing.append(f"code identity: heads {heads}, freeze {freeze or None}")
        else:
            diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, heads[0][0],
                                   "--", *FROZEN], capture_output=True)
            checks["frozen_diff_rc"] = diff.returncode
            if diff.returncode != 0:
                missing.append(f"code identity: frozen files differ between {freeze} and {heads[0][0]}")
    else:
        missing.append("observe/start1/driver.log")
        checks["failed_attempt_lines"] = []

    manifest = load(ev / "manifest.json", missing)
    context = load(ev / "context" / "summary.json", missing)
    if manifest is not None and context is not None:
        cap = (manifest.get("profile") or {}).get("context_cap")
        checks["context"] = {"verdict": context.get("verdict"), "total_tokens": context.get("total_tokens"),
                             "min_mem_available_gib": context.get("min_mem_available_gib"),
                             "context_cap": cap}
        tokens = context.get("total_tokens")
        if cap != CONTEXT_CAP or not isinstance(tokens, int):
            missing.append("context: cap or total_tokens malformed")
        elif context.get("verdict") != "PASS" or tokens < MIN_CONTEXT_TOKENS:
            fail.append("clause 4: direct 1M fill")

    probe = load(ev / f"probe-{START}.json", missing)
    reference = load(REFERENCE_PROBE, missing)
    if probe is not None and reference is not None:
        try:
            a, b = ([g["sha256"] for g in p["greedy"]] for p in (probe, reference))
        except (KeyError, TypeError):
            missing.append("probe greedy rows malformed")
        else:
            checks["text_identity_vs_candidate_5"] = [x == y for x, y in zip(a, b)]
            if len(a) != 3 or len(b) != 3:
                missing.append("probe: expected 3 greedy rows")
            elif a != b:
                fail.append("clause 5: text path differs from candidate 5")

    toolcall = load(ev / "toolcall" / "toolcall.json", missing)
    if toolcall is not None:
        cases = toolcall.get("per_case") or []
        passed = toolcall["passed"] if toolcall.get("passed") is not None else sum(bool(c.get("passed")) for c in cases)
        total = toolcall["total"] if toolcall.get("total") is not None else len(cases)
        checks["toolcall"] = {"passed": passed, "total": total}
        if total != 20:
            missing.append(f"toolcall total {total} != 20")
        elif passed < TOOLCALL_MIN:
            fail.append("clause 6: tool-call")

    agent = load(ev / "agent-gate.json", missing)
    if agent is not None:
        results = {r.get("test"): r.get("pass") for r in agent.get("results") or []}
        checks["agent_gate"] = results
        if set(results) != AGENT_GATE_TESTS:
            missing.append(f"agent-gate tests {sorted(results)}")
        elif not all(v is True for v in results.values()):
            fail.append("clause 7: agent gate")

    soak = load(ev / "soak" / "soak.json", missing)
    if soak is not None:
        checks["soak_pass"] = soak.get("pass")
        if not isinstance(soak.get("pass"), bool):
            missing.append("soak.pass malformed")
        elif not soak["pass"]:
            fail.append("clause 8: soak")

    vision = load(ev / "vision" / "summary.json", missing)
    if vision is not None:
        n, correct, errors = vision.get("n"), vision.get("correct"), vision.get("error_count")
        checks["vision"] = {"n": n, "correct": correct, "invalid_count": vision.get("invalid_count"),
                            "error_count": errors, "default_correct": DEFAULT_VISION_CORRECT}
        if n != VISION_N or type(correct) is not int or type(errors) is not int:
            missing.append(f"vision summary malformed or not {VISION_N} cases")
        else:
            checks["vision"]["z_vs_default"] = z = worse_z(correct)
            if errors or z < WORSE_Z:
                fail.append("clause 9: vision")

    verdict = "FAIL" if fail else ("NO_RESULT" if missing or checks["failed_attempt_lines"] else "PASS")
    out = {"formula": "PREREG.md", "checks": checks, "fail": fail, "missing": missing, "verdict": verdict}
    (ev / "gate-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, "fail": fail, "missing": missing,
                      "mem_low_gib": checks.get("mem_available_low_gib"),
                      "peak_used_gib": checks.get("peak_used_gib"), "context": checks.get("context")}))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
