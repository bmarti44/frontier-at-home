#!/usr/bin/env python3
"""Fixed M1 smoke verdict (DIAGNOSTIC) per PREREG.md.

Usage: verdict.py <evidence dir> [weights manifest]
Writes <dir>/summary.json and prints the verdict. Fails closed: missing or
malformed evidence is NO_RESULT, never FITS. Fidelity is measured and
reported beside the verdict; it does not gate it.
"""
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

PEAK_MAX_GIB, SWAP_MAX_GIB, TOOLCALL_MIN, BUDGET_GIB = 106.0, 0.25, 19, 104.0
MAX_GAP_S = 3  # meminfo samples must cover start 1 at most this far apart
MIN_TOKENS, FIRST_EVENT_MAX = 200, 8  # kit --min-completion-tokens; DFlash block_size
KILL = {
    "kernel-smoke.log": re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I),
    "server": re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid"),
    "memwatch.log": re.compile(r"BREACH|EMERGENCY_STOP|FAIL_CLOSED"),
}
EARLY_STOP = re.compile(r"^early stop: \d+ timestamped tokens, minimum is \d+$")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) ")
FLOAT = r"([0-9]+\.[0-9]+)"


def block_aware(rep):
    """(N - 8) / (tN - t1) from client-counted tokens, or None if the rep is ineligible."""
    if not rep.get("valid"):
        reasons = [r.strip() for r in str(rep.get("error", "")).split(";")]
        if not reasons or not all(EARLY_STOP.match(r) for r in reasons):
            return None
    n, ts = rep.get("client_completion_tokens"), rep.get("token_timestamps_ns") or []
    if not isinstance(n, int) or n < MIN_TOKENS or len(ts) < 2 or ts[-1] <= ts[0]:
        return None
    if rep.get("timing_source") != "sse_content_events" or not rep.get("ttft_s", 0) > 0:
        return None
    return {"decode": (n - FIRST_EVENT_MAX) / ((ts[-1] - ts[0]) / 1e9),
            "prefill": rep["prompt_tokens"] / rep["ttft_s"], "tokens": n, "events": len(ts)}


def epoch(stamp):
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def covered(times, first, last):
    """Samples reach from first to last with no gap above MAX_GAP_S."""
    inside = [epoch(t) for t in times if first <= t <= last]
    if not inside:
        return False
    edges = [epoch(first)] + inside + [epoch(last)]
    return all(b - a <= MAX_GAP_S for a, b in zip(edges, edges[1:]))


def stamped(lines, *needles):
    return [m.group(1) for line in lines
            if all(n in line for n in needles) and (m := STAMP.match(line))]


def fidelity(path):
    text = path.read_text(errors="replace")
    def one(pattern):
        hits = re.findall(pattern, text)
        if len(hits) != 1:
            raise ValueError(f"{pattern!r} matched {len(hits)} times")
        return float(hits[0])
    ppl_a = one(rf"-- A perplexity:\s+{FLOAT}")
    ppl_b = one(rf"-- B perplexity:\s+{FLOAT}")
    agree = re.search(rf"Top-K agreement, A vs B:\s+K = 1: {FLOAT}", text)
    top1_a = re.search(rf"A label in top-K:\s+K = 1: {FLOAT}", text)
    top1_b = re.search(rf"B label in top-K:\s+K = 1: {FLOAT}", text)
    if not (agree and top1_a and top1_b):
        raise ValueError("top-K blocks missing")
    out = {"ppl_pack_a": ppl_a, "ppl_official_b": ppl_b,
           "delta_nll_pack_minus_official": math.log(ppl_a / ppl_b),
           "kld_a_b": one(rf"KL divergence \(A, B\):\s+{FLOAT}"),
           "kld_b_a": one(rf"KL divergence \(B, A\):\s+{FLOAT}"),
           "top1_agreement": float(agree.group(1)),
           "top1_label_pack": float(top1_a.group(1)), "top1_label_official": float(top1_b.group(1))}
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("non-finite fidelity metric")
    return out


def main(ev, weights_manifest):
    obs, missing, c = ev / "observe", [], {}
    driver = (obs / "driver.log").read_text().splitlines() if (obs / "driver.log").is_file() else []
    if not driver:
        missing.append("observe/driver.log")

    hits = []
    for name, rx in KILL.items():
        files = sorted(obs.glob("server-start*.log")) if name == "server" else [obs / name]
        if name == "server" and len(files) != 2:
            missing.append(f"observe/server-start{{1,2}}.log ({len(files)} found)")
        for f in files:
            if not f.is_file():
                missing.append(str(f.relative_to(ev)))
                continue
            hits += [f"{f.name}: {l.strip()}" for l in f.read_text(errors="replace").splitlines() if rx.search(l)]
    c["clause1_kill_lines"] = hits

    rows = []
    for line in (obs / "meminfo.log").read_text().splitlines() if (obs / "meminfo.log").is_file() else []:
        kv = dict(re.findall(r"(\w+):=(\d+)", line))
        m = STAMP.match(line)
        if m and {"MemFree", "MemAvailable", "SwapTotal", "SwapFree"} <= kv.keys():
            rows.append((m.group(1), kv))
    launch, ready = stamped(driver, "start 1: launching in scope"), stamped(driver, "start 1: ready after")
    empty = stamped(driver, "mimo26-smoke-1-", "scope empty after stop")
    if len(rows) < 300:
        missing.append("observe/meminfo.log (fewer than 300 samples)")
    elif len(launch) != 1 or len(ready) != 1 or len(empty) != 1:
        missing.append(f"start 1 launch/ready/empty lines: {len(launch)}/{len(ready)}/{len(empty)}")
    else:
        swap = [(int(r["SwapTotal"]) - int(r["SwapFree"])) / 2**20 for _, r in rows]
        c["clause2_swap_growth_gib"] = max(swap) - swap[0]
        before = [int(r["MemAvailable"]) / 2**20 for t, r in rows if t <= launch[0]]
        load = [int(r["MemAvailable"]) / 2**20 for t, r in rows if launch[0] <= t <= ready[0]]
        run = [int(r["MemAvailable"]) / 2**20 for t, r in rows if launch[0] <= t <= empty[0]]
        free_load = [int(r["MemFree"]) / 2**20 for t, r in rows if launch[0] <= t <= ready[0]]
        if not before or len(load) < 10 or not covered([t for t, _ in rows], launch[0], empty[0]):
            missing.append(f"meminfo samples do not cover start 1 at most {MAX_GAP_S} s apart")
        else:
            c["launch_mem_available_gib"] = before[-1]
            c["clause3_load_peak_gib"] = before[-1] - min(load)
            c["clause3_run_peak_gib"] = before[-1] - min(run)
            c["mem_available_low_gib"] = min(run)
            c["mem_free_low_during_load_gib"] = min(free_load)
        c["meminfo_samples"] = len(rows)

    try:
        doc = json.loads((ev / "kit/toolcall/toolcall.json").read_text())
        cases = doc.get("per_case") or []
        passed = doc["passed"] if doc.get("passed") is not None else sum(bool(x.get("passed")) for x in cases)
        total = doc["total"] if doc.get("total") is not None else len(cases)
        c["clause4_toolcall"] = {"passed": passed, "total": total}
        if total != 20:
            missing.append(f"toolcall total {total} != 20")
    except (OSError, ValueError, KeyError, TypeError) as e:
        missing.append(f"kit/toolcall/toolcall.json ({e})")

    levels = {}
    try:
        speed = json.loads((ev / "kit/speed/speed.json").read_text())
        for cell in speed["cells"]:
            got = [g for g in map(block_aware, cell["reps"]) if g]
            levels[str(cell["ctx_tokens"])] = {
                "eligible_reps": len(got), "reps": len(cell["reps"]), "detail": got,
                "median_decode_block_aware": statistics.median(g["decode"] for g in got) if got else None,
                "median_prefill_lower_bound": statistics.median(g["prefill"] for g in got) if got else None}
    except (OSError, ValueError, KeyError, TypeError) as e:
        missing.append(f"kit/speed/speed.json ({e})")
    c["speed_record_only"] = levels
    ready_lines = [stamped(driver, f"start {n}: ready after") for n in (1, 2)]
    c["clause5_both_ready"] = all(len(r) == 1 for r in ready_lines)
    c["clause5_eligible_rep_ctx0"] = bool((levels.get("0") or {}).get("eligible_reps"))

    for n in (1, 2):
        p = ev / f"probe-start{n}.json"
        try:
            c[f"probe_start{n}"] = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            missing.append(f"{p.name} ({e})")
    if all(f"probe_start{n}" in c for n in (1, 2)):
        a, b = (c[f"probe_start{n}"]["greedy"] for n in (1, 2))
        c["greedy_identity_record_only"] = [x["sha256"] == y["sha256"] for x, y in zip(a, b)]
    if (ev / "observe/cgroup.log").is_file():
        c["cgroup_record_only"] = (ev / "observe/cgroup.log").read_text().splitlines()

    try:
        manifest = json.loads(Path(weights_manifest).read_text())
        files = manifest["artifacts"]["exl3-2.52-mikeroz"]["files"]
        c["pack_weight_gib"] = sum(f["bytes"] for f in files if f["name"].endswith(".safetensors")) / 2**30
    except (OSError, ValueError, KeyError, TypeError) as e:
        missing.append(f"weights manifest ({e})")

    out = {"formula": "PREREG.md", "checks": c}
    try:
        if len(stamped(driver, "fidelity rc=0")) != 1 or stamped(driver, "FIDELITY_FAILED"):
            raise ValueError("the fidelity run did not finish cleanly (driver log)")
        out["fidelity"] = fidelity(ev / "fidelity/model_diff.log")
        out["fidelity_status"] = "MEASURED"
    except (OSError, ValueError) as e:
        out["fidelity"], out["fidelity_status"] = {"error": str(e)}, "NO_RESULT"
    tail = obs / "kernel-tail.log"
    c["kernel_tail_lines_record_only"] = [
        l.strip() for l in (tail.read_text(errors="replace").splitlines() if tail.is_file() else [])
        if KILL["kernel-smoke.log"].search(l)]

    failed = [l for l in driver if "FAILED_ATTEMPT" in l]
    c["failed_attempt_lines"] = failed
    c["restore_failed_lines"] = [l for l in driver if "RESTORE_FAILED" in l]
    misfit = []
    if hits:
        misfit.append("clause 1: kill/OOM/NVRM/Xid line")
    if c.get("clause2_swap_growth_gib", 0) > SWAP_MAX_GIB:
        misfit.append("clause 2: swap growth")
    for key in ("clause3_load_peak_gib", "clause3_run_peak_gib"):
        if c.get(key, 0) > PEAK_MAX_GIB:
            misfit.append(f"clause 3: {key} above {PEAK_MAX_GIB} GiB")
    if misfit:
        verdict = "DOES_NOT_FIT"
        if "clause3_run_peak_gib" in c and "pack_weight_gib" in c:
            out["m2_target_weights_gib"] = BUDGET_GIB - (c["clause3_run_peak_gib"] - c["pack_weight_gib"])
    elif missing or failed or not (c["clause5_both_ready"] and c["clause5_eligible_rep_ctx0"]):
        verdict = "NO_RESULT"
    elif c["clause4_toolcall"]["passed"] < TOOLCALL_MIN:
        verdict = "FITS_TOOLCALL_LOW"
    else:
        verdict = "FITS"
    out.update(missing=missing, does_not_fit=misfit, verdict=verdict)
    (ev / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    speed0 = (levels.get("0") or {}).get("median_decode_block_aware")
    print(json.dumps({"verdict": verdict, "does_not_fit": misfit, "missing": missing,
                      "load_peak_gib": c.get("clause3_load_peak_gib"), "run_peak_gib": c.get("clause3_run_peak_gib"),
                      "toolcall": c.get("clause4_toolcall"), "decode0_block_aware": speed0,
                      "fidelity_status": out["fidelity_status"]}))
    return 0 if verdict in ("FITS", "FITS_TOOLCALL_LOW") else 1


if __name__ == "__main__":
    here = Path(sys.argv[1])
    default = Path(__file__).resolve().parents[3] / "weights/mimo-v2.6-flash/manifest.json"
    sys.exit(main(here, sys.argv[2] if len(sys.argv) > 2 else default))
