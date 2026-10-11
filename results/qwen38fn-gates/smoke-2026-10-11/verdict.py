#!/usr/bin/env python3
"""Fixed Phase-1 smoke verdict (DIAGNOSTIC) per PREREG.md, amendment 1.

Usage: verdict.py <evidence dir>   -> writes <dir>/summary.json, prints the verdict.
Fails closed: missing or malformed evidence is NO_RESULT, never GO.
"""
import json
import re
import statistics
import sys
from pathlib import Path

DECODE_FLOOR, PREFILL_FLOOR, TOOLCALL_MIN, SWAP_MAX_GIB = 17.46, 698.7, 17, 0.25
MIN_TOKENS, FIRST_EVENT_MAX = 200, 4  # kit --min-completion-tokens; 1 + num_speculative_tokens
KILL = {
    "kernel.log": re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I),
    "server": re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid"),
    "memwatch.log": re.compile(r"BREACH|EMERGENCY_STOP"),
}
EARLY_STOP = re.compile(r"^early stop: \d+ timestamped tokens, minimum is \d+$")


def block_aware(rep):
    """(N - 4) / (tN - t1) from client-counted tokens, or None if the rep is ineligible."""
    if not rep.get("valid"):
        reasons = [r.strip() for r in str(rep.get("error", "")).split(";")]
        if not reasons or not all(EARLY_STOP.match(r) for r in reasons):
            return None
    n, ts = rep.get("client_completion_tokens"), rep.get("token_timestamps_ns") or []
    if not isinstance(n, int) or n < MIN_TOKENS or len(ts) < 2 or ts[-1] <= ts[0]:
        return None
    if rep.get("timing_source") != "sse_content_events" or rep.get("ttft_s", 0) <= 0:
        return None
    return {"decode": (n - FIRST_EVENT_MAX) / ((ts[-1] - ts[0]) / 1e9),
            "prefill": rep["prompt_tokens"] / rep["ttft_s"], "tokens": n, "events": len(ts)}


def main(ev):
    obs, missing, out = ev / "observe", [], {"formula": "PREREG.md amendment 1", "checks": {}}
    hits = []
    for name, rx in KILL.items():
        files = sorted(obs.glob("server-*.log")) if name == "server" else [obs / name]
        for f in files:
            if not f.is_file():
                missing.append(str(f.relative_to(ev)))
                continue
            hits += [f"{f.name}: {l.strip()}" for l in f.read_text(errors="replace").splitlines() if rx.search(l)]
    out["checks"]["clause1_kill_lines"] = hits

    rows = []
    for line in (obs / "meminfo.log").read_text().splitlines() if (obs / "meminfo.log").is_file() else []:
        kv = dict(re.findall(r"(\w+):=(\d+)", line))
        if {"MemAvailable", "SwapTotal", "SwapFree"} <= kv.keys():
            rows.append(kv)
    if len(rows) < 60:
        missing.append("observe/meminfo.log (fewer than 60 samples)")
    else:
        used = [(int(r["SwapTotal"]) - int(r["SwapFree"])) / 2**20 for r in rows]
        out["checks"]["clause2_swap_growth_gib"] = max(used) - used[0]
        out["checks"]["mem_available_low_gib"] = min(int(r["MemAvailable"]) for r in rows) / 2**20
        out["checks"]["meminfo_samples"] = len(rows)

    try:
        doc = json.loads((ev / "kit/toolcall/toolcall.json").read_text())
        cases = doc.get("per_case") or []
        passed = doc["passed"] if doc.get("passed") is not None else sum(bool(c.get("passed")) for c in cases)
        total = doc["total"] if doc.get("total") is not None else len(cases)
        out["checks"]["clause3_toolcall"] = {"passed": passed, "total": total}
        if total != 20:
            missing.append(f"toolcall total {total} != 20")
    except (OSError, ValueError, KeyError) as e:
        missing.append(f"kit/toolcall/toolcall.json ({e})")

    levels = {}
    try:
        speed = json.loads((ev / "kit/speed/speed.json").read_text())
        for cell in speed["cells"]:
            got = [g for g in map(block_aware, cell["reps"]) if g]
            levels[str(cell["ctx_tokens"])] = {
                "eligible_reps": len(got), "reps": len(cell["reps"]), "detail": got,
                "median_decode_block_aware": statistics.median(g["decode"] for g in got) if got else None,
                "median_prefill_lower_bound": statistics.median(g["prefill"] for g in got) if got else None,
                "bench_median_decode_nongating": cell.get("median_decode")}
        out["checks"]["bench_suite_valid_nongating"] = speed.get("suite_valid")
    except (OSError, ValueError, KeyError, TypeError) as e:
        missing.append(f"kit/speed/speed.json ({e})")
    out["checks"]["speed"] = levels
    decode = (levels.get("0") or {}).get("median_decode_block_aware")
    prefill = (levels.get("28672") or {}).get("median_prefill_lower_bound")
    if decode is None or prefill is None:
        missing.append("speed: no eligible rep at ctx 0 or 28672")

    for n in (1, 2):
        p = ev / f"probe-start{n}.json"
        if p.is_file():
            out["checks"][f"probe_start{n}"] = json.loads(p.read_text())
        else:
            missing.append(p.name)
    if all(f"probe_start{n}" in out["checks"] for n in (1, 2)):
        a, b = (out["checks"][f"probe_start{n}"]["greedy"] for n in (1, 2))
        out["checks"]["greedy_identity_record_only"] = [x["sha256"] == y["sha256"] for x, y in zip(a, b)]
    failed = [l for l in ((obs / "driver.log").read_text().splitlines() if (obs / "driver.log").is_file() else [])
              if "FAILED_ATTEMPT" in l]
    out["checks"]["failed_attempt_lines"] = failed

    c = out["checks"]
    no_go = []
    if hits:
        no_go.append("clause 1: kill/OOM/NVRM/Xid line")
    if c.get("clause2_swap_growth_gib", 0) > SWAP_MAX_GIB:
        no_go.append("clause 2: swap growth")
    if "clause3_toolcall" in c and c["clause3_toolcall"]["passed"] < TOOLCALL_MIN:
        no_go.append("clause 3: tool-call")
    if decode is not None and prefill is not None and decode < DECODE_FLOOR and prefill < PREFILL_FLOOR:
        no_go.append("clause 4: no better than qwen38-1m on decode or prefill")
    out["missing"], out["no_go"] = missing, no_go
    out["verdict"] = "NO_GO" if no_go else ("NO_RESULT" if missing or failed else "GO")
    (ev / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": out["verdict"], "no_go": no_go, "missing": missing,
                      "decode0_block_aware": decode, "prefill28672_lower_bound": prefill,
                      "toolcall": c.get("clause3_toolcall"), "swap_growth_gib": c.get("clause2_swap_growth_gib"),
                      "mem_low_gib": c.get("mem_available_low_gib")}))
    return 0 if out["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
