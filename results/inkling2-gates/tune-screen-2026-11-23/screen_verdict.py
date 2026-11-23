#!/usr/bin/env python3
"""Inkling 2 Mini lever screen scorer (DIAGNOSTIC; the keep rule is PREREG.md's, fixed by its commit).

  screen_verdict.py <screen dir>                    score every arm, write summary.json, print the verdict
  screen_verdict.py --may-run <screen dir> <arm>    exit 0 only if <arm> is due
  screen_verdict.py --effort-due <screen dir> <arm> exit 0 only if the screen is COMPLETE with <arm> as carrier
  screen_verdict.py --effort-pick <screen dir> <arm> score effort-<arm>, write its pick.json, print the effort
  screen_verdict.py --self-test                     synthetic cases; all must match

Each lever arm is compared with the baseline pair a0 and a0r (a0 repeated at
the end of the window, "repeat_of": "a0"): a gain must clear its threshold
against the better of the two, and a no-regression floor must hold against the
worse. An arm with "retry_of" replaces its original's decision. The combined
arm (named in arms.json with "levers") carries every kept lever. Missing or
malformed evidence makes an arm NO_RESULT, and its lever is not kept.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

KERNEL_KILL = re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I)
SERVER_KILL = re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid")
MEMWATCH_KILL = re.compile(r"BREACH|EMERGENCY_STOP|FAIL_CLOSED")
GIB_KIB = 1024 * 1024
SWAP_GROWTH_MAX_GIB = 0.25
FILL_MARGIN_GIB = 3.0      # a 28K screen fill vs the 1M direct fill: activations only (the arena is preallocated)
PROJECTED_LOW_MIN_GIB = 12.0
TOOLCALL_MIN = 19
K = 8                      # one bench, one decode formula: max tokens per SSE event in every arm
CONCURRENCY_TARGET = 4.0
BATCH_FLAGS = {"batch4096": 4096, "batch8192": 8192}
LEVER_LINES = {
    # lever: server-log needle that must appear exactly when the lever is on
    "max_ctas_pow2": "INKLING2_MAX_CTAS_POW2=1",
    "dspark7": "'method': 'dspark'",
    "moe_humming": "'moe_backend': 'humming'",
}
METRICS_UP = ("decode0", "decode28", "prefill28", "cold_prefill", "code_decode")   # higher is better
METRICS_DOWN = ("ttft0",)                                                          # lower is better


class Bad(Exception):
    """Missing or malformed evidence."""


def ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Bad(f"{path.name}: {exc}") from exc


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def is_single(entry: dict) -> bool:
    return "levers" not in entry and entry.get("lever") is not None


def arm_levers(entry: dict) -> list[str]:
    if "levers" in entry:
        return list(entry["levers"])
    return [] if entry.get("lever") is None else [entry["lever"]]


def memory(arm_dir: Path) -> dict:
    driver = (arm_dir / "driver.log").read_text()
    m = re.search(r"^(\S+) \S+: launching in scope", driver, re.M)
    if not m:
        raise Bad("driver.log has no launching line")
    launch = ts(m.group(1))
    samples = []
    for line in (arm_dir / "meminfo.log").read_text().splitlines():
        a = re.search(r"MemAvailable:=(\d+)", line)
        s = re.search(r"SwapFree:=(\d+)", line)
        if not (a and s):
            continue
        try:
            stamp = ts(line.split()[0])
        except ValueError:
            continue   # a torn line
        samples.append((stamp, int(a.group(1)) / GIB_KIB, int(s.group(1)) / GIB_KIB))
    before = [x for x in samples if x[0] <= launch]
    after = [x for x in samples if x[0] >= launch]
    if not before or len(after) < 10:
        raise Bad("meminfo.log does not cover the launch")
    base_avail, base_swap = before[-1][1], before[-1][2]
    low = min(x[1] for x in after)
    return {"launch_avail_gib": base_avail, "low_avail_gib": low, "peak_use_gib": base_avail - low,
            "swap_growth_gib": max(0.0, base_swap - min(x[2] for x in after)),
            "projected_1m_low_gib": low - FILL_MARGIN_GIB}


def score_arm(arm_dir: Path, entry: dict) -> dict:
    """Facts and safety for one arm; raises Bad on missing/malformed evidence."""
    levers = arm_levers(entry)
    out: dict = {"levers": levers, "unsafe": []}
    if not (arm_dir / "driver.log").exists():
        raise Bad("driver.log missing")
    driver = (arm_dir / "driver.log").read_text()
    m = re.search(r"^\S+ arm \S+ profile (\S+) ", driver, re.M)
    if not m or m.group(1) != entry["profile"]:
        raise Bad(f"driver.log profile {m and m.group(1)} != arms.json {entry['profile']}")
    t = re.search(r"vllm_tree ([0-9a-f]{64}) files=(\d+)", driver)
    if not t:
        raise Bad("driver.log has no vllm_tree line")
    out["vllm_tree"] = t.group(1)
    out["failed_attempt"] = "FAILED_ATTEMPT" in driver
    out["memory"] = memory(arm_dir)
    kernel = (arm_dir / "kernel.log").read_text()
    server = (arm_dir / "server.log").read_text()
    memwatch = (arm_dir / "memwatch.log").read_text()
    if "ARMED" not in memwatch:
        raise Bad("memwatch.log has no ARMED line")
    kills = [l for l in kernel.splitlines() if KERNEL_KILL.search(l)]
    kills += [l for l in server.splitlines() if SERVER_KILL.search(l)]
    kills += [l for l in memwatch.splitlines() if MEMWATCH_KILL.search(l)]
    out["kill_lines"] = kills[:5]
    mem = out["memory"]
    if out["failed_attempt"]:
        out["unsafe"].append("FAILED_ATTEMPT")
    if kills:
        out["unsafe"].append("kill line")
    if mem["swap_growth_gib"] > SWAP_GROWTH_MAX_GIB:
        out["unsafe"].append("swap growth")
    if out["failed_attempt"]:
        if mem["projected_1m_low_gib"] < PROJECTED_LOW_MIN_GIB:
            out["unsafe"].append("projected 1M low")
        return out   # nothing after readiness is required
    c = re.search(r"KV concurrency ([0-9.]+)x at 262,144", driver)
    if not c:
        raise Bad("driver.log has no KV concurrency line")
    out["kv_concurrency"] = conc = float(c.group(1))
    nd = re.search(r"non-default args: (\{.*\})", server)
    if not nd:
        raise Bad("server.log has no non-default args line")
    arena = re.search(r"'kv_cache_memory_bytes': (\d+)", nd.group(1))
    if not arena:
        raise Bad("non-default args carry no kv_cache_memory_bytes")
    out["arena_bytes"] = int(arena.group(1))
    # An arm admitted below 4.0x (DSpark's draft KV shares the arena) is charged the
    # larger arena 4.0x would need, against its projected 1M low.
    floor = entry.get("min_concurrency", CONCURRENCY_TARGET)
    if conc < floor:
        out["unsafe"].append(f"KV concurrency {conc} < {floor}")
    needed = out["arena_bytes"] * max(1.0, CONCURRENCY_TARGET / conc) if conc > 0 else float("inf")
    out["arena_needed_bytes"] = needed
    mem["projected_1m_low_gib"] = mem["low_avail_gib"] - FILL_MARGIN_GIB - (needed - out["arena_bytes"]) / 2**30
    if mem["projected_1m_low_gib"] < PROJECTED_LOW_MIN_GIB:
        out["unsafe"].append("projected 1M low")
    # The startup lines show exactly the arm's levers.
    batch = re.search(r"'max_num_batched_tokens': (\d+)", nd.group(1))
    want_batch = next((BATCH_FLAGS[l] for l in levers if l in BATCH_FLAGS), 2048)
    if not batch or int(batch.group(1)) != want_batch:
        raise Bad(f"max_num_batched_tokens {batch and batch.group(1)} != {want_batch}")
    for lever, needle in LEVER_LINES.items():
        if (needle in server) != (lever in levers):
            raise Bad(f"lever line {needle!r} {'missing' if lever in levers else 'present'}")
    if "INKLING2_MAX_IMAGE_SIDE=4096" not in server:
        raise Bad("patch 0004 startup line missing")
    probe = load_json(arm_dir / "probe.json")
    greedy = probe.get("greedy") if isinstance(probe, dict) else None
    if not (isinstance(greedy, list) and len(greedy) == 3 and all(isinstance(g, dict) for g in greedy)):
        raise Bad("probe.json greedy malformed")
    hashes = [g.get("sha256") for g in greedy]
    if not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise Bad("probe.json greedy hashes malformed")
    out["probe"] = hashes
    cold = load_json(arm_dir / "cold.json")
    rows = cold.get("rows") or []
    if len(rows) != 6 or not all(finite(r.get("prompt_tokens")) for r in rows) \
            or not finite(cold.get("aggregate_prefill_tok_per_s")):
        raise Bad("cold.json incomplete")
    out["cold_prefill"] = cold["aggregate_prefill_tok_per_s"]
    speed = load_json(arm_dir / "speed.json")
    meta = speed.get("metadata") or {}
    if meta.get("max_tokens_per_event") != K or "block-aware" not in str(meta.get("decode_method")):
        raise Bad("speed.json is not the block-aware bench at K=8")
    cells = {c.get("ctx_tokens"): c for c in speed.get("cells") or []}
    if speed.get("suite_valid") is not True or set(cells) != {0, 28672}:
        raise Bad("speed.json invalid or missing a context")
    for ctx, cell in cells.items():
        if cell.get("valid") is not True or not finite(cell.get("median_decode")):
            raise Bad(f"speed cell {ctx} invalid")
    prefill = [r.get("prefill_tok_s") for r in cells[28672].get("reps") or [] if r.get("valid")]
    if len(prefill) < 3 or not all(finite(p) for p in prefill):
        raise Bad("prefill reps at 28,672 missing")
    out["decode0"] = cells[0]["median_decode"]
    out["decode28"] = cells[28672]["median_decode"]
    out["prefill28"] = statistics.median(prefill)
    out["ttft0"] = cells[0].get("median_ttft")
    if not finite(out["ttft0"]):
        raise Bad("speed cell 0 has no median TTFT")
    code = load_json(arm_dir / "code.json")
    if code.get("valid") is not True or code.get("max_tokens_per_event") != K \
            or not finite(code.get("median_decode_tok_s")):
        raise Bad("code.json invalid")
    out["code_decode"] = code["median_decode_tok_s"]
    tool = load_json(arm_dir / "toolcall.json")
    if not isinstance(tool.get("passed"), int) or tool.get("total") != 20:
        raise Bad("toolcall.json malformed")
    out["toolcall"] = tool["passed"]
    if tool["passed"] < TOOLCALL_MIN:
        out["unsafe"].append("tool-call < 19/20")
    if "scope empty after stop" not in driver:
        out["unsafe"].append("scope not empty after stop")
    return out


def baseline(bases: list[dict]) -> tuple[dict, dict]:
    """(best, worst) per metric over the baseline starts."""
    best, worst = {}, {}
    for m in METRICS_UP:
        best[m], worst[m] = max(b[m] for b in bases), min(b[m] for b in bases)
    for m in METRICS_DOWN:
        best[m], worst[m] = min(b[m] for b in bases), max(b[m] for b in bases)
    return best, worst


def decide(lever: str, a: dict, best: dict, worst: dict, probe0: list[str]) -> tuple[bool, str]:
    """PREREG keep rule for one lever arm: gains against the best baseline start,
    no-regression floors against the worst."""
    if a["unsafe"]:
        return False, "unsafe: " + ", ".join(a["unsafe"])
    g = {m: a[m] / best[m] for m in METRICS_UP}          # gain ratios (>1 is faster)
    r = {m: a[m] / worst[m] for m in METRICS_UP}         # regression ratios
    rt0 = a["ttft0"] / best["ttft0"]                     # <1 is faster than the best baseline TTFT
    geo = math.sqrt(g["decode0"] * g["decode28"])
    worst_decode = min(r["decode0"], r["decode28"])
    same = a["probe"] == probe0
    a["ratios"] = {"gain": g, "floor": r, "rt0": rt0, "geo": geo, "min_decode_floor": worst_decode,
                   "probe_identical": same}
    if lever in BATCH_FLAGS:
        ok = g["prefill28"] >= 1.03 and r["cold_prefill"] >= 1.0 and worst_decode >= 0.98
        return ok, (f"rp {g['prefill28']:.4f} (>=1.03), rc floor {r['cold_prefill']:.4f} (>=1.0), "
                    f"min decode {worst_decode:.4f} (>=0.98)")
    if lever == "max_ctas_pow2":
        if not same:
            return False, "greedy probes differ from a0 (must be byte-identical)"
        ok = (g["cold_prefill"] >= 1.03 or rt0 <= 0.97) and worst_decode >= 0.98 and r["prefill28"] >= 0.97
        return ok, (f"rc {g['cold_prefill']:.4f} (>=1.03) or rt0 {rt0:.4f} (<=0.97); "
                    f"min decode {worst_decode:.4f}; rp floor {r['prefill28']:.4f}")
    if lever == "dspark7":
        floors = worst_decode >= 0.98 and r["prefill28"] >= 0.95
        ok = floors and (geo >= 1.03 or g["code_decode"] >= 1.25)
        return ok, (f"geo {geo:.4f} (>=1.03) or code {g['code_decode']:.4f} (>=1.25); "
                    f"min decode {worst_decode:.4f}; rp floor {r['prefill28']:.4f}")
    if lever == "moe_humming":
        if not same:
            return False, "greedy probes differ from a0 (not byte-identical; the fidelity gate is not spent)"
        ok = (geo >= 1.03 or g["prefill28"] >= 1.03) and worst_decode >= 0.98 and r["prefill28"] >= 0.95
        return ok, f"geo {geo:.4f} or rp {g['prefill28']:.4f} (>=1.03); min decode {worst_decode:.4f}"
    raise ValueError(lever)


def score(ev: Path) -> dict:
    arms = load_json(ev / "arms.json")
    result: dict = {"arms": {}, "kept": [], "dropped": {}, "no_result": {}, "decided_by": {}}

    def fail(why: str) -> dict:
        return {**result, "verdict": "NO_RESULT", "why": why}

    try:
        base = score_arm(ev / "a0", arms["a0"])
    except (Bad, OSError) as exc:
        return fail(f"a0: {exc}")
    result["arms"]["a0"] = base
    if base["unsafe"]:
        return fail("a0 unsafe: " + ", ".join(base["unsafe"]))
    tree0 = base["vllm_tree"]
    repeats = [a for a in arms if arms[a].get("repeat_of") == "a0"]
    bases = [base]
    pending = []
    for arm in repeats:
        if not (ev / arm).exists():
            pending.append(arm)
            continue
        try:
            rep = score_arm(ev / arm, arms[arm])
        except (Bad, OSError) as exc:
            return fail(f"{arm}: {exc}")
        result["arms"][arm] = rep
        if rep["unsafe"]:
            return fail(f"{arm} unsafe: " + ", ".join(rep["unsafe"]))
        if rep["vllm_tree"] != tree0:
            return fail(f"{arm} runtime tree differs from a0's")
        bases.append(rep)
    best, worst = baseline(bases)
    result["baseline"] = {"best": best, "worst": worst, "starts": ["a0"] + [a for a in repeats if a not in pending],
                          # reported: a repeat that differs shows greedy identity is not run-to-run stable
                          "repeat_probes_identical": [b["probe"] == base["probe"] for b in bases[1:]]}

    # Single-lever arms: originals first, then retries; a retry replaces its original's decision.
    singles = [a for a in sorted(arms) if is_single(arms[a])]
    singles.sort(key=lambda a: (arms[a].get("retry_of") is not None, a))
    decisions: dict[str, tuple[str, str, str]] = {}
    for arm in singles:
        entry = arms[arm]
        lever = entry["lever"]
        if entry.get("retry_of") and arms.get(entry["retry_of"], {}).get("lever") != lever:
            return fail(f"{arm} retries {entry.get('retry_of')} with a different lever")
        if not (ev / arm).exists():
            decisions[lever] = (arm, "not_run", "not run")
            continue
        try:
            facts = score_arm(ev / arm, entry)
            want_tree = entry.get("vllm_tree", tree0)
            if facts["vllm_tree"] != want_tree:
                raise Bad(f"runtime tree {facts['vllm_tree'][:12]} != expected {want_tree[:12]}")
        except (Bad, OSError) as exc:
            decisions[lever] = (arm, "no_result", str(exc))
            continue
        keep, why = decide(lever, facts, best, worst, base["probe"])
        facts["keep"], facts["why"] = keep, why
        result["arms"][arm] = facts
        decisions[lever] = (arm, "kept" if keep else "dropped", why)
    for lever, (arm, outcome, why) in decisions.items():
        result["decided_by"][lever] = arm
        if outcome == "kept":
            result["kept"].append(lever)
        elif outcome == "dropped":
            result["dropped"][lever] = why
        else:
            result["no_result"][lever] = why
    batches = [l for l in result["kept"] if l in BATCH_FLAGS]
    if len(batches) > 1:   # one batch size: the larger prefill gain
        gain = lambda l: result["arms"][decisions[l][0]]["ratios"]["gain"]["prefill28"]  # noqa: E731
        best_batch = max(batches, key=gain)
        for l in batches:
            if l != best_batch:
                result["kept"].remove(l)
                result["dropped"][l] = f"kept, but {best_batch} gained more prefill"
    untested = sorted(l for l, (a, o, w) in decisions.items() if o in ("not_run", "no_result"))
    result["untested_levers"] = untested
    if pending:
        result["carrier"] = None
        return {**result, "verdict": "INCOMPLETE", "why": f"baseline repeat not run: {pending}"}

    combined = [a for a in arms if "levers" in arms[a]]
    kept_arms = {l: decisions[l][0] for l in result["kept"]}
    if len(result["kept"]) <= 1:
        result["carrier"] = next(iter(kept_arms.values()), "a0")
        verdict = "COMPLETE"
    elif not combined:
        result["carrier"], verdict = None, "INCOMPLETE"   # the combined arm still has to be authored and run
    else:
        arm = combined[-1]
        if sorted(arms[arm]["levers"]) != sorted(result["kept"]):
            return fail(f"{arm} levers {arms[arm]['levers']} != kept {result['kept']}")
        if not (ev / arm).exists():
            result["carrier"], verdict = None, "INCOMPLETE"
        else:
            try:
                facts = score_arm(ev / arm, arms[arm])
                if facts["vllm_tree"] != arms[arm].get("vllm_tree", tree0):
                    raise Bad("runtime tree differs from the expected one")
            except (Bad, OSError) as exc:
                return fail(f"{arm}: {exc}")
            ok = not facts["unsafe"]
            if ok:
                geo = math.sqrt(facts["decode0"] / best["decode0"] * facts["decode28"] / best["decode28"])
                rp = facts["prefill28"] / worst["prefill28"]
                facts["ratios"] = {"geo_vs_best": geo, "rp_floor": rp}
                ok = geo >= 1.0 and rp >= 0.97
            facts["keep"] = ok
            result["arms"][arm] = facts
            if ok:
                result["carrier"] = arm
            else:   # the kept single-lever arm with the best decode geomean
                result["carrier"] = max(kept_arms.values(), key=lambda a: result["arms"][a]["ratios"]["geo"])
            verdict = "COMPLETE"
    if untested and verdict == "COMPLETE":
        verdict = "INCOMPLETE"
    result["verdict"] = verdict
    return result


def may_run(ev: Path, arm: str) -> bool:
    arms = load_json(ev / "arms.json")
    if arm not in arms or (ev / arm).exists():
        return False
    if arm == "a0":
        return True
    if not (ev / "a0").exists():
        return False
    entry = arms[arm]
    if entry.get("retry_of") and not (ev / entry["retry_of"]).exists():
        return False
    if "levers" in entry:
        result = score(ev)
        singles = [a for a in arms if is_single(arms[a])]
        return all((ev / a).exists() for a in singles) and sorted(entry["levers"]) == sorted(result["kept"]) \
            and len(result["kept"]) > 1
    return True


def effort_due(ev: Path, arm: str) -> bool:
    if (ev / f"effort-{arm}").exists():
        return False
    result = score(ev)
    return result["verdict"] == "COMPLETE" and result.get("carrier") == arm


DEV_ROWS = {"gsm8k": 100, "mmlu-pro": 253}
EFFORT_MARGIN = 4          # medium must beat low by >= 4 of 353 dev items (~1%); otherwise low


def effort_pick(ev: Path, arm: str) -> dict:
    """PREREG effort rule over effort-<arm>/{low,medium}/acc-<suite>.json."""
    obs = ev / f"effort-{arm}"
    driver = (obs / "driver.log").read_text() if (obs / "driver.log").exists() else ""
    out: dict = {"carrier": arm, "sums": {}, "rows": {}}
    try:
        if "FAILED_ATTEMPT" in driver:
            raise Bad("driver.log has FAILED_ATTEMPT")
        if "scope empty after stop" not in driver:
            raise Bad("driver.log has no clean stop")
        for effort in ("low", "medium"):
            total = 0
            for suite, n in DEV_ROWS.items():
                if f"started {suite} dev at effort {effort} " not in driver:
                    raise Bad(f"driver.log never started {suite} at {effort}")
                acc = load_json(obs / effort / f"acc-{suite}.json")
                gen = acc.get("generation") or {}
                if (acc.get("suite"), acc.get("split"), acc.get("n")) != (suite, "dev", n) \
                        or gen.get("encoder") != "inkling2" or gen.get("max_tokens") != 16384 \
                        or not str(acc.get("transcript_dir", "")).endswith(f"/{effort}/transcripts/{suite}") \
                        or not isinstance(acc.get("correct"), int) or not 0 <= acc["correct"] <= n:
                    raise Bad(f"{effort}/acc-{suite}.json is not the declared dev run")
                out["rows"][f"{effort}/{suite}"] = {"correct": acc["correct"], "n": n,
                                                    "invalid": acc.get("invalid_count")}
                total += acc["correct"]
            out["sums"][effort] = total
    except (Bad, OSError) as exc:
        return {**out, "verdict": "NO_RESULT", "why": str(exc), "effort": None}
    margin = out["sums"]["medium"] - out["sums"]["low"]
    out["margin"] = margin
    out["effort"] = "medium" if margin >= EFFORT_MARGIN else "low"
    out["verdict"] = "COMPLETE"
    return out


def _write_effort(root: Path, arm: str, low: tuple[int, int], medium: tuple[int, int], **kw) -> Path:
    obs = root / f"effort-{arm}"
    obs.mkdir(parents=True)
    lines = []
    for effort, counts in (("low", low), ("medium", medium)):
        for (suite, n), correct in zip(DEV_ROWS.items(), counts):
            lines.append(f"2026-11-23T08:00:00Z started {suite} dev at effort {effort} (pid 1)")
            if kw.get("missing") == f"{effort}/{suite}":
                continue
            (obs / effort).mkdir(exist_ok=True)
            (obs / effort / f"acc-{suite}.json").write_text(json.dumps({
                "suite": suite, "split": kw.get("split", "dev"), "n": kw.get("n", {}).get(suite, n),
                "correct": correct, "invalid_count": 0,
                "transcript_dir": f"/x/effort-{arm}/{effort}/transcripts/{suite}",
                "generation": {"encoder": "inkling2", "max_tokens": 16384}}))
    if kw.get("failed"):
        lines.append("2026-11-23T09:00:00Z FAILED_ATTEMPT a0: kernel kill line during the accuracy suites")
    lines.append(f"2026-11-23T10:00:00Z effort-{arm}: scope empty after stop")
    (obs / "driver.log").write_text("\n".join(lines) + "\n")
    return root


# --------------------------------------------------------------------------- self-test

TREE = "a" * 64   # synthetic digests (real ones stay out of the source: lint_secrets)
SHA = [c * 64 for c in "bcd"]


def _write_arm(root: Path, arm: str, levers: list[str], profile: str, **kw) -> None:
    d = root / arm
    d.mkdir(parents=True)
    batch = next((BATCH_FLAGS[l] for l in levers if l in BATCH_FLAGS), 2048)
    driver = [f"2026-11-23T05:00:00Z arm {arm} profile {kw.get('logged_profile', profile)} stack s tokens_per_event 8",
              f"2026-11-23T05:00:00Z vllm_tree {kw.get('tree', TREE)} files=6525",
              f"2026-11-23T05:00:01Z {arm}: launching in scope u"]
    if kw.get("failed"):
        driver.append(f"2026-11-23T05:03:00Z FAILED_ATTEMPT {arm}: launch did not reach ready")
    else:
        driver.append(f"2026-11-23T05:04:00Z {arm}: KV concurrency {kw.get('conc', 4.07)}x at 262,144 per request")
        driver.append(f"2026-11-23T05:30:00Z {arm}: scope empty after stop")
    (d / "driver.log").write_text("\n".join(driver) + "\n")
    lines = []
    low = kw.get("low", 25.0)
    for i in range(40):
        avail = 115.5 if i < 2 else (low if i == 20 else 40.0)
        swap = 16.0 - (kw.get("swap", 0.0) if i == 30 else 0.0)
        lines.append(f"2026-11-23T05:00:{i:02d}Z MemAvailable:={int(avail * GIB_KIB)} SwapFree:={int(swap * GIB_KIB)} ")
    (d / "meminfo.log").write_text("\n".join(lines) + "\n")
    (d / "kernel.log").write_text(kw.get("kernel", ""))
    (d / "memwatch.log").write_text("ts=x ARMED target_pid=1\n" + kw.get("memwatch", ""))
    nd = {"kv_cache_memory_bytes": 68000000000, "max_num_batched_tokens": batch}
    if "moe_humming" in levers:
        nd["moe_backend"] = "humming"
    if "dspark7" in levers:
        nd["speculative_config"] = {"method": "dspark"}
    server = [f"INFO non-default args: {nd!r}", "INFO INKLING2_MAX_IMAGE_SIDE=4096: encoder budget"]
    if "max_ctas_pow2" in levers:
        server.append("INFO INKLING2_MAX_CTAS_POW2=1: max_ctas bucketed")
    if kw.get("drop_line"):
        server = [l for l in server if kw["drop_line"] not in l]
    (d / "server.log").write_text("\n".join(server) + "\n" + kw.get("server", ""))
    if kw.get("failed"):
        return
    s = kw.get("s", {})
    probe = kw.get("probe", SHA)
    (d / "probe.json").write_text(json.dumps({"greedy": [{"prompt": "p", "sha256": h, "completion_tokens": 300,
                                                          "text": "t"} for h in probe], "granularity": []}))
    if not kw.get("no_cold"):
        (d / "cold.json").write_text(json.dumps({"rows": [{"prompt_tokens": 1000}] * 6,
                                                 "aggregate_prefill_tok_per_s": 2000 * s.get("rc", 1.0)}))
    reps = [{"valid": True, "prefill_tok_s": 2700 * s.get("rp", 1.0)} for _ in range(3)]
    if kw.get("null_prefill"):
        reps[0]["prefill_tok_s"] = None
    (d / "speed.json").write_text(json.dumps({
        "metadata": {"max_tokens_per_event": kw.get("k", 8), "decode_method": "block-aware: ..."},
        "suite_valid": True, "cells": [
            {"ctx_tokens": 0, "valid": True, "median_decode": 34.0 * s.get("r0", 1.0),
             "median_ttft": 0.5 * s.get("rt0", 1.0), "reps": reps},
            {"ctx_tokens": 28672, "valid": True, "median_decode": 29.0 * s.get("r28", 1.0), "median_ttft": 11.0,
             "reps": reps}]}))
    (d / "code.json").write_text(json.dumps({"valid": not kw.get("code_invalid"), "max_tokens_per_event": 8,
                                             "median_decode_tok_s": 34.0 * s.get("rcode", 1.0)}))
    (d / "toolcall.json").write_text(json.dumps({"passed": kw.get("tool", 20), "total": 20}))


SINGLE = {"a1": "batch4096", "a2": "batch8192", "a3": "max_ctas_pow2", "a4": "dspark7", "a5": "moe_humming"}


def _screen(root: Path, arms_extra=None, **per_arm) -> Path:
    arms = {"a0": {"profile": "p/a0", "lever": None, "base": None}}
    arms.update({a: {"profile": f"p/{a}", "lever": l, "base": "a0"} for a, l in SINGLE.items()})
    arms["a0r"] = {"profile": "p/a0", "lever": None, "base": None, "repeat_of": "a0"}
    arms.update(arms_extra or {})
    root.mkdir()
    (root / "arms.json").write_text(json.dumps(arms))
    for a, entry in arms.items():
        if per_arm.get(a, {}).get("skip"):
            continue
        _write_arm(root, a, arm_levers(entry), entry["profile"], **per_arm.get(a, {}))
    return root


def self_test() -> int:
    gain = {"s": {"r0": 1.05, "r28": 1.05, "rp": 1.05, "rc": 1.05}}
    cases = []

    def case(name, check, **kw):
        cases.append((name, check, kw))

    case("nothing gains: COMPLETE, a0 carrier, nothing kept",
         lambda r: r["verdict"] == "COMPLETE" and r["kept"] == [] and r["carrier"] == "a0")
    case("one lever gains: carrier is that arm",
         lambda r: r["kept"] == ["dspark7"] and r["carrier"] == "a4", a4=gain)
    case("two levers kept without a combined arm: INCOMPLETE",
         lambda r: r["verdict"] == "INCOMPLETE" and set(r["kept"]) == {"dspark7", "max_ctas_pow2"},
         a3={"s": {"rc": 1.05}}, a4=gain)
    case("combined arm scored: carrier a6",
         lambda r: r["verdict"] == "COMPLETE" and r["carrier"] == "a6",
         extra={"a6": {"profile": "p/a6", "levers": ["max_ctas_pow2", "dspark7"]}},
         a3={"s": {"rc": 1.05}}, a4=gain, a6=gain)
    case("combined arm slower than a0: best kept single lever carries",
         lambda r: r["carrier"] == "a4",
         extra={"a6": {"profile": "p/a6", "levers": ["max_ctas_pow2", "dspark7"]}},
         a3={"s": {"rc": 1.05}}, a4=gain, a6={"s": {"r0": 0.9, "r28": 0.9}})
    case("combined arm levers disagree with decisions: NO_RESULT",
         lambda r: r["verdict"] == "NO_RESULT",
         extra={"a6": {"profile": "p/a6", "levers": ["moe_humming", "dspark7"]}},
         a3={"s": {"rc": 1.05}}, a4=gain, a6=gain)
    case("a0 unsafe: NO_RESULT", lambda r: r["verdict"] == "NO_RESULT", a0={"kernel": "NVRM: Xid 31\n"})
    case("a0 FAILED_ATTEMPT: NO_RESULT", lambda r: r["verdict"] == "NO_RESULT", a0={"failed": True})
    case("a0r unsafe: NO_RESULT", lambda r: r["verdict"] == "NO_RESULT", a0r={"memwatch": "FAIL_CLOSED\n"})
    case("a0r not run: INCOMPLETE, no carrier", lambda r: r["verdict"] == "INCOMPLETE" and r["carrier"] is None,
         a0r={"skip": True}, a4=gain)
    case("a0r faster: a 5% gain against a 4%-faster a0r is not kept",
         lambda r: "dspark7" in r["dropped"], a4=gain, a0r={"s": {"r0": 1.04, "r28": 1.04}})
    case("a0r slower: decode floors use the slower start, so a 1.5% dip vs a0 passes",
         lambda r: "dspark7" in r["kept"], a4={"s": {"rcode": 1.5, "r0": 0.985}}, a0r={"s": {"r0": 0.99}})
    case("kill line drops a gaining lever", lambda r: "dspark7" in r["dropped"],
         a4={**gain, "kernel": "Out of memory: Killed process\n"})
    case("server OOM drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "server": "torch.OutOfMemoryError\n"})
    case("memwatch breach drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "memwatch": "BREACH\n"})
    case("swap growth drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "swap": 0.5})
    case("projected 1M low drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "low": 14.0})
    case("KV concurrency < 4 drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "conc": 3.9})
    case("DSpark admitted at 3.8x with min 3.0: kept, charged the larger arena",
         lambda r: "dspark7" in r["kept"] and r["arms"]["a4"]["arena_needed_bytes"] > 68e9,
         extra={"a4": {"profile": "p/a4", "lever": "dspark7", "base": "a0", "min_concurrency": 3.0}},
         a4={**gain, "conc": 3.8})
    case("DSpark at 3.0x: the larger arena breaks the projected low",
         lambda r: "dspark7" in r["dropped"],
         extra={"a4": {"profile": "p/a4", "lever": "dspark7", "base": "a0", "min_concurrency": 3.0}},
         a4={**gain, "conc": 3.0, "low": 25.0})
    case("tool-call 18/20 drops", lambda r: "dspark7" in r["dropped"], a4={**gain, "tool": 18})
    case("readiness failure drops (no later files needed)", lambda r: "dspark7" in r["dropped"],
         a4={"failed": True})
    case("pow2 with different probes drops", lambda r: "max_ctas_pow2" in r["dropped"],
         a3={"s": {"rc": 1.2}, "probe": [SHA[1], SHA[0], SHA[2]]})
    case("pow2 kept on TTFT@0 alone", lambda r: "max_ctas_pow2" in r["kept"], a3={"s": {"rt0": 0.9}})
    case("humming with different probes drops", lambda r: "moe_humming" in r["dropped"],
         a5={**gain, "probe": [SHA[1], SHA[0], SHA[2]]})
    case("dspark kept on code alone", lambda r: "dspark7" in r["kept"], a4={"s": {"rcode": 1.5}})
    case("dspark prose regression drops even with code gain", lambda r: "dspark7" in r["dropped"],
         a4={"s": {"rcode": 2.0, "r28": 0.95}})
    case("batch needs 3% prefill", lambda r: "batch4096" in r["dropped"], a1={"s": {"rp": 1.02}})
    case("two batches kept: the larger prefill gain stays", lambda r: "batch8192" in r["kept"]
         and "batch4096" not in r["kept"], a1={"s": {"rp": 1.04}}, a2={"s": {"rp": 1.08}})
    case("missing cold.json: NO_RESULT lever, INCOMPLETE", lambda r: "dspark7" in r["no_result"]
         and r["verdict"] == "INCOMPLETE", a4={**gain, "no_cold": True})
    case("null prefill rep: NO_RESULT lever", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "null_prefill": True})
    case("invalid code.json: NO_RESULT lever", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "code_invalid": True})
    case("speed bench at another K: NO_RESULT lever", lambda r: "dspark7" in r["no_result"], a4={**gain, "k": 1})
    case("driver profile differs from arms.json: NO_RESULT lever", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "logged_profile": "p/a5"})
    case("runtime tree differs from a0's: NO_RESULT lever", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "tree": "f" * 64})
    case("missing lever line: NO_RESULT lever", lambda r: "max_ctas_pow2" in r["no_result"],
         a3={"s": {"rc": 1.05}, "drop_line": "INKLING2_MAX_CTAS_POW2"})
    case("missing patch 0004 line: NO_RESULT", lambda r: "dspark7" in r["no_result"],
         a4={**gain, "drop_line": "INKLING2_MAX_IMAGE_SIDE"})
    case("arm not run: INCOMPLETE", lambda r: r["verdict"] == "INCOMPLETE" and "moe_humming" in r["untested_levers"],
         a5={"skip": True})
    case("a4 failed, a4b (retry, patched tree) gains: carrier a4b, kept once, not dropped",
         lambda r: r["verdict"] == "COMPLETE" and r["carrier"] == "a4b" and r["kept"] == ["dspark7"]
         and "dspark7" not in r["dropped"],
         extra={"a4b": {"profile": "p/a4b", "lever": "dspark7", "base": "a0", "retry_of": "a4", "vllm_tree": "e" * 64}},
         a4={"failed": True}, a4b={**gain, "tree": "e" * 64})
    case("a3 NO_RESULT, a3b retry dropped: lever decided, COMPLETE",
         lambda r: r["verdict"] == "COMPLETE" and "max_ctas_pow2" in r["dropped"] and not r["untested_levers"],
         extra={"a3b": {"profile": "p/a3b", "lever": "max_ctas_pow2", "base": "a0", "retry_of": "a3"}},
         a3={"no_cold": True})
    case("a1 failed, a1b kept with a2 kept: tie-break over kept arms, no crash",
         lambda r: r["kept"] == ["batch8192"] or r["kept"] == ["batch4096"],
         extra={"a1b": {"profile": "p/a1b", "lever": "batch4096", "base": "a0", "retry_of": "a1"}},
         a1={"failed": True}, a1b={"s": {"rp": 1.04}}, a2={"s": {"rp": 1.08}})
    case("declared retry not run: INCOMPLETE",
         lambda r: r["verdict"] == "INCOMPLETE" and "dspark7" in r["untested_levers"],
         extra={"a4b": {"profile": "p/a4b", "lever": "dspark7", "base": "a0", "retry_of": "a4"}},
         a4={"failed": True}, a4b={"skip": True})

    failures = 0
    for name, check, kw in cases:
        tmp = Path(tempfile.mkdtemp(prefix="inkling2-screen-"))
        try:
            extra = kw.pop("extra", None)
            ev = _screen(tmp / "s", extra, **kw)
            result = score(ev)
            ok = bool(check(result))
        except Exception as exc:  # a crash is a failed case
            ok, result = False, {"crash": repr(exc)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": {json.dumps(result, default=str)[:600]}"))
    # --may-run / --effort-due
    tmp = Path(tempfile.mkdtemp(prefix="inkling2-screen-"))
    try:
        root = tmp / "m"
        root.mkdir()
        arms = {"a0": {"profile": "p/a0", "lever": None, "base": None},
                "a4": {"profile": "p/a4", "lever": "dspark7", "base": "a0"},
                "a4b": {"profile": "p/a4b", "lever": "dspark7", "base": "a0", "retry_of": "a4"},
                "a0r": {"profile": "p/a0", "lever": None, "base": None, "repeat_of": "a0"}}
        (root / "arms.json").write_text(json.dumps(arms))
        checks = [may_run(root, "a0"), not may_run(root, "a4"), not may_run(root, "zz")]
        _write_arm(root, "a0", [], "p/a0")
        checks += [not may_run(root, "a0"), may_run(root, "a4"), not may_run(root, "a4b"), may_run(root, "a0r")]
        _write_arm(root, "a4", ["dspark7"], "p/a4", failed=True)
        checks += [may_run(root, "a4b")]
        _write_arm(root, "a4b", ["dspark7"], "p/a4b", s={"r0": 1.05, "r28": 1.05})
        checks += [not effort_due(root, "a4b")]          # a0r not run yet
        _write_arm(root, "a0r", [], "p/a0")
        checks += [effort_due(root, "a4b"), not effort_due(root, "a4"), not effort_due(root, "a0")]
        (root / "effort-a4b").mkdir()
        checks += [not effort_due(root, "a4b")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(checks)
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} --may-run and --effort-due admit only due runs {checks}")
    effort_cases = [
        ("medium +4 items: medium", (90, 180), (92, 182), {}, lambda r: r["effort"] == "medium"),
        ("medium +3 items: low", (90, 180), (92, 181), {}, lambda r: r["effort"] == "low"),
        ("tie: low", (90, 180), (90, 180), {}, lambda r: r["effort"] == "low"),
        ("low higher: low", (95, 190), (90, 180), {}, lambda r: r["effort"] == "low"),
        ("missing suite file: NO_RESULT", (90, 180), (92, 190), {"missing": "medium/mmlu-pro"},
         lambda r: r["verdict"] == "NO_RESULT" and r["effort"] is None),
        ("holdout split: NO_RESULT", (90, 180), (92, 190), {"split": "holdout"}, lambda r: r["verdict"] == "NO_RESULT"),
        ("short row count: NO_RESULT", (90, 180), (92, 190), {"n": {"mmlu-pro": 250}},
         lambda r: r["verdict"] == "NO_RESULT"),
        ("FAILED_ATTEMPT: NO_RESULT", (90, 180), (92, 190), {"failed": True}, lambda r: r["verdict"] == "NO_RESULT"),
    ]
    for name, low, medium, kw, check in effort_cases:
        tmp = Path(tempfile.mkdtemp(prefix="inkling2-effort-"))
        try:
            result = effort_pick(_write_effort(tmp, "a0", low, medium, **kw), "a0")
            ok = bool(check(result))
        except Exception as exc:  # a crash is a failed case
            ok, result = False, {"crash": repr(exc)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} effort pick: {name}" + ("" if ok else f": {result}"))
    total = len(cases) + 1 + len(effort_cases)
    print(f"{total - failures}/{total} self-test cases match")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return self_test()
    if argv[:1] == ["--may-run"] and len(argv) == 3:
        return 0 if may_run(Path(argv[1]), argv[2]) else 1
    if argv[:1] == ["--effort-due"] and len(argv) == 3:
        return 0 if effort_due(Path(argv[1]), argv[2]) else 1
    if argv[:1] == ["--effort-pick"] and len(argv) == 3:
        if self_test() != 0:
            print("self-test failed: refusing to score", file=sys.stderr)
            return 1
        ev, arm = Path(argv[1]), argv[2]
        result = effort_pick(ev, arm)
        (ev / f"effort-{arm}" / "pick.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return 0 if result["verdict"] == "COMPLETE" else 1
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    ev = Path(argv[0])
    if self_test() != 0:
        print("self-test failed: refusing to score", file=sys.stderr)
        return 1
    result = score(ev)
    (ev / "summary.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({k: result.get(k) for k in ("verdict", "why", "kept", "dropped", "no_result", "carrier",
                                                 "untested_levers", "decided_by")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
