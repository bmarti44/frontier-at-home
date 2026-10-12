#!/usr/bin/env python3
"""Fixed candidate-10 gate verdict per PREREG.md in this directory.

Usage: gate_verdict.py <bundle dir>  -> writes <dir>/gate-summary.json, prints the verdict.
Fails closed: missing or malformed evidence is NO_RESULT, never PASS.
Candidate 9's clauses 1-12 (run 1), extended to run 2's starts and to the
runtime-root release, plus clause 13 (matched speed through the shared
97_matched_speed_blocks.py scorer) and clause 14 (TTFT at depth 0). The
kill-line patterns are imported from the smoke scorer so every run uses
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

SOAK_FLOOR_GIB = 5.0
SWAP_MAX_GIB, FLOOR_GIB, TOOLCALL_MIN = 0.25, 8.0, 19
ADMISSION_GIB = 115.0
# qwen38-1m MMMU-val-100 (results/qwen38-gates/qualify-2026-09-11-kit-check/vision): 64/100.
DEFAULT_VISION_CORRECT, VISION_N, WORSE_Z = 64, 100, -1.645
MIN_CONTEXT_TOKENS, CONTEXT_CAP = 1_000_000, 1_048_576
KV_CAPACITY = re.compile(r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for 262,144 tokens per request")
HEAD = re.compile(r"^\S+ head ([0-9a-f]{40}) dirty=(\d+)$")
LAUNCH = re.compile(r"^(\S+) start1: launching in scope ")
PRECONDITION = re.compile(r"^\S+ precondition ([0-9.]+) GiB (\S+)$")
GGUF = ("/home/bmarti44/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
        "/home/bmarti44/models/qwen3.8-27b/mmproj-Qwen3.8-27B-f16.gguf")
R1 = "/home/bmarti44/.cache/qwen38-flash-next/native-runtime-001/runtime/lib/python3.12/site-packages/"
RUNTIME_LIBS = tuple(R1 + p for p in (
    "nvidia/cu13/lib/libcublasLt.so.13", "triton/_C/libtriton.so", "torch/lib/libtorch_cuda.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so", "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
    "nvidia/cu13/lib/libcufft.so.12", "nvidia/cusparselt/lib/libcusparseLt.so.0",
    "torch/lib/libtorch_cpu.so", "nvidia/nccl/lib/libnccl.so.2"))
GGUF_MIN_GIB, RUNTIME_MIN_GIB = 15.0, 2.5
RELEASE = re.compile(r"host page cache released under (\S+) \((\d+) files\); MemFree ([0-9.]+) GiB, "
                     r"KV arena ([0-9.]+) GiB \+ margin ([0-9.]+) GiB")
RUNTIME_RELEASE = re.compile(r"host page cache released under (\S+) \((\d+) files\) before the KV arena")
REFUSED = "refusing to allocate the KV arena"
MODEL_ROOT = "/home/bmarti44/models"
RUNTIME_ROOT = "/home/bmarti44/.cache/qwen38-flash-next"
MARKERS = {"server.log": "session start", "memwatch.log": "ARMED"}
START = "start1"
AGENT_GATE_TESTS = {"prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"}
# Run 2 (PREREG "Run 2" and clause 13).
A_PROFILE = "qwen3.8-flash-next/cuda-spark-128g-1m"
B_PROFILE = "qwen3.8-flash-next/cuda-spark-128g-1m-tune-c10"
DOMAIN, BLOCKS = "qwen38fn-c10-blocks", 5
DECODE0_POINT_MIN, DECODE_LOWER_MIN, PREFILL_LOWER_MIN, TTFT0_MAX_S = 1.10, 1.0, 0.95, 0.5
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
REPO = Path(os.environ.get("GATE_REPO") or CODE)
CANDIDATE = "results/qwen38fn-gates/qualify-2026-10-12-c10"
FROZEN = (
    "configs/profiles/qwen3.8-flash-next", "configs/build-manifests/qwen38-flash-next-runtime.json",
    "configs/build-manifests/qwen38-flash-next-runtime-002.json", "patches/qwen38fn",
    "scripts/lib", "scripts/dev/regression-suite.py", "scripts/01_memwatch.sh", "scripts/02_membudget.py",
    "scripts/03_memory_guard.py", "scripts/30_bench_speed.py", "scripts/30_bench_speed_blocks.py",
    "scripts/35_soak.py", "scripts/38_bench_vision.py", "scripts/39_bench_toolcall.py",
    "scripts/50_probe_context.py", "scripts/92_resolve_profile.py", "scripts/93_profile_serve.sh",
    "scripts/94_qualify_profile.py", "scripts/97_matched_speed_blocks.py",
    "scripts/103_verify_drand_receipt_bundle.mjs", "fixtures", "evalsets/mmmu-val-100", "evalsets/toolcall",
    "results/qwen38fn-gates/smoke-2026-10-11/probe.py", "results/qwen38fn-gates/smoke-2026-10-11/verdict.py",
    "results/qwen38fn-gates/qualify-2026-10-11-c4/probe-start1.json",
    "results/qwen38fn-gates/qualify-2026-10-11-c7/pagecache_residency.py",
    f"{CANDIDATE}/PREREG.md", f"{CANDIDATE}/run-c10.sh", f"{CANDIDATE}/gate_verdict.py",
    f"{CANDIDATE}/host-worktree.json",
)
REFERENCE_PROBE = Path(os.environ.get("GATE_REFERENCE_PROBE")
                       or HERE.parent / "qualify-2026-10-11-c4" / "probe-start1.json")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KILL = _module("smoke_verdict", HERE.parent / "smoke-2026-10-11" / "verdict.py").KILL
MATCHED = _module("matched_speed_blocks", CODE / "scripts" / "97_matched_speed_blocks.py")


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


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def code_identity(lines, label, freeze, missing, checks):
    heads = [m.groups() for m in map(HEAD.match, lines) if m]
    checks[f"{label}_heads"] = heads
    if len(heads) != 1 or heads[0][1] != "0" or not freeze:
        missing.append(f"{label} code identity: heads {heads}, freeze {freeze or None}")
        return
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, heads[0][0], "--", *FROZEN],
                          capture_output=True)
    checks[f"{label}_frozen_diff_rc"] = diff.returncode
    if diff.returncode != 0:
        missing.append(f"{label} code identity: frozen files differ between {freeze} and {heads[0][0]}")


def run1(ev, freeze, missing, fail, checks):
    obs = ev / "observe" / START
    server = obs / "server.log"
    if server.is_file():
        text = server.read_text(errors="replace")
        capacity = [int(m.replace(",", "")) for m in KV_CAPACITY.findall(text)]
        checks["kv_capacity_tokens"] = capacity[-1] if capacity else None
        if not capacity:
            missing.append("observe/start1/server.log: no KV capacity line at 262,144")
        elif capacity[-1] < CONTEXT_CAP:
            fail.append(f"clause 4: KV capacity {capacity[-1]} < {CONTEXT_CAP}")
        releases = RELEASE.findall(text)
        runtime_releases = RUNTIME_RELEASE.findall(text)
        checks["host_cache_release"] = [
            {"root": r, "files": int(n), "mem_free_gib": float(f), "arena_gib": float(a), "margin_gib": float(m)}
            for r, n, f, a, m in releases]
        checks["runtime_cache_release"] = [{"root": r, "files": int(n)} for r, n in runtime_releases]
        if REFUSED in text:
            fail.append("clause 12: the worker refused the KV arena (MemFree short after the release)")
        else:
            if (len(releases) != 1 or releases[0][0] != MODEL_ROOT or int(releases[0][1]) < 1
                    or float(releases[0][3]) < 17.0):
                missing.append(f"clause 12: want one release line under {MODEL_ROOT}, at least one file, "
                               f"for the 17.9 GiB arena, got {releases}")
            if (len(runtime_releases) != 1 or runtime_releases[0][0] != RUNTIME_ROOT
                    or int(runtime_releases[0][1]) < 1):
                missing.append(f"clause 12: want one runtime release line under {RUNTIME_ROOT}, at least "
                               f"one file, got {runtime_releases}")
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
        fail.append("clause 1: kill/OOM/NVRM/Xid line (run 1)")

    driver = obs / "driver.log"
    lines = driver.read_text().splitlines() if driver.is_file() else []
    if not driver.is_file():
        missing.append("observe/start1/driver.log")
    launches = [m.group(1) for m in map(LAUNCH.match, lines) if m]
    checks["launch_stamps"] = launches
    primed = {path: float(gib) for gib, path in (m.groups() for m in map(PRECONDITION.match, lines) if m)}
    checks["precondition_resident_gib"] = primed
    gguf = sum(primed.get(p, 0.0) for p in GGUF)
    runtime = sum(primed.get(p, 0.0) for p in RUNTIME_LIBS)
    checks["precondition_totals_gib"] = {"gguf": gguf, "runtime_001": runtime}
    if set(primed) != set(GGUF) | set(RUNTIME_LIBS) or gguf < GGUF_MIN_GIB or runtime < RUNTIME_MIN_GIB:
        missing.append(f"clause 11: primed files not resident at launch (gguf {gguf:.2f}, runtime {runtime:.2f} "
                       f"GiB, {len(primed)} files); the fix was not tested")
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
            fail.append("clause 2: swap growth (run 1)")
        if low < FLOOR_GIB:
            fail.append("clause 3: external low point below 8 GiB (run 1)")
        before = [a for (t, _), a in zip(rows, available) if len(launches) == 1 and t <= launches[0]]
        if not before:
            missing.append(f"clause 10: no meminfo sample at or before one launch line (launches {launches})")
        else:
            checks["launch_mem_available_gib"] = max(before)
            checks["peak_used_gib"] = peak = max(before) - low
            if peak + FLOOR_GIB > ADMISSION_GIB:
                fail.append("clause 10: peak use + 8 GiB exceeds the 115 GiB switch admission")
    checks["failed_attempt_lines"] = [l for l in lines if "FAILED_ATTEMPT" in l]
    if driver.is_file():
        code_identity(lines, "run1", freeze, missing, checks)

    manifest = load(ev / "manifest.json", missing)
    context = load(ev / "context" / "summary.json", missing)
    if manifest is not None and context is not None:
        cap = (manifest.get("profile") or {}).get("context_cap")
        checks["context"] = {"verdict": context.get("verdict"), "total_tokens": context.get("total_tokens"),
                             "min_mem_available_gib": context.get("min_mem_available_gib"), "context_cap": cap}
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
                missing.append("clause 5: text path differs from candidate 5; the holdout decides accuracy")

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
        checks["soak_floor"] = {"mem_floor_gib": soak.get("mem_floor_gib"),
                                "mem_floor_source": soak.get("mem_floor_source"),
                                "mem_available_min_gib": soak.get("mem_available_min_gib")}
        if not isinstance(soak.get("pass"), bool):
            missing.append("soak.pass malformed")
        elif soak.get("mem_floor_gib") != SOAK_FLOOR_GIB or soak.get("mem_floor_source") != "default":
            missing.append(f"clause 8: soak not held to the repository {SOAK_FLOOR_GIB} GiB floor")
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

    speed = load(ev / "speed" / "speed.json", missing)
    if speed is not None:
        cells = [c for c in speed.get("cells") or [] if c.get("ctx_tokens") == 0]
        ttft = cells[0].get("median_ttft") if len(cells) == 1 else None
        checks["ttft_0_s"] = ttft
        if not isinstance(ttft, (int, float)) or isinstance(ttft, bool) or not math.isfinite(ttft):
            missing.append("clause 14: no finite TTFT at depth 0 in speed/speed.json")
        elif ttft > TTFT0_MAX_S:
            fail.append(f"clause 14: TTFT at depth 0 {ttft:.3f} s > {TTFT0_MAX_S}")


def run2(ev, freeze, missing, fail, checks):
    plan = load(ev / "plan.json", missing)
    blocks = ev / "blocks"
    if plan is None:
        return
    receipt = plan.get("drand") or {}
    try:
        frozen_at = int(git("show", "-s", "--format=%ct", freeze)) if freeze else None
    except ValueError:
        frozen_at = None
    checks["drand"] = {"round": receipt.get("round"), "frozen_commit": receipt.get("frozen_commit")}
    problems = []
    if (plan.get("a_profile"), plan.get("b_profile"), plan.get("domain")) != (A_PROFILE, B_PROFILE, DOMAIN):
        problems.append("plan arms or domain")
    if receipt.get("frozen_commit") != freeze or frozen_at is None:
        problems.append("drand receipt does not name the freeze commit")
    elif receipt.get("round") != MATCHED.first_drand_round_after(frozen_at):
        problems.append("drand round is not the first after the freeze")
    if receipt.get("randomness") != plan.get("randomness") or not isinstance(plan.get("randomness"), str):
        problems.append("plan randomness is not the receipt's")
    else:
        try:
            if plan.get("blocks") != MATCHED.block_orders(plan["randomness"], BLOCKS, DOMAIN):
                problems.append("block orders disagree with the receipt")
        except ValueError:
            problems.append("randomness malformed")
    if not problems:
        try:
            MATCHED.verify_drand_round({k: receipt.get(k) for k in MATCHED.DRAND_FIELDS})
        except (RuntimeError, OSError, KeyError) as error:
            problems.append(f"drand BLS verification: {error}")
    if problems:
        missing.append("clause 13: " + "; ".join(problems))
        return
    driver = blocks / "driver.log"
    if not driver.is_file():
        missing.append("blocks/driver.log")
    else:
        code_identity(driver.read_text().splitlines(), "run2", freeze, missing, checks)
    unsafe = {"kill": [], "swap": [], "low": []}
    for start in sorted(blocks.glob("block-*/*/start.json")):
        try:
            s = json.loads(start.read_text())
        except (OSError, ValueError):
            continue   # the shared scorer below reports it as NO_RESULT
        label = f"{start.parent.parent.name}/{start.parent.name}"
        if s.get("kill_lines"):
            unsafe["kill"].append(label)
        swap, low = s.get("swap_growth_gib"), s.get("low_mem_available_gib")
        if isinstance(swap, (int, float)) and swap > SWAP_MAX_GIB:
            unsafe["swap"].append(label)
        if isinstance(low, (int, float)) and low < FLOOR_GIB:
            unsafe["low"].append(label)
    checks["run2_unsafe_starts"] = unsafe
    for key, clause in (("kill", "clause 1: kill line (run 2)"), ("swap", "clause 2: swap growth (run 2)"),
                        ("low", "clause 3: low point below 8 GiB (run 2)")):
        if unsafe[key]:
            fail.append(f"{clause} at {unsafe[key]}")
    if not (blocks / "plan.json").is_file() or load(blocks / "plan.json", missing) != plan:
        missing.append("clause 13: blocks/plan.json is not plan.json")
        return
    result = MATCHED.score(blocks)
    checks["matched"] = {"verdict": result["verdict"], "reasons": result["reasons"][:10],
                         **{k: {m: v[m] for m in ("ratio", "lower_95", "upper_95")}
                            for k, v in result.get("metrics", {}).items()}}
    if result["verdict"] != "COMPLETE":
        missing.append(f"clause 13: matched blocks {result['verdict']}: {result['reasons'][:3]}")
        return
    m = result["metrics"]
    if m["decode_0"]["ratio"] < DECODE0_POINT_MIN:
        fail.append(f"clause 13: decode at 0 ratio {m['decode_0']['ratio']:.4f} < {DECODE0_POINT_MIN}")
    for key, floor in (("decode_0", DECODE_LOWER_MIN), ("decode_28672", DECODE_LOWER_MIN),
                       ("prefill_28672", PREFILL_LOWER_MIN)):
        if m[key]["lower_95"] < floor:
            fail.append(f"clause 13: {key} lower-95 {m[key]['lower_95']:.4f} < {floor}")


def main(ev):
    missing, fail, checks = [], [], {}
    freeze = git("log", "-1", "--format=%H", "--", f"{CANDIDATE}/PREREG.md")
    checks["freeze"] = freeze
    run1(ev, freeze, missing, fail, checks)
    run2(ev, freeze, missing, fail, checks)
    verdict = "FAIL" if fail else ("NO_RESULT" if missing or checks["failed_attempt_lines"] else "PASS")
    out = {"formula": "PREREG.md", "checks": checks, "fail": fail, "missing": missing, "verdict": verdict}
    (ev / "gate-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, "fail": fail, "missing": missing,
                      "mem_low_gib": checks.get("mem_available_low_gib"),
                      "peak_used_gib": checks.get("peak_used_gib"), "matched": checks.get("matched")}))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
