#!/usr/bin/env python3
"""Fixed candidate-10 gate verdict per PREREG.md in this directory.

Usage: gate_verdict.py <bundle dir>  -> writes <dir>/gate-summary.json, prints the verdict.
       gate_verdict.py --run2-preflight <bundle dir>  -> exit 0 iff run 2 may start (no file written).
Fails closed: missing or malformed evidence is NO_RESULT, never PASS.
Candidate 9's clauses 1-12 (run 1), extended to run 2's starts and to the
runtime-root release, plus clause 13 (matched speed through the shared
97_matched_speed_blocks.py scorer) and clause 14 (TTFT at depth 0). The
kill-line patterns are imported from the smoke scorer, and this scorer applies
them to the raw logs of run 1 and of every run-2 start, so both runs use
exactly the same ones (97's own start.json kill lines are checked as well).
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SOAK_FLOOR_GIB = 5.0
SWAP_MAX_GIB, FLOOR_GIB, TOOLCALL_MIN = 0.25, 8.0, 19
ADMISSION_GIB = 115.0
# qwen38-1m MMMU-val-100 (results/qwen38-gates/qualify-2026-09-11-kit-check/vision): 64/100.
DEFAULT_VISION_CORRECT, VISION_N, WORSE_Z = 64, 100, -1.645
MIN_CONTEXT_TOKENS, CONTEXT_CAP = 1_000_000, 1_048_576
KV_CAPACITY = re.compile(r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for 262,144 tokens per request")
HEAD = re.compile(r"^\S+ head ([0-9a-f]{40}) dirty=(\d+)$")
# run-c10.sh logs both runtimes' site-packages/vllm digests (16 hex) before launch and after the
# run, by the runtime-002 manifest's vllm_tree method.
RUNTIME_TREE = re.compile(r"^\S+ runtime-tree (native-runtime-00[12]) ([0-9a-f]{16}) (\d+)$")
RUNTIME_MANIFEST = "configs/build-manifests/qwen38-flash-next-runtime-002.json"
# Candidate 10's levers as its startup lines show them (screen arm a3's server.log): run 1 and every
# B start must show all of them, and no A start may show any. Every start shows the 3,200-token block.
LEVER_LINES = ("checkpoint_mapped PLE: QWEN38FN_PLE_WILLNEED=1",
               "Qwen4Exp MTP draft vocabulary: token ids [0, 65536)",
               "Quantized a nvfp4 copy of the lm_head", "'draft_lm_head_quantization': 'nvfp4'")
BLOCK_LINE = "Setting attention block size to 3200 tokens"
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
    "scripts/tests/test_qwen38fn_tune_contract.py",
    "scripts/103_verify_drand_receipt_bundle.mjs", "fixtures", "evalsets/mmmu-val-100", "evalsets/toolcall",
    "results/qwen38fn-gates/smoke-2026-10-11/probe.py", "results/qwen38fn-gates/smoke-2026-10-11/verdict.py",
    "results/qwen38fn-gates/qualify-2026-10-11-c4/probe-start1.json",
    "results/qwen38fn-gates/qualify-2026-10-11-c7/pagecache_residency.py",
    f"{CANDIDATE}/PREREG.md", f"{CANDIDATE}/run-c10.sh", f"{CANDIDATE}/gate_verdict.py",
    f"{CANDIDATE}/host-worktree.json", f"{CANDIDATE}/gate_mutations.py", f"{CANDIDATE}/cjk_decode.py",
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
    # Every invocation logs its head (a resumed run 2 logs it again); all must be one clean commit.
    heads = sorted({m.groups() for m in map(HEAD.match, lines) if m})
    checks[f"{label}_heads"] = heads
    if len(heads) != 1 or heads[0][1] != "0" or not freeze:
        missing.append(f"{label} code identity: heads {heads}, freeze {freeze or None}")
        return
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, heads[0][0], "--", *FROZEN],
                          capture_output=True)
    checks[f"{label}_frozen_diff_rc"] = diff.returncode
    if diff.returncode != 0:
        missing.append(f"{label} code identity: frozen files differ between {freeze} and {heads[0][0]}")


def runtime_trees(lines, label, freeze, missing, checks):
    """Both runtimes' vllm trees, as logged, equal the frozen runtime-002 manifest's digests."""
    try:
        manifest = json.loads(git("show", f"{freeze}:{RUNTIME_MANIFEST}")) if freeze else {}
        want = {"native-runtime-002": (manifest["vllm_tree"]["sha256"][:16], manifest["vllm_tree"]["files"]),
                "native-runtime-001": (manifest["base_runtime"]["base_vllm_tree"]["sha256"][:16],
                                       manifest["base_runtime"]["base_vllm_tree"]["files"])}
    except (ValueError, KeyError, TypeError):
        missing.append(f"{label} runtime trees: no {RUNTIME_MANIFEST} digests at the freeze")
        return
    seen = [(name, sha, int(files)) for name, sha, files in (m.groups() for m in map(RUNTIME_TREE.match, lines) if m)]
    checks[f"{label}_runtime_trees"] = seen
    for name, (sha, files) in want.items():
        got = {(s, f) for n, s, f in seen if n == name}
        if got != {(sha, files)}:
            missing.append(f"{label} runtime trees: {name} logged {sorted(got)}, frozen {(sha, files)}")


def raw_kill_lines(d):
    """Kill lines in a directory's kernel/server/memwatch logs, by the smoke scorer's patterns.
    The engine's arena-guard refusal names NVRM as what it prevents: not a kill line here (run 1
    scores a refusal by clause 12)."""
    hits = []
    for name, rx in KILL.items():
        f = d / ("server.log" if name == "server" else name)
        if f.is_file():
            hits += [f"{f.name}: {l.strip()}" for l in f.read_text(errors="replace").splitlines()
                     if rx.search(l) and REFUSED not in l]
    return hits


# Run 1's bash sampler writes SwapTotal; 97's sampler (run 2) writes its MEMINFO_KEYS, which do not
# include it, and run 2's swap growth uses SwapFree alone.
RUN1_SAMPLE_KEYS, RUN2_SAMPLE_KEYS = {"MemAvailable", "SwapTotal", "SwapFree"}, {"MemAvailable", "SwapFree"}
if not RUN2_SAMPLE_KEYS <= set(MATCHED.MEMINFO_KEYS):
    raise SystemExit(f"97 samples {MATCHED.MEMINFO_KEYS}, not {sorted(RUN2_SAMPLE_KEYS)}")


def meminfo_rows(path, keys):
    rows = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        kv = dict(re.findall(r"(\w+):=(\d+)", line))
        if keys <= kv.keys():
            rows.append((line.split(" ", 1)[0], kv))
    return rows


def seconds(stamp):
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def covered(stamps, first_by, last_by):
    """Samples from at or before first_by to within 15 s of last_by, never more than 5 s apart."""
    try:
        t = [seconds(x) for x in stamps]
        return (t[0] <= seconds(first_by) and t[-1] >= seconds(last_by) - 15
                and max(b - a for a, b in zip(t, t[1:])) <= 5)
    except (TypeError, ValueError, IndexError):
        return False


def start_identity(text, want_levers):
    """Lever and readiness lines a start's server log must (or, for an A start, must not) show."""
    problems = [f"{'missing' if want_levers else 'unexpected'} lever line {line!r}"
                for line in LEVER_LINES if (line in text) != want_levers]
    capacity = [int(m.replace(",", "")) for m in KV_CAPACITY.findall(text)]
    if BLOCK_LINE not in text:
        problems.append("no 3,200-token attention block line")
    if not capacity or capacity[-1] < CONTEXT_CAP:
        problems.append(f"KV capacity {capacity[-1:] or None} at 262,144 (want >= {CONTEXT_CAP})")
    return problems


def scoring_identity(freeze, missing, checks):
    """The scorer and every frozen path, as they are while scoring, are the freeze's."""
    checks["scorer"] = {"sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                        "scoring_head": git("rev-parse", "HEAD"),
                        "overrides": {k: os.environ[k] for k in ("GATE_REPO", "GATE_REFERENCE_PROBE")
                                      if k in os.environ}}
    if not freeze:
        return
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, "--", *FROZEN], capture_output=True)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN)
    checks["scorer"].update(frozen_worktree_diff_rc=diff.returncode, untracked_frozen=untracked.splitlines())
    if diff.returncode != 0 or untracked:
        missing.append(f"scoring identity: the working tree differs from the freeze {freeze} under FROZEN "
                       f"(diff rc {diff.returncode}, untracked {untracked.splitlines()[:5]})")


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
        checks["run1_identity_problems"] = identity = [p for p in start_identity(text, True)
                                                       if not p.startswith("KV capacity")]
        if identity:
            missing.append(f"run 1 identity: {identity}")
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
    rows = meminfo_rows(obs / "meminfo.log", RUN1_SAMPLE_KEYS)
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
        ends = [l.split(" ", 1)[0] for l in lines if l.endswith(f"{START}: scope empty after stop")]
        if len(launches) != 1 or len(ends) != 1 or not covered([t for t, _ in rows], launches[0], ends[0]):
            missing.append("observe/start1/meminfo.log: samples do not cover launch to the empty scope, "
                           "at most 5 s apart")
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
        runtime_trees(lines, "run1", freeze, missing, checks)

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
        rows = agent.get("results") or []
        results = {r.get("test"): r.get("pass") for r in rows}
        checks["agent_gate"] = results
        if sorted(r.get("test") for r in rows) != sorted(AGENT_GATE_TESTS):
            missing.append(f"agent-gate tests {[r.get('test') for r in rows]} (each of the four exactly once)")
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
        if speed.get("suite_valid") is not True or len(cells) != 1 or cells[0].get("valid") is not True:
            missing.append("clause 14: speed/speed.json suite or depth-0 cell is not valid")
        elif not isinstance(ttft, (int, float)) or isinstance(ttft, bool) or not math.isfinite(ttft):
            missing.append("clause 14: no finite TTFT at depth 0 in speed/speed.json")
        elif ttft > TTFT0_MAX_S:
            fail.append(f"clause 14: TTFT at depth 0 {ttft:.3f} s > {TTFT0_MAX_S}")


def plan_problems(ev, freeze, missing, checks):
    """Clause 13's plan checks: arms, domain, the freeze's drand round, BLS, block orders."""
    plan = load(ev / "plan.json", missing)
    if plan is None:
        return None, ["no plan.json"]
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
    return plan, problems


def run2(ev, freeze, missing, fail, checks):
    blocks = ev / "blocks"
    plan, problems = plan_problems(ev, freeze, missing, checks)
    if problems and plan is not None:
        missing.append("clause 13: " + "; ".join(problems))
    # The safety scan below runs whatever the plan's state: a defective plan must not mask a kill line.
    driver = blocks / "driver.log"
    if not driver.is_file():
        missing.append("blocks/driver.log")
    else:
        lines = driver.read_text().splitlines()
        code_identity(lines, "run2", freeze, missing, checks)
        runtime_trees(lines, "run2", freeze, missing, checks)
    run_log = (blocks / "run.log").read_text().splitlines() if (blocks / "run.log").is_file() else []
    unsafe = {"kill": [], "swap": [], "low": []}
    gaps, identity = [], []
    # Every start directory, with or without start.json: a start whose driver died still has its
    # kernel log and samples, and an OOM there is a FAIL, not only a NO_RESULT.
    for d in sorted(p for p in blocks.glob("block-*/*") if p.is_dir()):
        label = f"{d.parent.name}/{d.name}"
        if raw_kill_lines(d):
            unsafe["kill"].append(label)
        text = (d / "server.log").read_text(errors="replace") if (d / "server.log").is_file() else ""
        if REFUSED in text and d.name.endswith("-B"):
            fail.append(f"clause 12 (run 2): candidate 10's worker refused the KV arena at {label}")
        identity += [f"{label}: {p}" for p in start_identity(text, d.name.endswith("-B"))]
        rows = meminfo_rows(d / "meminfo.log", RUN2_SAMPLE_KEYS)
        available = [int(r["MemAvailable"]) / 2**20 for _, r in rows]
        free = [int(r["SwapFree"]) / 2**20 for _, r in rows]
        if rows and min(available) < FLOOR_GIB:
            unsafe["low"].append(label)
        if rows and free[0] - min(free) > SWAP_MAX_GIB:
            unsafe["swap"].append(label)
        try:
            s = json.loads((d / "start.json").read_text())
        except (OSError, ValueError):
            continue   # the shared scorer below reports it as NO_RESULT
        if s.get("kill_lines"):
            unsafe["kill"].append(label)
        swap, low, ready = s.get("swap_growth_gib"), s.get("low_mem_available_gib"), s.get("ready_s")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (swap, low, ready)):
            gaps.append(f"{label}: swap, low point or ready_s not finite")
            continue
        if swap > SWAP_MAX_GIB:
            unsafe["swap"].append(label)
        if low < FLOOR_GIB:
            unsafe["low"].append(label)
        # The sampler covered the start: from launch to its result line in run.log, at most 5 s apart.
        done = [l.split(" ", 1)[0] for l in run_log
                if re.match(rf"^\S+ block {s.get('block')} position {s.get('position')} arm {s.get('arm')}: failed=", l)]
        if not (len(rows) >= 0.9 * ready and len(done) == 1
                and covered([t for t, _ in rows], s.get("launched_at"), done[0])):
            gaps.append(f"{label}: meminfo samples do not cover the start ({len(rows)} rows, ready_s {ready})")
    checks["run2_unsafe_starts"] = unsafe = {k: sorted(set(v)) for k, v in unsafe.items()}
    checks["run2_coverage_gaps"] = gaps
    checks["run2_identity_problems"] = identity
    if gaps:
        missing.append(f"clauses 2-3 (run 2): {gaps[:3]}")
    if identity:
        missing.append(f"run 2 identity: {identity[:3]}")
    for key, clause in (("kill", "clause 1: kill line (run 2)"), ("swap", "clause 2: swap growth (run 2)"),
                        ("low", "clause 3: low point below 8 GiB (run 2)")):
        if unsafe[key]:
            fail.append(f"{clause} at {unsafe[key]}")
    if problems:
        return
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


def run2_preflight(ev):
    """Run 2 starts only after a clean run 1 (no failed clause, no FAILED_ATTEMPT, nothing missing except
    clause 5: a changed text path sends accuracy to the holdout, not to a new run 1), with a plan that is
    valid for the current freeze, from a working tree that is the freeze's."""
    missing, fail, checks = [], [], {}
    freeze = git("log", "-1", "--format=%H", "--", f"{CANDIDATE}/PREREG.md")
    run1(ev, freeze, missing, fail, checks)
    _, problems = plan_problems(ev, freeze, [], checks)
    scoring_identity(freeze, missing, checks)
    blocking = [m for m in missing if not m.startswith("clause 5:")] + [f"plan: {p}" for p in problems]
    print(json.dumps({"fail": fail, "missing": blocking, "failed_attempt_lines": checks.get("failed_attempt_lines")}))
    return 0 if not fail and not blocking and not checks.get("failed_attempt_lines") else 1


def main(ev):
    missing, fail, checks = [], [], {}
    freeze = git("log", "-1", "--format=%H", "--", f"{CANDIDATE}/PREREG.md")
    checks["freeze"] = freeze
    scoring_identity(freeze, missing, checks)
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
    if sys.argv[1:2] == ["--run2-preflight"]:
        sys.exit(run2_preflight(Path(sys.argv[2])))
    sys.exit(main(Path(sys.argv[1])))
