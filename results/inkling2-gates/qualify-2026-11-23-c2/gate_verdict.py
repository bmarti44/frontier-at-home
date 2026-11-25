#!/usr/bin/env python3
"""Fixed M3 gate verdict for Inkling 2 Mini candidate 2 per PREREG.md in this directory.

Usage: gate_verdict.py <bundle dir>  -> writes <dir>/gate-summary.json, prints the verdict.
       (gate_mutations.py adds --mutation, which alone lets GATE_REPO / GATE_REFERENCE_PROBE apply.)
       gate_verdict.py --preflight   -> exit 0 iff the working tree is the freeze's under FROZEN.
Fails closed: missing or malformed evidence is NO_RESULT, never PASS; a failed clause is FAIL
whatever else is missing. Adapted from results/mimo-v26-flash-gates/qualify-2026-10-13-c1 (one
start, the kit's cells) and results/qwen38fn-gates/qualify-2026-10-12-c10 (direct 1M, the
host page-cache release worker).
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---- carrier-dependent constants (set from the screen and the effort pick before the freeze)
SCREEN_CARRIER = "a6"                        # tune-screen-2026-11-23 carrier arm
EFFORT = "low"                          # effort-a6/pick.json
# Server-log lines of the carrier's configuration: DSpark x7, humming, #69288, and the screen's
# unchanged 2048-token batch (a1 batch 4096 was not kept, a2 batch 8192 failed its floor).
CONFIG_NEEDLES = ("'method': 'dspark'", "'moe_backend': 'humming'", "INKLING2_MAX_CTAS_POW2=1",
                  "'max_num_batched_tokens': 2048")
SPEC_BLOCK_DROP = True                       # DSpark is carried
SPEED_K = 8                                  # the profile's speed_max_tokens_per_event
# ----

KERNEL_KILL = re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I)
SERVER_KILL = re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid")
MEMWATCH_KILL = re.compile(r"BREACH|EMERGENCY_STOP|FAIL_CLOSED")
SWAP_MAX_GIB, FLOOR_GIB, ADMISSION_GIB, TOOLCALL_MIN = 0.25, 10.0, 115, 19
GATE_FREE_GIB, GATE_AVAILABLE_GIB = 100, 110
GATE_LINE = re.compile(r"MemFree gate pass (\d+) kB, MemAvailable (\d+) kB \(3 samples\)$")
SOAK_FLOOR_GIB = 5.0
CONTEXT_CAP, MIN_CONTEXT_TOKENS = 1_048_576, 1_000_000
KV_CAPACITY = re.compile(r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for 262,144 tokens per request")
# Flash-Next (the default) c10 MMMU-val-100: 64/100.
DEFAULT_VISION_CORRECT, VISION_N, WORSE_Z = 64, 100, -1.645
VISION_MODE, VISION_BUDGET = "chat", "4096"
SERVED = "inkling-2-mini"
AGENT_GATE_TESTS = {"prefix-cache", "turn-continuation", "slot-thrash", "prefill-throughput"}
AGENT_GATE_SEMANTICS = {"engine": "vllm", "model": SERVED, "prefix_block_tokens": 64,
                        "spec_block_drop": SPEC_BLOCK_DROP}
MARKERS = {"server.log": re.compile(r"session start"), "memwatch.log": re.compile(r"(^|\s)ARMED target_pid=")}
JOURNAL_ALIVE = "kernel journal alive after the engine stop"
HEAD = re.compile(r"^\S+ head ([0-9a-f]{40}) dirty=(\d+)$")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) ")
FIXTURE_LINE = re.compile(r"^\S+ agent-gate fixture sha256_16 ([0-9a-f]{16})$")
AGENT_FIXTURE = "fixtures/ctx-32k.txt"  # regression-suite.py reads it from the production checkout
INTERPRETER = re.compile(r"^interpreter match=(True|False)$")
TREE_LINE = re.compile(r"^vllm-tree ([0-9a-f]{16}) files=(\d+) after=.* match=(True|False)$")
PATCHES = re.compile(r"^patches (\d+) mismatches=\[(.*)\]$")
WEIGHTS_LINE = re.compile(r"^weights files=(\d+) bytes=(\d+) mismatches=(\d+)")
FREEZE_LINE = re.compile(r"^packages freeze sha256_16 ([0-9a-f]{16}) match=(True|False)$")
# The block-aware bench's decode method, exactly as 30_bench_speed_blocks.py writes it under
# metadata["decode_method"] (review A-M3-L4; candidate 1 read a "method" key no bench writes).
SPEED_METHOD = "block-aware: (client_completion_tokens - max_tokens_per_event) / (t_last_event - t_first_event)"
RUNTIME_RELEASE = re.compile(r"host page cache released under (\S+) \((\d+) files\) before the KV arena")
RELEASE = re.compile(r"host page cache released under (\S+) \((\d+) files\); MemFree ([0-9.]+) GiB, "
                     r"KV arena ([0-9.]+) GiB \+ margin ([0-9.]+) GiB")
REFUSED = "refusing to allocate the KV arena"
RUNTIME_ROOTS = ("/home/bmarti44/.cache/inkling-2-mini", "/home/bmarti44/.cache/qwen38-flash-next")
MODEL_ROOT = "/home/bmarti44/models"
BUILD_MANIFEST = "configs/build-manifests/inkling-2-mini-runtime.json"
WEIGHTS_MANIFEST = "weights/inkling-2-mini/manifest.json"
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
REPO = Path(os.environ.get("GATE_REPO") or CODE)
CANDIDATE = "results/inkling2-gates/qualify-2026-11-23-c2"
SCREEN = "results/inkling2-gates/tune-screen-2026-11-23"
SMOKE = "results/inkling2-gates/smoke-2026-11-23"
FROZEN = (
    "configs/profiles/inkling-2-mini", BUILD_MANIFEST, WEIGHTS_MANIFEST, "configs/hosts/spark-aba1.json",
    "patches/inkling2", "scripts/lib", "scripts/dev/regression-suite.py", "scripts/01_memwatch.sh",
    "scripts/02_membudget.py", "scripts/03_memory_guard.py", "scripts/30_bench_speed.py",
    "scripts/30_bench_speed_blocks.py", "scripts/35_soak.py", "scripts/38_bench_vision.py",
    "scripts/39_bench_toolcall.py", "scripts/50_probe_context.py", "scripts/92_resolve_profile.py",
    "scripts/93_profile_serve.sh", "scripts/94_qualify_profile.py", "fixtures", "evalsets/mmmu-val-100",
    "evalsets/toolcall", f"{SMOKE}/probe.py", f"{SMOKE}/host-worktree.json",
    f"{SCREEN}/{SCREEN_CARRIER}/probe.json",
    f"{CANDIDATE}/PREREG.md", f"{CANDIDATE}/run-c2.sh", f"{CANDIDATE}/gate_verdict.py",
    f"{CANDIDATE}/gate_mutations.py", f"{CANDIDATE}/verify_identity.py", f"{CANDIDATE}/probe_low.py",
    # The carrier and the effort PREREG cites (review A-M3-L1).
    f"{SCREEN}/summary.json", f"{SCREEN}/effort-{SCREEN_CARRIER}/pick.json",
)
REFERENCE_PROBE_PATH = f"{SCREEN}/{SCREEN_CARRIER}/probe.json"
REFERENCE_PROBE = Path(os.environ.get("GATE_REFERENCE_PROBE") or CODE / REFERENCE_PROBE_PATH)
OVERRIDES = ("GATE_REPO", "GATE_REFERENCE_PROBE")
MUTATION = "--mutation" in sys.argv[1:]


def load(path, missing):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        missing.append(f"{path.name} ({type(error).__name__})")
        return None
    if not isinstance(data, dict):
        missing.append(f"{path.name} (top level is {type(data).__name__}, not an object)")
        return None
    return data


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def frozen_bytes(freeze, path) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{freeze}:{path}"], capture_output=True).stdout


def frozen_json(freeze, path):
    try:
        return json.loads(git("show", f"{freeze}:{path}"))
    except ValueError:
        return None


def number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def worse_z(correct: int, n: int = VISION_N, reference: int = DEFAULT_VISION_CORRECT) -> float:
    """One-sided two-proportion z of correct/n against reference/n (pooled)."""
    pooled = (correct + reference) / (2 * n)
    se = math.sqrt(pooled * (1 - pooled) * 2 / n)
    return (correct / n - reference / n) / se if se else 0.0


def seconds(stamp):
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def scoring_identity(freeze, missing, checks):
    """The scorer and every frozen path, as they are while scoring, are the freeze's."""
    checks["scorer"] = {"sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                        "scoring_head": git("rev-parse", "HEAD"),
                        "overrides": {k: os.environ[k] for k in OVERRIDES if k in os.environ},
                        "mutation": MUTATION}
    if checks["scorer"]["overrides"] and not MUTATION:
        missing.append(f"scoring identity: overrides {sorted(checks['scorer']['overrides'])} outside --mutation")
    if not freeze:
        missing.append("scoring identity: no commit adds PREREG.md (no freeze)")
        return
    if not MUTATION:
        for path, actual in ((f"{CANDIDATE}/gate_verdict.py", Path(__file__)), (REFERENCE_PROBE_PATH, REFERENCE_PROBE)):
            try:
                same = actual.read_bytes() == frozen_bytes(freeze, path)
            except OSError:
                same = False
            if not same:
                missing.append(f"scoring identity: {actual} is not the freeze's {path}")
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, "--", *FROZEN], capture_output=True)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN).splitlines()
    checks["scorer"].update(frozen_worktree_diff_rc=diff.returncode, untracked_frozen=untracked)
    if diff.returncode != 0 or untracked:
        missing.append(f"scoring identity: the working tree differs from the freeze {freeze} under FROZEN "
                       f"(diff rc {diff.returncode}, untracked {untracked[:5]})")


def code_identity(lines, freeze, missing, checks):
    heads = sorted({m.groups() for m in map(HEAD.match, lines) if m})
    checks["heads"] = heads
    if len(heads) != 1 or heads[0][1] != "0" or not freeze:
        missing.append(f"code identity: heads {heads}, freeze {freeze or None}")
        return None
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, heads[0][0], "--", *FROZEN],
                          capture_output=True)
    checks["frozen_diff_rc"] = diff.returncode
    if diff.returncode != 0:
        missing.append(f"code identity: frozen files differ between {freeze} and {heads[0][0]}")
    return heads[0][0]


def artifact_identity(lines, freeze, missing, checks):
    """verify_identity.py's lines, against the freeze's build and weights manifests."""
    build, weights = (frozen_json(freeze, p) for p in (BUILD_MANIFEST, WEIGHTS_MANIFEST))
    if not (build and weights):
        missing.append("artifact identity: build or weights manifest unreadable at the freeze")
        return
    interp = [m.group(1) for m in map(INTERPRETER.match, lines) if m]
    tree = [m.groups() for m in map(TREE_LINE.match, lines) if m]
    patches = [m.groups() for m in map(PATCHES.match, lines) if m]
    found = [m.groups() for m in map(WEIGHTS_LINE.match, lines) if m]
    freeze_lines = [m.groups() for m in map(FREEZE_LINE.match, lines) if m]
    checks["artifact_identity"] = {"interpreter": interp, "vllm_tree": tree, "patches": patches, "weights": found,
                                   "packages_freeze": freeze_lines}
    # Every installed package at its version (uv pip freeze), not only the vLLM tree (review A-M3-L2).
    if freeze_lines != [(build["freeze_sha256"][:16], "True")]:
        missing.append(f"artifact identity: packages freeze logged {freeze_lines}, frozen {build['freeze_sha256'][:16]}")
    if interp != ["True"]:
        missing.append(f"artifact identity: interpreter lines {interp}")
    want_tree = (build["vllm_tree"]["sha256"][:16], str(build["vllm_tree"]["files"]), "True")
    if tree != [want_tree]:
        missing.append(f"artifact identity: vllm-tree logged {tree}, frozen {want_tree}")
    if patches != [(str(len(build["patches"])), "")]:
        missing.append(f"artifact identity: patches logged {patches}")
    files = weights["files"]
    want = (str(len(files)), str(sum(f["bytes"] for f in files)), "0")
    if found != [want]:
        missing.append(f"artifact identity: weights logged {found}, frozen {want}")


def memory(obs, lines, missing, fail, checks):
    """Clauses 2-4 from the external 1 Hz sampler, launch to the empty scope."""
    rows = []
    path = obs / "meminfo.log"
    for line in path.read_text().splitlines() if path.is_file() else []:
        kv = dict(re.findall(r"(\w+):=(\d+)", line))
        if {"MemAvailable", "SwapTotal", "SwapFree"} <= kv.keys():
            rows.append((line.split(" ", 1)[0], kv))
    stamp = lambda marker: [STAMP.match(l).group(1) for l in lines if marker in l and STAMP.match(l)]
    gates, launches, ends = stamp("MemFree gate pass"), stamp("launching in scope"), stamp("scope empty after stop")
    checks["stamps"] = {"memfree_gate": gates, "launch": launches, "scope_empty": ends}
    gate = [tuple(int(x) / 2**20 for x in m.groups()) for m in (GATE_LINE.search(l) for l in lines) if m]
    checks["start_gate_gib"] = gate
    if len(gate) != 1 or gate[0][0] < GATE_FREE_GIB or gate[0][1] < GATE_AVAILABLE_GIB:
        missing.append(f"start gate: logged {gate}, preregistered MemFree >= {GATE_FREE_GIB} and "
                       f"MemAvailable >= {GATE_AVAILABLE_GIB} GiB")
    if len(rows) < 600 or len(gates) != 1 or len(launches) != 1 or len(ends) != 1:
        missing.append(f"clauses 2-4: {len(rows)} samples, gate/launch/scope-empty stamps {checks['stamps']}")
        return
    try:
        window = [kv for t, kv in rows if launches[0] <= t <= ends[0]]
        upto_t = [t for t, _ in rows if t <= launches[0]]
        edges = [seconds(launches[0]), *(seconds(t) for t, _ in rows if launches[0] <= t <= ends[0]), seconds(ends[0])]
        covered = (bool(window) and bool(upto_t) and seconds(launches[0]) - seconds(upto_t[-1]) <= 5
                   and max(b - a for a, b in zip(edges, edges[1:])) <= 5)
    except ValueError:
        covered = False
    upto = [kv for t, kv in rows if t <= launches[0]]
    before = [kv for t, kv in rows if gates[0] <= t <= launches[0]] or upto[-1:]
    if not covered or not before:
        missing.append("clauses 2-4: samples do not cover the launch to the empty scope at most 5 s apart, "
                       "or none within 5 s before the launch")
        return
    swap = lambda kv: (int(kv["SwapTotal"]) - int(kv["SwapFree"])) / 2**20
    low = min(int(kv["MemAvailable"]) for kv in window) / 2**20
    launch = max([int(kv["MemAvailable"]) / 2**20 for kv in before] + [g[1] for g in gate[:1]])
    peak = launch - low
    checks.update(swap_growth_gib=max(map(swap, window)) - swap(upto[-1]), mem_available_low_gib=low,
                  launch_mem_available_gib=launch, peak_used_gib=peak, admission_gib=math.ceil(peak + FLOOR_GIB),
                  samples=len(rows))
    if checks["swap_growth_gib"] > SWAP_MAX_GIB:
        fail.append("clause 2: swap growth above 0.25 GiB")
    if low < FLOOR_GIB:
        fail.append(f"clause 3: external low point below {FLOOR_GIB} GiB")
    if math.ceil(peak + FLOOR_GIB) > ADMISSION_GIB:
        fail.append(f"clause 4: admission ceil(peak + {FLOOR_GIB}) {math.ceil(peak + FLOOR_GIB)} > {ADMISSION_GIB}")


def server_side(obs, missing, fail, checks):
    files = {"kernel.log": KERNEL_KILL, "server.log": SERVER_KILL, "memwatch.log": MEMWATCH_KILL}
    hits = []
    for name, rx in files.items():
        f = obs / name
        if not f.is_file():
            missing.append(f"observe/{name}")
            continue
        hits += [f"{name}: {l.strip()}" for l in f.read_text(errors="replace").splitlines() if rx.search(l)]
    checks["kill_lines"] = hits
    if hits:
        fail.append("clause 1: kill/OOM/NVRM/Xid line")
    for name, marker in MARKERS.items():
        f = obs / name
        if f.is_file() and not any(marker.search(l) for l in f.read_text(errors="replace").splitlines()):
            missing.append(f"observe/{name}: no '{marker.pattern}' line")
    server = obs / "server.log"
    if not server.is_file():
        return
    text = server.read_text(errors="replace")
    # The configuration under test: the carrier's levers and the picked effort, and nothing else.
    nd = re.search(r"non-default args: (\{.*\})", text)
    shown = {n: n in text for n in CONFIG_NEEDLES}
    checks["configuration_lines"] = {"levers": shown, "effort_line": bool(nd and f"'reasoning_effort': '{EFFORT}'" in nd.group(1))}
    if not nd or f"'reasoning_effort': '{EFFORT}'" not in nd.group(1):
        missing.append(f"configuration: non-default args do not set reasoning_effort {EFFORT}")
    if not all(shown.values()):
        missing.append(f"configuration: lever lines {shown}")
    if "INKLING2_MAX_IMAGE_SIDE=4096" not in text:
        missing.append("configuration: patch 0004 line missing")
    # Clause 5's capacity and clause 12's worker.
    capacity = [int(m.replace(",", "")) for m in KV_CAPACITY.findall(text)]
    checks["kv_capacity_tokens"] = capacity[-1] if capacity else None
    if not capacity:
        missing.append("server.log: no KV capacity line at 262,144")
    elif capacity[-1] < CONTEXT_CAP:
        fail.append(f"clause 5: KV capacity {capacity[-1]} < {CONTEXT_CAP}")
    runtime = RUNTIME_RELEASE.findall(text)
    releases = RELEASE.findall(text)
    checks["runtime_cache_release"] = [{"root": r, "files": int(n)} for r, n in runtime]
    checks["host_cache_release"] = [{"root": r, "files": int(n), "mem_free_gib": float(f), "arena_gib": float(a),
                                     "margin_gib": float(m)} for r, n, f, a, m in releases]
    if REFUSED in text:
        fail.append("clause 12: the worker refused the KV arena (MemFree short after the release)")
    else:
        if sorted(r for r, _ in runtime) != sorted(RUNTIME_ROOTS) or any(int(n) < 1 for _, n in runtime):
            missing.append(f"clause 12: want one runtime release line under each of {RUNTIME_ROOTS}, got {runtime}")
        if len(releases) != 1 or releases[0][0] != MODEL_ROOT or int(releases[0][1]) < 1 \
                or float(releases[0][3]) < 40.0:
            missing.append(f"clause 12: want one release line under {MODEL_ROOT} before the arena, got {releases}")


def cells(ev, head, missing, fail, checks):
    manifest = load(ev / "manifest.json", missing)
    if manifest is not None:
        checks["kit_head"] = (manifest.get("git") or {}).get("head")
        if head and checks["kit_head"] != head:
            missing.append(f"kit manifest head {checks['kit_head']} is not the driver's {head}")
        vision_argv = ((manifest.get("cells") or {}).get("vision") or {}).get("argv") or []
        try:
            mode = vision_argv[vision_argv.index("--thinking-mode") + 1]
            budget = vision_argv[vision_argv.index("--max-tokens") + 1]
        except (ValueError, IndexError):
            mode = budget = None
        checks["vision_protocol"] = {"mode": mode, "max_tokens": budget}
        if (mode, budget) != (VISION_MODE, VISION_BUDGET):
            missing.append(f"vision protocol {mode}/{budget}, preregistered {VISION_MODE}/{VISION_BUDGET}")

    context = load(ev / "context" / "summary.json", missing)
    if manifest is not None and context is not None:
        cap = (manifest.get("profile") or {}).get("context_cap")
        tokens = context.get("total_tokens")
        checks["context"] = {"verdict": context.get("verdict"), "total_tokens": tokens, "context_cap": cap,
                             "max_ttft_s": context.get("max_ttft_s"),
                             "min_mem_available_gib": context.get("min_mem_available_gib")}
        if cap != CONTEXT_CAP or type(tokens) is not int:
            missing.append("context: cap or total_tokens malformed")
        elif context.get("verdict") != "PASS" or tokens < MIN_CONTEXT_TOKENS:
            fail.append("clause 5: direct 1M fill")

    probe, reference = load(ev / "probe-c2.json", missing), load(REFERENCE_PROBE, missing)
    if probe is not None and reference is not None:
        try:
            a, b = ([g["sha256"] for g in p["greedy"]] for p in (probe, reference))
        except (KeyError, TypeError):
            missing.append("probe greedy rows malformed")
        else:
            checks["text_identity_vs_screen_carrier"] = [x == y for x, y in zip(a, b)]
            if len(a) != 3 or len(b) != 3:
                missing.append("probe: expected 3 greedy rows")
            elif a != b:
                fail.append("clause 6: greedy text differs from the screen carrier's")

    toolcall = load(ev / "toolcall" / "toolcall.json", missing)
    if toolcall is not None:
        passed, total = toolcall.get("passed"), toolcall.get("total")
        checks["toolcall"] = {"passed": passed, "total": total}
        if type(passed) is not int or total != 20:
            missing.append(f"toolcall passed/total malformed: {passed}/{total}")
        elif passed < TOOLCALL_MIN:
            fail.append("clause 7: tool-call")

    agent = load(ev / "agent-gate.json", missing)
    if agent is not None:
        rows = agent.get("results") or []
        results = {r.get("test"): r.get("pass") for r in rows}
        semantics = {k: agent.get(k) for k in AGENT_GATE_SEMANTICS}
        checks["agent_gate"] = {"results": results, "semantics": semantics}
        if sorted(r.get("test") for r in rows) != sorted(AGENT_GATE_TESTS) or semantics != AGENT_GATE_SEMANTICS:
            missing.append(f"agent gate: tests {[r.get('test') for r in rows]}, semantics {semantics}")
        elif not all(v is True for v in results.values()):
            fail.append("clause 8: agent gate")

    soak = load(ev / "soak" / "soak.json", missing)
    if soak is not None:
        checks["soak"] = {k: soak.get(k) for k in ("pass", "mem_floor_gib", "mem_floor_source",
                                                   "mem_available_min_gib", "failed_gates")}
        if not isinstance(soak.get("pass"), bool):
            missing.append("soak.pass malformed")
        elif soak.get("mem_floor_gib") != SOAK_FLOOR_GIB or soak.get("mem_floor_source") != "default":
            missing.append(f"clause 9: soak not held to the repository {SOAK_FLOOR_GIB} GiB floor")
        elif not soak["pass"]:
            fail.append("clause 9: soak")

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
                fail.append("clause 10: vision")

    speed = load(ev / "speed" / "speed.json", missing)
    if speed is not None:
        levels = {c.get("ctx_tokens"): c for c in speed.get("cells") or []}
        meta = speed.get("metadata") or {}
        checks["speed"] = {str(k): {m: c.get(m) for m in ("median_decode", "median_ttft", "valid")}
                           for k, c in levels.items()}
        checks["speed_k"] = meta.get("max_tokens_per_event")
        if (speed.get("suite_valid") is not True or set(levels) != {0, 28672}
                or not all(c.get("valid") is True and number(c.get("median_decode")) for c in levels.values())
                or meta.get("max_tokens_per_event", 1) != SPEED_K
                or meta.get("decode_method") != SPEED_METHOD):
            missing.append("clause 11: speed suite or a level is not valid, or not at the profile's K "
                           "(reported numbers need a valid suite)")


def evaluate(ev):
    missing, fail, checks = [], [], {}
    freeze = git("log", "-1", "--format=%H", "--", f"{CANDIDATE}/PREREG.md")
    checks["freeze"] = freeze
    scoring_identity(freeze, missing, checks)
    obs = ev / "observe"
    driver = obs / "driver.log"
    lines = driver.read_text().splitlines() if driver.is_file() else []
    if not driver.is_file():
        missing.append("observe/driver.log")
    checks["failed_attempt_lines"] = [l for l in lines if "FAILED_ATTEMPT" in l]
    if driver.is_file() and not any(l.endswith(JOURNAL_ALIVE) for l in lines):
        missing.append(f"observe/driver.log: no '{JOURNAL_ALIVE}' line")
    head = code_identity(lines, freeze, missing, checks)
    artifact_identity(lines, freeze, missing, checks)
    logged = [m.group(1) for m in map(FIXTURE_LINE.match, lines) if m]
    want = hashlib.sha256(frozen_bytes(freeze, AGENT_FIXTURE)).hexdigest()[:16] if freeze else None
    checks["agent_fixture"] = {"logged": logged, "frozen": want}
    if logged != [want]:
        missing.append(f"agent-gate fixture: logged {logged}, the freeze's {AGENT_FIXTURE} is {want}")
    server_side(obs, missing, fail, checks)
    memory(obs, lines, missing, fail, checks)
    try:
        cells(ev, head, missing, fail, checks)
    except (AttributeError, TypeError, KeyError) as error:
        missing.append(f"cell evidence malformed ({type(error).__name__}: {error})")
    verdict = "FAIL" if fail else ("NO_RESULT" if missing or checks["failed_attempt_lines"] else "PASS")
    return {"formula": "PREREG.md", "checks": checks, "fail": fail, "missing": missing, "verdict": verdict}


def main(ev):
    out = evaluate(ev)
    (ev / "gate-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    c = out["checks"]
    print(json.dumps({"verdict": out["verdict"], "fail": out["fail"], "missing": out["missing"],
                      "peak_used_gib": c.get("peak_used_gib"), "mem_low_gib": c.get("mem_available_low_gib"),
                      "context": c.get("context"), "vision": c.get("vision")}))
    return 0 if out["verdict"] == "PASS" else 1


def preflight():
    missing, checks = [], {}
    freeze = git("log", "-1", "--format=%H", "--", f"{CANDIDATE}/PREREG.md")
    scoring_identity(freeze, missing, checks)
    print(json.dumps({"freeze": freeze, "missing": missing}))
    return 0 if not missing else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        sys.exit(preflight())
    paths = [a for a in sys.argv[1:] if a != "--mutation"]
    if len(paths) != 1:
        raise SystemExit("usage: gate_verdict.py <bundle dir> [--mutation] | --preflight")
    sys.exit(main(Path(paths[0])))
