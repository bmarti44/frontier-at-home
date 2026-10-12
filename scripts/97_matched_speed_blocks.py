#!/usr/bin/env python3
"""Matched speed comparison of two profiles: fresh-server ABBA/BAAB blocks.

AGENTS.md asks matched comparisons to use five fresh-server ABBA/BAAB blocks
with confidence intervals derived in the harness. This is the shared tool:

  plan   (--randomness HEX | --drand-after REV) --domain TEXT [--blocks 5]
         --a-profile M/F --b-profile M/F --out plan.json
         Each block's order is ABBA when sha256(domain NUL randomness block)[0]
         is even, else BAAB. --drand-after takes the first drand round
         published after commit REV (the freeze), waits for it, requires three
         relays to agree, BLS-verifies it with the bundled verifier, and keeps
         the receipt in plan.json.
  run    --plan plan.json --out DIR [--host FILE] [--port 8017] [--required-gib G]
         [--runtime-max-sec S] [--reps 3] [--release-tree DIR ...] [--abort-low-gib 8]
         [--resume] [--dry-run]
         Starts each arm fresh, in plan order, on a dev port through
         93_profile_serve.sh inside a user scope carrying the profile's
         containment (MemorySwapMax, OOMPolicy, MemoryHigh/Max). Each start is
         preceded by 03_memory_guard, sampled at 1 Hz from /proc/meminfo with
         the kernel log, measured with 30_bench_speed_blocks.py (K from the
         profile's qualification_options) at contexts 0 and 28,672, and stopped.
         Writes DIR/block-<b>/<position>-<arm>/{start.json,speed.json,...}.
         The campaign stops at the first failed start or one with a kill
         line, swap growth over 0.25 GiB or a low point under --abort-low-gib.
  score  --out DIR
         Reduces each block to its A and B means and gives paired log ratios
         B/A with one-sided 95% bounds (Student t, df = blocks - 1). Writes
         DIR/summary.json. Any missing, failed, unsafe, invalid, non-finite,
         out-of-order or unexpected start makes the verdict NO_RESULT.

The tool reports bounds; the thresholds belong to the gate that cites it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/lib"))

CONTEXTS = (0, 28672)
METRICS = {  # name: (ctx, field)
    "decode_0": (0, "decode"),
    "decode_28672": (28672, "decode"),
    "prefill_28672": (28672, "prefill"),
    "ttft_0": (0, "ttft"),
}
# One-sided 95% Student t critical values by degrees of freedom.
T95 = {1: 6.313751514675043, 2: 2.919985580355516, 3: 2.3533634348018264,
       4: 2.131846786326649, 5: 2.015048372669157, 6: 1.9431802803927816,
       7: 1.894578605061305, 8: 1.8595480375228421, 9: 1.8331129326536335}
MAX_SWAP_GROWTH_GIB = 0.25
# re.I: the kernel's global OOM line is "Out of memory: Killed process ..." (as
# the smoke and gate scorers match it).
KERNEL_KILL = re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I)
SERVER_KILL = re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid")
MEMWATCH_KILL = re.compile(r"BREACH|EMERGENCY_STOP")
# The engine's arena guard refuses cleanly and names NVRM as what it prevents;
# a refusal fails the start, but it is not a driver kill line.
GUARD_REFUSAL = "refusing to allocate the KV arena"
KIB_PER_GIB = 1024 * 1024


def kill_lines(fname: str, text: str) -> list[str]:
    pattern = {"kernel.log": KERNEL_KILL, "server.log": SERVER_KILL, "memwatch.log": MEMWATCH_KILL}[fname]
    return [f"{fname}: {line[:200]}" for line in text.splitlines()
            if pattern.search(line) and not (fname == "server.log" and GUARD_REFUSAL in line)]


def release_trees(roots: list[str]) -> list[str]:
    """Drop the unmapped page cache of every file under each root (no privilege)."""
    from checkpoint_cache import release_tree
    return [f"page cache released under {root}: {release_tree(root)} files" for root in roots]


DRAND_GENESIS, DRAND_PERIOD = 1595431050, 30   # drand mainnet (default chain)
DRAND_RELAYS = ("https://api.drand.sh", "https://api2.drand.sh", "https://api3.drand.sh")
DRAND_VERIFIER = REPO / "scripts/103_verify_drand_receipt_bundle.mjs"
DRAND_FIELDS = ("round", "randomness", "signature", "previous_signature")


def drand_publication_time(number: int) -> int:
    return DRAND_GENESIS + (number - 1) * DRAND_PERIOD


def first_drand_round_after(timestamp: int) -> int:
    return (timestamp - DRAND_GENESIS) // DRAND_PERIOD + 2


def fetch_drand_round(relay: str, number: int) -> dict:
    request = urllib.request.Request(f"{relay}/public/{number}", headers={"User-Agent": "frontier-at-home"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def verify_drand_round(doc: dict) -> None:
    # Run it with the interpreter its shebang names (the bundle is not executable).
    with open(DRAND_VERIFIER) as fh:
        node = fh.readline().removeprefix("#!").strip()
    result = subprocess.run([node, str(DRAND_VERIFIER), *(str(doc[k]) for k in DRAND_FIELDS)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"drand BLS verification failed: {result.stderr.strip() or result.stdout.strip()}")


def drand_receipt(frozen_at: int, commit: str, fetch=fetch_drand_round, verify=verify_drand_round,
                  now=time.time, sleep=time.sleep) -> dict:
    number = first_drand_round_after(frozen_at)
    while now() < drand_publication_time(number) + 2:
        sleep(max(1.0, drand_publication_time(number) + 2 - now()))
    docs = [fetch(relay, number) for relay in DRAND_RELAYS]
    picked = [{k: d.get(k) for k in DRAND_FIELDS} for d in docs]
    if any(p != picked[0] for p in picked) or picked[0]["round"] != number:
        raise ValueError(f"drand relays disagree on round {number}: {picked}")
    verify(picked[0])
    return {**picked[0], "relays": list(DRAND_RELAYS), "frozen_commit": commit, "frozen_at_unix": frozen_at,
            "rule": "first drand round published after the frozen commit's committer time"}


# --- plan ---------------------------------------------------------------------

def block_orders(randomness_hex: str, blocks: int, domain: str) -> list[str]:
    seed = bytes.fromhex(randomness_hex)
    if len(seed) < 16:
        raise ValueError("randomness must carry at least 128 bits")
    return ["ABBA" if hashlib.sha256(domain.encode() + b"\0" + seed + bytes([b])).digest()[0] % 2 == 0
            else "BAAB" for b in range(blocks)]


def schedule(orders: list[str]) -> list[tuple[int, int, str]]:
    return [(b, p, arm) for b, order in enumerate(orders) for p, arm in enumerate(order)]


# --- run ----------------------------------------------------------------------

def _profile(profile: str, host: dict) -> dict:
    import profile_resolver as resolver
    slug, name = profile.split("/", 1)
    p = resolver.load_profile(slug, name + ".json")
    model = resolver.load_model(slug)
    rendered = resolver.resolve(p, model, host)
    argv = rendered["argv"]
    artifact = model["artifacts"][p["artifact_roles"]["model"]]
    tokenizer = next((s for s in artifact.get("shards", []) if s["path"].endswith("/tokenizer.json")), None)
    model_dir = artifact["path"].replace("{model_root}", host["paths"]["model_root"])
    return {
        "rendered": rendered,
        "k": (p.get("qualification_options") or {}).get("speed_max_tokens_per_event"),
        "stack": p["bench"]["stack_label"],
        "slug": profile.replace("/", "-"),
        "tokenizer": (model_dir + "/tokenizer.json", tokenizer["sha256"]) if tokenizer else None,
        "model_id": argv[argv.index("--served-model-name") + 1] if "--served-model-name" in argv else "default",
    }


def _commands(profile: dict, name: str, unit: str, args) -> tuple[list[str], list[str]]:
    props = profile["rendered"]["systemd"]["properties"]
    launch = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}"]
    for key in ("MemoryHigh", "MemoryMax", "MemorySwapMax", "OOMPolicy"):
        if key in props:
            launch += ["-p", f"{key}={props[key]}"]
    launch += ["-p", f"RuntimeMaxSec={args.runtime_max_sec}", "--",
               "bash", str(REPO / "scripts/93_profile_serve.sh"), "--profile", name,
               "--host", str(args.host), "--port", str(args.port), "start"]
    k = profile["k"]
    bench = [sys.executable, str(REPO / ("scripts/30_bench_speed_blocks.py" if k else "scripts/30_bench_speed.py")),
             "--base-url", f"http://127.0.0.1:{args.port}", "--out", "{out}/speed.json",
             "--stack-label", profile["stack"], "--reps", str(args.reps), "--warmup", "1",
             "--max-tokens", "384", "--min-completion-tokens", "200", "--seed", "42",
             "--context-levels", ",".join(str(c) for c in CONTEXTS), "--ignore-eos-supported",
             "--request-timeout", "900", "--extra-body", '{"min_tokens":320}',
             "--model-id", profile["model_id"]]
    if k:
        bench += ["--max-tokens-per-event", str(k)]
    if profile["tokenizer"]:
        bench += ["--output-tokenizer-path", profile["tokenizer"][0],
                  "--output-tokenizer-sha256", profile["tokenizer"][1]]
    return launch, bench


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# The 1 Hz sampler's fields (KiB) and line format; scorers parse exactly these.
MEMINFO_KEYS = ("MemAvailable", "MemFree", "SwapFree", "Cached")


def _meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, value = line.split(":", 1)
            if key in MEMINFO_KEYS:
                out[key] = int(value.split()[0])
    return out


def meminfo_line(stamp: str, sample: dict[str, int]) -> str:
    return stamp + " " + " ".join(f"{k}:={v}" for k, v in sample.items())


def _run_start(step, plan, profiles, args, out: Path, log) -> None:
    block, position, arm = step
    name = plan["a_profile" if arm == "A" else "b_profile"]
    profile = profiles[name]
    d = out / f"block-{block}" / f"{position}-{arm}"
    # The guard runs before the start directory exists: a guard timeout leaves
    # nothing behind and --resume retries the same start.
    released = release_trees(args.release_tree)
    guard = subprocess.run([sys.executable, str(REPO / "scripts/03_memory_guard.py"), "--required-gib",
                            str(args.required_gib), "--stable-samples", "3", "--timeout-seconds", "600"],
                           capture_output=True, text=True)
    if guard.returncode != 0:
        raise SystemExit(f"{d}: memory guard did not pass; nothing was started\n{guard.stdout}{guard.stderr}")
    d.mkdir(parents=True)
    (d / "guard.txt").write_text("".join(r + "\n" for r in released) + guard.stdout + guard.stderr)
    unit = f"matched-{block}-{position}-{arm}-{os.getpid()}"
    launch, bench = _commands(profile, name, unit, args)
    bench = [x.replace("{out}", str(d)) for x in bench]
    start = {"block": block, "position": position, "arm": arm, "profile": name,
             "unit": unit, "launch": launch, "bench": bench, "failed": None}
    logs = Path.home() / ".frontier-at-home/logs"
    server_log, memwatch_log = logs / f"{profile['slug']}.server.log", logs / f"{profile['slug']}.memwatch.log"
    offsets = {p: (p.stat().st_size if p.exists() else 0) for p in (server_log, memwatch_log)}
    samples: list[tuple[str, dict]] = []
    stop = threading.Event()

    def sampler():
        with open(d / "meminfo.log", "w") as fh:
            while not stop.is_set():
                m = _meminfo()
                samples.append((_stamp(), m))
                fh.write(meminfo_line(_stamp(), m) + "\n")
                fh.flush()
                stop.wait(1.0)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    kernel = subprocess.Popen(["journalctl", "-k", "-f", "-o", "short-iso", "--since", "now"],
                              stdout=open(d / "kernel.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(1.5)
    launch_index = len(samples) - 1
    start["launched_at"] = _stamp()
    log(f"block {block} position {position} arm {arm}: launching {name}")
    t0 = time.monotonic()
    serve = launch[launch.index("--") + 1:]
    try:
        rc = subprocess.run(launch, stdout=open(d / "launch.log", "w"), stderr=subprocess.STDOUT).returncode
        start["ready_s"] = round(time.monotonic() - t0, 1)
        if rc != 0:
            start["failed"] = f"launch exit {rc}"
        else:
            b = subprocess.run(bench, stdout=open(d / "bench.log", "w"), stderr=subprocess.STDOUT)
            if b.returncode != 0:
                start["failed"] = f"bench exit {b.returncode}"
    finally:
        subprocess.run(serve[:-1] + ["stop"], stdout=open(d / "stop.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if subprocess.run(["systemctl", "--user", "is-active", "-q", f"{unit}.scope"]).returncode != 0:
                break
            time.sleep(1)
        else:
            start["failed"] = start["failed"] or "scope still active after stop"
            subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"])
        time.sleep(2)
        stop.set()
        thread.join(timeout=5)
        kernel.terminate()
        for path, name in ((server_log, "server.log"), (memwatch_log, "memwatch.log")):
            text = path.read_bytes()[offsets[path]:].decode(errors="replace") if path.exists() else ""
            (d / name).write_text(text)
    for fname, marker in (("server.log", "session start"), ("memwatch.log", "ARMED")):
        if marker not in (d / fname).read_text(errors="replace"):
            start["failed"] = start["failed"] or f"{fname} lacks {marker!r}"
    base = samples[launch_index][1]
    after = [m for _, m in samples[launch_index:]]
    kills = []
    for fname in ("kernel.log", "server.log", "memwatch.log"):
        path = d / fname
        if path.exists():
            kills += kill_lines(fname, path.read_text(errors="replace"))
    start.update({
        "kill_lines": kills,
        "launch_mem_available_gib": base["MemAvailable"] / KIB_PER_GIB,
        "low_mem_available_gib": min(m["MemAvailable"] for m in after) / KIB_PER_GIB,
        "swap_growth_gib": (base["SwapFree"] - min(m["SwapFree"] for m in after)) / KIB_PER_GIB,
    })
    (d / "start.json").write_text(json.dumps(start, indent=1) + "\n")
    log(f"block {block} position {position} arm {arm}: failed={start['failed']} "
        f"kills={len(kills)} low={start['low_mem_available_gib']:.2f}")
    return start


def unsafe_start(start: dict, floor_gib: float) -> list[str]:
    """Why a finished start must stop the campaign (empty when it may continue)."""
    reasons = []
    if start.get("failed"):
        reasons.append(f"failed: {start['failed']}")
    if start.get("kill_lines"):
        reasons.append(f"kill lines: {start['kill_lines'][:3]}")
    swap, low = start.get("swap_growth_gib"), start.get("low_mem_available_gib")
    if not _is_finite(swap) or swap > MAX_SWAP_GROWTH_GIB:
        reasons.append(f"swap growth {swap!r} GiB")
    if not _is_finite(low) or low < floor_gib:
        reasons.append(f"low point {low!r} GiB (floor {floor_gib})")
    return reasons


def existing_start(d: Path, resume: bool) -> dict | None:
    """A start already on disk: None when there is none. With --resume, its
    record (an unreadable one as {}, which is never safe); otherwise, or when it
    died before writing start.json, the campaign cannot continue."""
    if not d.exists():
        return None
    if not resume:
        raise SystemExit(f"{d} exists; use a fresh --out (or --resume after a clean stop)")
    if not (d / "start.json").exists():
        raise SystemExit(f"{d} has no start.json: that start died; the campaign cannot resume")
    try:
        return json.loads((d / "start.json").read_text())
    except (OSError, ValueError):
        return {}


def run_steps(steps, run_one, floor_gib: float, log) -> None:
    """Run each step; stop the campaign at the first failed or unsafe start, so
    no further large load follows an NVRM/Xid/OOM line, swap growth or a low
    point under the floor. run_one returns the start's record: a new start's,
    or (--resume) the one on disk, so a resume stops at an unsafe start too."""
    for step in steps:
        start = run_one(step)
        reasons = unsafe_start(start, floor_gib)
        if reasons:
            log(f"campaign stopped after block {start.get('block')} position {start.get('position')} "
                f"arm {start.get('arm')}: {'; '.join(reasons)}")
            raise SystemExit(f"campaign stopped at an unsafe or failed start: {'; '.join(reasons)}")


def exit_on_sigterm() -> None:
    """SIGTERM raises SystemExit(143), so a start unwinds through its finally:
    the engine is stopped and its scope waited for, not orphaned until RuntimeMaxSec."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


def cmd_run(args) -> int:
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    import profile_resolver as resolver
    host = resolver.load_host(args.host)
    if not args.dry_run and Path(host["paths"]["repo"]).resolve() != REPO:
        raise SystemExit(f"host file repo {host['paths']['repo']} is not this checkout {REPO}")
    profiles = {n: _profile(n, host) for n in {plan["a_profile"], plan["b_profile"]}}
    steps = schedule(plan["blocks"])
    if args.dry_run:
        for block, position, arm in steps:
            name = plan["a_profile" if arm == "A" else "b_profile"]
            launch, bench = _commands(profiles[name], name, f"matched-{block}-{position}-{arm}", args)
            print(json.dumps({"block": block, "position": position, "arm": arm, "profile": name,
                              "release_trees": args.release_tree, "launch": launch, "bench": bench}))
        return 0
    # Hold the system inference lock a serving engine holds: no start beside a
    # live engine, and no switch start beside a campaign. os.open is
    # non-inheritable, so no engine or memwatch outlives the run holding it.
    import fcntl
    lock = os.open(args.inference_lock, os.O_WRONLY | os.O_CREAT, 0o660)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{args.inference_lock} is held: an engine is serving; stop it first") from None
    exit_on_sigterm()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        if json.loads((out / "plan.json").read_text()) != plan:
            raise SystemExit(f"{out}/plan.json differs from --plan; use a fresh --out")
    else:
        (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")

    def log(message: str) -> None:
        line = f"{_stamp()} {message}"
        print(line, flush=True)
        with open(out / "run.log", "a") as fh:
            fh.write(line + "\n")

    def run_one(step):
        existing = existing_start(out / f"block-{step[0]}" / f"{step[1]}-{step[2]}", args.resume)
        return existing if existing is not None else _run_start(step, plan, profiles, args, out, log)

    run_steps(steps, run_one, args.abort_low_gib, log)
    log("all starts done")
    return 0


# --- score --------------------------------------------------------------------

def _is_finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _finite(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} is not finite: {value!r}")
    return float(value)


def _start_metrics(d: Path, expected_arm: str, expected_profile: str, reasons: list[str]) -> dict | None:
    label = f"{d.parent.name}/{d.name}"
    try:
        start = json.loads((d / "start.json").read_text())
        speed = json.loads((d / "speed.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        reasons.append(f"{label}: missing or malformed evidence ({error})")
        return None
    bad = []
    if start.get("arm") != expected_arm:
        bad.append(f"arm {start.get('arm')!r} is out of plan order (want {expected_arm})")
    if start.get("profile") != expected_profile:
        bad.append(f"profile {start.get('profile')!r} is not the plan's {expected_profile!r}")
    if not _is_finite(start.get("low_mem_available_gib")):
        bad.append(f"low point {start.get('low_mem_available_gib')!r} is not finite")
    if start.get("failed"):
        bad.append(f"failed: {start['failed']}")
    if start.get("kill_lines"):
        bad.append(f"kill lines: {len(start['kill_lines'])}")
    swap = start.get("swap_growth_gib")
    if not isinstance(swap, (int, float)) or not math.isfinite(swap) or swap > MAX_SWAP_GROWTH_GIB:
        bad.append(f"swap growth {swap!r} GiB")
    cells = {c.get("ctx_tokens"): c for c in speed.get("cells", [])}
    if set(cells) != set(CONTEXTS) or not speed.get("suite_valid") \
            or not all(c.get("valid") for c in cells.values()):
        bad.append("speed cells invalid or incomplete")
    metrics = {}
    if not bad:
        try:
            for name, (ctx, field) in METRICS.items():
                cell = cells[ctx]
                if field == "decode":
                    metrics[name] = _finite(cell.get("median_decode"), f"{name}")
                elif field == "ttft":
                    metrics[name] = _finite(cell.get("median_ttft"), f"{name}")
                else:
                    reps = [_finite(r.get("prefill_tok_s"), f"{name} rep") for r in cell.get("reps", [])]
                    if not reps:
                        raise ValueError(f"{name} has no reps")
                    metrics[name] = statistics.median(reps)
                if metrics[name] <= 0:
                    raise ValueError(f"{name} is not positive and finite")
        except ValueError as error:
            bad.append(str(error))
    if bad:
        reasons += [f"{label}: {b}" for b in bad]
        return None
    return metrics


def score(out: Path) -> dict:
    plan = json.loads((out / "plan.json").read_text())
    orders = plan["blocks"]
    reasons: list[str] = []
    expected = {(f"block-{b}", f"{p}-{arm}"): arm for b, p, arm in schedule(orders)}
    present = {(d.parent.name, d.name) for d in out.glob("block-*/*") if d.is_dir()}
    for extra in sorted(present - set(expected)):
        reasons.append(f"{extra[0]}/{extra[1]}: unexpected start directory")
    for block_dir in out.glob("block-*"):
        if block_dir.name not in {f"block-{b}" for b in range(len(orders))}:
            reasons.append(f"{block_dir.name}: unexpected block directory")
    per_start = {}
    for key, arm in expected.items():
        per_start[key] = _start_metrics(out / key[0] / key[1], arm,
                                        plan["a_profile" if arm == "A" else "b_profile"], reasons)
    result = {"schema_version": 1, "plan": {k: plan[k] for k in ("domain", "randomness", "blocks",
                                                                "a_profile", "b_profile")},
              "method": "per block: arithmetic mean of each arm's two fresh starts; paired log ratio "
                        "B/A per block; ratio = exp(mean); one-sided 95% bounds exp(mean -/+ t95(df) "
                        "* sd / sqrt(n)), df = blocks - 1"}
    if reasons:
        result.update(verdict="NO_RESULT", reasons=reasons)
        return result
    n = len(orders)
    if n - 1 not in T95:
        result.update(verdict="NO_RESULT", reasons=[f"no t value for {n} blocks"])
        return result
    t = T95[n - 1]
    metrics = {}
    for name in METRICS:
        logs, blocks = [], []
        for b, order in enumerate(orders):
            vals = {"A": [], "B": []}
            for p, arm in enumerate(order):
                vals[arm].append(per_start[(f"block-{b}", f"{p}-{arm}")][name])
            a, bb = statistics.fmean(vals["A"]), statistics.fmean(vals["B"])
            logs.append(math.log(bb / a))
            blocks.append({"block": b, "order": order, "a_mean": a, "b_mean": bb})
        mean = statistics.fmean(logs)
        sem = statistics.stdev(logs) / math.sqrt(n)
        metrics[name] = {"ratio": math.exp(mean), "lower_95": math.exp(mean - t * sem),
                         "upper_95": math.exp(mean + t * sem), "log_ratios": logs,
                         "t95": t, "blocks": blocks}
    result.update(verdict="COMPLETE", reasons=[], metrics=metrics)
    return result


def cmd_score(args) -> int:
    out = Path(args.out)
    result = score(out)
    (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"verdict": result["verdict"], "reasons": result["reasons"][:10],
                      **({k: {m: round(v[m], 4) for m in ("ratio", "lower_95", "upper_95")}
                          for k, v in result.get("metrics", {}).items()})}))
    return 0 if result["verdict"] == "COMPLETE" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--randomness")
    source.add_argument("--drand-after", metavar="REV", help="frozen commit; draw the first drand round after it")
    p.add_argument("--domain", required=True)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--a-profile", required=True)
    p.add_argument("--b-profile", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--host", type=Path, default=None, help="host file (default: resolver's choice)")
    r.add_argument("--port", type=int, default=8017)
    r.add_argument("--required-gib", type=float, default=114.5)
    r.add_argument("--runtime-max-sec", type=int, default=3600)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--abort-low-gib", type=float, default=8.0,
                   help="stop the campaign after a start whose MemAvailable low point fell below this")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--inference-lock", default="/run/lock/frontier-at-home/inference.lock")
    r.add_argument("--release-tree", action="append", default=[],
                   help="before each start, drop this tree's unmapped page cache (repeatable); "
                        "name every runtime tree the compared profiles use")
    r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "plan":
        if args.blocks - 1 not in T95:
            parser.error(f"--blocks must be 2..{max(T95) + 1}")
        receipt = None
        randomness = args.randomness
        if args.drand_after:
            commit, frozen_at = subprocess.run(
                ["git", "-C", str(REPO), "show", "-s", "--format=%H %ct", args.drand_after],
                check=True, capture_output=True, text=True).stdout.split()
            receipt = drand_receipt(int(frozen_at), commit)
            randomness = receipt["randomness"]
        plan = {"schema_version": 1, "domain": args.domain, "randomness": randomness,
                "blocks": block_orders(randomness, args.blocks, args.domain),
                "a_profile": args.a_profile, "b_profile": args.b_profile, "created_at": _stamp(),
                **({"drand": receipt} if receipt else {})}
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        print(json.dumps(plan))
        return 0
    return cmd_run(args) if args.cmd == "run" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
