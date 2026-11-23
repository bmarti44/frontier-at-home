#!/usr/bin/env python3
"""Fixed M5 verdict for Inkling 2 Mini per README.md in this directory: the holdout draw's
validity, the accuracy comparison with the default (qwen38fn), and the owner's promotion rule.

Usage: holdout_verdict.py               -> writes holdout-summary.json here, prints the verdict
       holdout_verdict.py --self-test   -> synthetic bundles, each must give its expected verdict
       holdout_verdict.py --preflight   -> exit 0 iff the working tree is this draw's freeze under
                                           HOLDOUT_FROZEN (run-holdout.sh calls it before the stop)
Verdicts: PROMOTE (faster and more accurate), KEEP_DEFAULT (a valid draw that misses any leg),
NO_RESULT (the draw or M3's evidence is missing, malformed or not the declared run).
"""
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
C2 = HERE.parent / "qualify-2026-11-23-c2"
LEDGER = CODE / "results/holdout-ledger.json"
HOLDOUT = HERE.relative_to(CODE).as_posix()
# This draw's own inputs; candidate 2's FROZEN covers the serving configuration and the kit.
# The default's holdout draw: its row is the bar, and its acc files fix the questions.
DEFAULT_DRAW = "results/qwen38fn-gates/holdout-2026-10-11"
HOLDOUT_FROZEN = (f"{HOLDOUT}/README.md", f"{HOLDOUT}/run-holdout.sh", f"{HOLDOUT}/holdout_verdict.py",
                  "scripts/31_bench_accuracy.py", "vendor/official-encoding", "configs/pins/humaneval-runtime.json",
                  f"{DEFAULT_DRAW}/accuracy")
HEAD = re.compile(r"^\S+ head ([0-9a-f]{40}) dirty=(\d+)$", re.M)
STACK = "inkling-2-mini-cuda-spark-128g-1m"
MODEL = "inkling-2-mini"
EFFORT = "low"                                      # ../tune-screen-2026-11-23/effort-a6/pick.json
ENCODER = "inkling2"
# 31_bench_accuracy.py renders GSM8K and MMLU-Pro through the encoder at the effort, and sends
# HumanEval raw (a code continuation, no chat wrapper and no effort), as for the default.
RENDERING = {"gsm8k": f"official-encoder-{ENCODER}-thinking-effort-{EFFORT}",
             "mmlu-pro": f"official-encoder-{ENCODER}-thinking-effort-{EFFORT}", "humaneval": "raw"}
SUITES = {"gsm8k": ("holdout", 100), "mmlu-pro": ("holdout", 247), "humaneval": ("all", 164)}
LEDGERED = ("gsm8k", "mmlu-pro")                    # 31_bench_accuracy.py ledgers holdout splits only
# The default's holdout row: results/qwen38fn-gates/holdout-2026-10-11/RESULT.md, same questions.
DEFAULT = {"gsm8k": 98, "mmlu-pro": 218, "humaneval": 138}
DEFAULT_SUM = 454
# The same questions as the default's draw: each suite's rowset_sha256 in its acc-<suite>.json.
ROWSETS = {suite: json.loads((CODE / DEFAULT_DRAW / "accuracy" / f"acc-{suite}.json").read_text())["rowset_sha256"]
           for suite in ("gsm8k", "mmlu-pro", "humaneval")}
WORSE_Z = -1.645
# The owner's "faster": the default's published c10 row (README), each leg at least equal.
FASTER = {"prefill_28672": 2711.5, "decode_0": 34.07, "decode_28672": 40.92}
MMMU_MIN, TOOLCALL_MIN = 64, 19
DRAW_LINE = "probe identical to candidate 2 and KV covers 1M; drawing the holdout at effort low"
JOURNAL_ALIVE = "kernel journal alive after the engine stop"
# What this draw must be. real_evidence() passes the default's own draw through the same readers.
DECLARED = {"model": MODEL, "stack": STACK, "encoder": ENCODER, "rendering": RENDERING, "observe": "observe",
            "lines": (DRAW_LINE, JOURNAL_ALIVE, "scope empty after stop"), "tolerate": ()}
KERNEL_KILL = re.compile(r"out of memory|oom-kill|oom_reaper|NVRM|Xid", re.I)
SERVER_KILL = re.compile(r"CUDA out of memory|OutOfMemoryError|NVRM|Xid")
MEMWATCH_KILL = re.compile(r"BREACH|EMERGENCY_STOP|FAIL_CLOSED")
TRANSPORT = re.compile(r"^invalid: ")                # an exception, not a scored answer
TIMED_OUT = re.compile(r"timed out|TimeoutError", re.I)   # the clock is part of the contract: scored wrong


class Bad(Exception):
    pass


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise Bad(f"{path.name}: {type(error).__name__}") from None


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def worse_z(correct: int, n: int, reference: int) -> float:
    """One-sided pooled two-proportion z of correct/n against reference/n (as the default's holdout)."""
    pooled = (correct + reference) / (2 * n)
    se = math.sqrt(pooled * (1 - pooled) * 2 / n)
    return (correct - reference) / n / se if se else 0.0


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(CODE), *args], capture_output=True, text=True).stdout.strip()


def unchanged(base: str, head: str | None, paths) -> bool:
    """No change to paths between base and head (head None: the working tree, untracked included)."""
    if not base:
        return False
    if head is None:
        return not git("status", "--porcelain", "--", *paths) and \
            subprocess.run(["git", "-C", str(CODE), "diff", "--quiet", base, "--", *paths]).returncode == 0
    return subprocess.run(["git", "-C", str(CODE), "diff", "--quiet", base, head, "--", *paths]).returncode == 0


def m3_frozen():
    """Candidate 2's FROZEN paths and freeze commit, from its own scorer."""
    spec = importlib.util.spec_from_file_location("m3_gate_verdict", C2 / "gate_verdict.py")
    gv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gv)
    return gv.FROZEN, git("log", "-1", "--format=%H", "--", f"{gv.CANDIDATE}/PREREG.md")


def code_identity(driver: str) -> dict:
    heads = HEAD.findall(driver)
    if len(heads) != 1 or heads[0][1] != "0":
        raise Bad(f"driver.log head lines {heads}, want one clean head")
    head = heads[0][0]
    freeze = git("log", "-1", "--format=%H", "--", f"{HOLDOUT}/README.md")
    frozen, m3_freeze = m3_frozen()
    if not unchanged(freeze, head, HOLDOUT_FROZEN):
        raise Bad(f"head {head[:12]} differs from this draw's freeze {freeze[:12]} under HOLDOUT_FROZEN")
    if not unchanged(m3_freeze, head, frozen):
        raise Bad(f"head {head[:12]} differs from candidate 2's freeze {m3_freeze[:12]} under its FROZEN")
    return {"head": head, "freeze": freeze, "m3_freeze": m3_freeze}


def draw(ev: Path, ledger: Path, identity=code_identity, declared=None) -> dict:
    d = declared or DECLARED
    obs = ev / d["observe"]
    driver = (obs / "driver.log").read_text() if (obs / "driver.log").is_file() else ""
    if not driver:
        raise Bad("observe/driver.log missing")
    identity(driver)
    for marker in ("FAILED_ATTEMPT", ):
        if marker in driver:
            raise Bad(f"driver.log has {marker}")
    for line in d["lines"]:
        if line not in driver:
            raise Bad(f"driver.log has no '{line}' line")
    kills = []
    for name, rx in (("kernel.log", KERNEL_KILL), ("server.log", SERVER_KILL), ("memwatch.log", MEMWATCH_KILL)):
        if not (obs / name).is_file():
            raise Bad(f"observe/{name} missing")
        kills += [f"{name}: {l.strip()}" for l in (obs / name).read_text(errors="replace").splitlines()
                  if rx.search(l) and not any(t in l for t in d["tolerate"])]
    if kills:
        raise Bad(f"kill line (FAILED_ATTEMPT, the rule is not applied): {kills[0][:200]}")
    kit = load(ev / "summary.json").get("cells", {}).get("accuracy") or {}
    if kit.get("status") != "OK" or kit.get("suites_not_run_after_failure"):
        raise Bad(f"kit accuracy cell status {kit.get('status')}, not run {kit.get('suites_not_run_after_failure')}")
    entries = load(ledger)
    rows = {}
    for suite, (split, n) in SUITES.items():
        acc = load(ev / "accuracy" / f"acc-{suite}.json")
        gen = acc.get("generation") or {}
        if (acc.get("suite"), acc.get("split"), acc.get("n"), acc.get("model"), acc.get("stack_label"),
                acc.get("ledger_namespace"), acc.get("rowset_sha256")) != (suite, split, n, d["model"], d["stack"],
                                                                           "", ROWSETS[suite]) \
                or gen.get("encoder") != d["encoder"] or gen.get("max_tokens") != 16384 \
                or type(acc.get("correct")) is not int or not 0 <= acc["correct"] <= n:
            raise Bad(f"acc-{suite}.json is not the declared holdout run")
        transcripts = sorted((ev / "accuracy" / "transcripts" / suite).glob("*.json"))
        if len(transcripts) != n:
            raise Bad(f"{suite}: {len(transcripts)} transcripts for {n} rows")
        errors = []
        for t in transcripts:
            row = load(t)
            if row.get("rendering") != d["rendering"][suite]:
                raise Bad(f"{suite}/{t.name} rendering {row.get('rendering')!r}, want {d['rendering'][suite]!r}")
            reason = str(row.get("reason", ""))
            if TRANSPORT.match(reason) and not TIMED_OUT.search(reason):
                errors.append(reason[:120])
        if errors:
            raise Bad(f"{suite}: {len(errors)} transport/server errors (FAILED_ATTEMPT), first {errors[0]!r}")
        if suite in LEDGERED:
            phases = sorted(e.get("phase") for e in entries if isinstance(e, dict)
                            and e.get("stack_label") == d["stack"] and e.get("suite") == suite
                            and e.get("ledger_namespace") == "")
            done = [e for e in entries if isinstance(e, dict) and e.get("stack_label") == d["stack"]
                    and e.get("suite") == suite and e.get("phase") == "completed"]
            if phases != ["completed", "started"] or (done[0].get("n"), done[0].get("correct")) != (n, acc["correct"]):
                raise Bad(f"ledger: {suite} at {d['stack']} has phases {phases}, want one started and one completed "
                          "matching acc-{suite}.json")
        rows[suite] = {"correct": acc["correct"], "n": n, "invalid": acc.get("invalid_count"),
                       "default": DEFAULT[suite], "delta": acc["correct"] - DEFAULT[suite],
                       "z": worse_z(acc["correct"], n, DEFAULT[suite])}
    return rows


def m3(c2: Path) -> dict:
    gate = load(c2 / "gate-summary.json")
    if gate.get("verdict") != "PASS":
        raise Bad(f"M3 candidate 2 verdict {gate.get('verdict')}, not PASS")
    levels = (((load(c2 / "summary.json").get("cells") or {}).get("speed") or {}).get("result") or {}).get("levels") or {}
    try:
        values = {"prefill_28672": levels["28672"]["median_prefill_tok_s"],
                  "decode_0": levels["0"]["median_decode_tok_s"],
                  "decode_28672": levels["28672"]["median_decode_tok_s"]}
    except (KeyError, TypeError):
        raise Bad("M3 kit speed levels malformed") from None
    if not all(finite(v) for v in values.values()) or not all(levels[k].get("valid") is True for k in ("0", "28672")):
        raise Bad("M3 kit speed values not valid")
    vision, tool = load(c2 / "vision" / "summary.json"), load(c2 / "toolcall" / "toolcall.json")
    if vision.get("n") != 100 or type(vision.get("correct")) is not int:
        raise Bad("M3 vision summary malformed")
    if tool.get("total") != 20 or type(tool.get("passed")) is not int:
        raise Bad("M3 toolcall malformed")
    return {"speed": values, "mmmu": vision["correct"], "toolcall": tool["passed"]}


def evaluate(ev: Path, c2: Path, ledger: Path, identity=code_identity) -> dict:
    out = {"formula": "README.md", "default": {"holdout": DEFAULT, "sum": DEFAULT_SUM, "faster": FASTER,
                                               "mmmu_min": MMMU_MIN, "toolcall_min": TOOLCALL_MIN}}
    try:
        rows, measured = draw(ev, ledger, identity), m3(c2)
    except Bad as error:
        return {**out, "verdict": "NO_RESULT", "why": str(error)}
    total = sum(r["correct"] for r in rows.values())
    worse = [s for s, r in rows.items() if r["z"] < WORSE_Z]
    legs = {
        "faster": {k: {"value": measured["speed"][k], "target": t, "ok": measured["speed"][k] >= t}
                   for k, t in FASTER.items()},
        "more_accurate": {"sum": total, "default_sum": DEFAULT_SUM, "sum_ok": total > DEFAULT_SUM,
                          "significantly_worse": worse},
        "mmmu": {"value": measured["mmmu"], "min": MMMU_MIN, "ok": measured["mmmu"] >= MMMU_MIN},
        "toolcall": {"value": measured["toolcall"], "min": TOOLCALL_MIN, "ok": measured["toolcall"] >= TOOLCALL_MIN},
    }
    faster = all(v["ok"] for v in legs["faster"].values())
    accurate = total > DEFAULT_SUM and not worse and legs["mmmu"]["ok"] and legs["toolcall"]["ok"]
    return {**out, "rows": rows, "legs": legs, "faster": faster, "more_accurate": accurate,
            "verdict": "PROMOTE" if faster and accurate else "KEEP_DEFAULT"}


# ---- self-test ---------------------------------------------------------------------------------

def _bundle(root: Path, scores=(99, 225, 140), speed=(3000.0, 36.0, 42.0), mmmu=66, tool=20, **kw) -> tuple:
    ev, c2 = root / "holdout", root / "c2"
    obs = ev / "observe"
    obs.mkdir(parents=True)
    lines = [f"head {'a' * 40} dirty={kw.get('dirty', 0)}", DRAW_LINE, "scope empty after stop", JOURNAL_ALIVE]
    if kw.get("attempt"):
        lines.append("FAILED_ATTEMPT kernel journal not running")
    if kw.get("nodraw"):
        lines.remove(DRAW_LINE)
    (obs / "driver.log").write_text("\n".join(f"2026-11-24T00:00:00Z {l}" for l in lines) + "\n")
    (obs / "kernel.log").write_text("NVRM: Xid 31\n" if kw.get("xid") else "")
    (obs / "server.log").write_text("INFO session start\n")
    (obs / "memwatch.log").write_text("ARMED target_pid=1\n")
    (ev / "summary.json").write_text(json.dumps({"cells": {"accuracy": {
        "status": kw.get("kitstatus", "OK"), "suites_not_run_after_failure": []}}}))
    entries = []
    for (suite, (split, n)), correct in zip(SUITES.items(), scores):
        if kw.get("missing") == suite:
            continue
        tdir = ev / "accuracy" / "transcripts" / suite
        tdir.mkdir(parents=True)
        rows = n - 1 if kw.get("short") == suite else n
        for i in range(rows):
            reason = "correct" if i < correct else "wrong"
            if kw.get("transport") == suite and i == n - 1:
                reason = "invalid: URLError: <urlopen error [Errno 111] Connection refused>"
            if kw.get("timeout") == suite and i == n - 1:
                reason = "invalid: TimeoutError: timed out"
            rendering = RENDERING[suite]
            if kw.get("effort") == suite and i == 0:
                rendering = rendering.replace(f"-effort-{EFFORT}", "-effort-medium")
            if kw.get("wrapped") == suite and i == 0:
                rendering = f"official-encoder-{ENCODER}-thinking-effort-{EFFORT}"
            (tdir / f"{i:05d}.json").write_text(json.dumps({"rendering": rendering, "reason": reason}))
        (ev / "accuracy" / f"acc-{suite}.json").write_text(json.dumps({
            "suite": suite, "split": "dev" if kw.get("split") == suite else split, "n": n, "model": MODEL,
            "stack_label": STACK, "ledger_namespace": "", "correct": correct, "invalid_count": 0,
            "rowset_sha256": "0" * 64 if kw.get("rowset") == suite else ROWSETS[suite],
            "generation": {"encoder": "inkling2", "max_tokens": 16384}}))
        if suite in LEDGERED:
            entries.append({"stack_label": STACK, "suite": suite, "ledger_namespace": "", "phase": "started"})
            if kw.get("ledger") != suite:
                entries.append({"stack_label": STACK, "suite": suite, "ledger_namespace": "", "phase": "completed",
                                "n": n, "correct": correct})
    ledger = root / "ledger.json"
    ledger.write_text(json.dumps(entries))
    (c2 / "vision").mkdir(parents=True)
    (c2 / "toolcall").mkdir()
    (c2 / "gate-summary.json").write_text(json.dumps({"verdict": kw.get("m3", "PASS")}))
    prefill, d0, d28 = speed
    (c2 / "summary.json").write_text(json.dumps({"cells": {"speed": {"result": {"levels": {
        "0": {"median_decode_tok_s": d0, "valid": True},
        "28672": {"median_decode_tok_s": d28, "median_prefill_tok_s": prefill,
                  "valid": not kw.get("speedinvalid")}}}}}}))
    (c2 / "vision" / "summary.json").write_text(json.dumps({"n": 100, "correct": mmmu}))
    (c2 / "toolcall" / "toolcall.json").write_text(json.dumps({"passed": tool, "total": 20}))
    return ev, c2, ledger


CASES = [
    ("faster and more accurate: PROMOTE", {}, "PROMOTE"),
    ("sum equal to the default's 454: KEEP_DEFAULT", {"scores": (98, 218, 138)}, "KEEP_DEFAULT"),
    ("sum 455 with every suite near: PROMOTE", {"scores": (98, 219, 138)}, "PROMOTE"),
    ("higher sum but MMLU-Pro significantly worse: KEEP_DEFAULT", {"scores": (100, 205, 164)}, "KEEP_DEFAULT"),
    ("decode @28K 40.91 misses 40.92: KEEP_DEFAULT", {"speed": (3000.0, 36.0, 40.91)}, "KEEP_DEFAULT"),
    ("prefill 2,711.4 misses: KEEP_DEFAULT", {"speed": (2711.4, 36.0, 42.0)}, "KEEP_DEFAULT"),
    ("decode @0 34.06 misses: KEEP_DEFAULT", {"speed": (3000.0, 34.06, 42.0)}, "KEEP_DEFAULT"),
    ("MMMU 63: KEEP_DEFAULT", {"mmmu": 63}, "KEEP_DEFAULT"),
    ("tool-call 18/20: KEEP_DEFAULT", {"tool": 18}, "KEEP_DEFAULT"),
    ("a suite file missing: NO_RESULT", {"missing": "humaneval"}, "NO_RESULT"),
    ("a dev split: NO_RESULT", {"split": "mmlu-pro"}, "NO_RESULT"),
    ("a transcript short: NO_RESULT", {"short": "gsm8k"}, "NO_RESULT"),
    ("a connection-refused row: NO_RESULT", {"transport": "mmlu-pro"}, "NO_RESULT"),
    ("a timed-out row is scored, not refused", {"timeout": "mmlu-pro"}, "PROMOTE"),
    ("a row rendered at another effort: NO_RESULT", {"effort": "mmlu-pro"}, "NO_RESULT"),
    ("a HumanEval row sent through the chat wrapper: NO_RESULT", {"wrapped": "humaneval"}, "NO_RESULT"),
    ("a kernel Xid: NO_RESULT", {"xid": True}, "NO_RESULT"),
    ("FAILED_ATTEMPT: NO_RESULT", {"attempt": True}, "NO_RESULT"),
    ("no draw line: NO_RESULT", {"nodraw": True}, "NO_RESULT"),
    ("ledger without completed: NO_RESULT", {"ledger": "gsm8k"}, "NO_RESULT"),
    ("kit accuracy cell failed: NO_RESULT", {"kitstatus": "FAILED"}, "NO_RESULT"),
    ("M3 not PASS: NO_RESULT", {"m3": "FAIL"}, "NO_RESULT"),
    ("M3 speed level invalid: NO_RESULT", {"speedinvalid": True}, "NO_RESULT"),
    ("other questions than the default's: NO_RESULT", {"rowset": "gsm8k"}, "NO_RESULT"),
    ("a dirty head: NO_RESULT", {"dirty": 1}, "NO_RESULT"),
    ("code differs from a freeze: NO_RESULT", {"codediff": True}, "NO_RESULT"),
]


def real_evidence() -> bool:
    """Synthetic bundles share this file's assumptions about field names, so they cannot catch schema
    drift (M3 candidate 1's defect). Run the readers on real bundles: the default's own holdout draw
    must read back as 98 / 218 / 138 with every validity check passing under its own declared run,
    and candidate 2's committed M3 bundle must give its kit speed, MMMU and tool-call values."""
    default_ev = CODE / DEFAULT_DRAW
    qwen = {"model": "qwen3.8-flash-next", "stack": "qwen3.8-flash-next-cuda-spark-128g-1m", "encoder": "qwen38",
            "rendering": {"gsm8k": "official-encoder-qwen38-thinking-effort-low",
                          "mmlu-pro": "official-encoder-qwen38-thinking-effort-low", "humaneval": "raw"},
            "observe": "observe/holdout",
            "lines": ("probe identical to candidate 5; drawing the holdout", "holdout: scope empty after stop"),
            # That driver stopped the engine before disarming memwatch, so memwatch logged this at teardown
            # (12:06:47Z, the scope empty at 12:06:48Z). This draw's driver disarms first, as candidate 2's
            # did (its memwatch.log ends DISARMED), so DECLARED tolerates nothing.
            "tolerate": ("FAIL_CLOSED signal=TERM armed_target_unreadable",)}
    try:
        rows = draw(default_ev, LEDGER, identity=lambda driver: None, declared=qwen)
        got = {s: r["correct"] for s, r in rows.items()}
        ok_default = got == DEFAULT and all(r["z"] == 0.0 for r in rows.values())
    except Bad as error:
        got, ok_default = str(error), False
    print(f"{'ok ' if ok_default else 'BAD'}  real-default  the readers on the default's own draw: {got}")
    try:
        measured = m3(C2)
        gate = load(C2 / "gate-summary.json")["checks"]
        ok_m3 = measured["mmmu"] == gate["vision"]["correct"] and measured["toolcall"] == gate["toolcall"]["passed"] \
            and measured["speed"]["decode_0"] == gate["speed"]["0"]["median_decode"] \
            and measured["speed"]["decode_28672"] == gate["speed"]["28672"]["median_decode"] \
            and finite(measured["speed"]["prefill_28672"])
    except (Bad, KeyError, TypeError) as error:
        measured, ok_m3 = str(error), False
    print(f"{'ok ' if ok_m3 else 'BAD'}  real-m3       m3() on candidate 2's committed bundle: {measured}")
    return ok_default and ok_m3


def driver_literals() -> bool:
    """The lines draw() requires are the ones run-holdout.sh logs, verbatim."""
    driver = HERE / "run-holdout.sh"
    if not driver.is_file():
        print("BAD  driver        run-holdout.sh missing")
        return False
    text = driver.read_text()
    need = (f"EFFORT={EFFORT}\n", DRAW_LINE.replace(f"effort {EFFORT}", "effort $EFFORT"), JOURNAL_ALIVE,
            "scope empty after stop", 'log "head $(git -C "$WT" rev-parse HEAD) dirty=', "FAILED_ATTEMPT",
            '--reasoning-effort "$EFFORT"', "--accuracy-split holdout", 'OBS=$EV/observe')
    absent = [n for n in need if n not in text]
    print(f"{'ok ' if not absent else 'BAD'}  driver        run-holdout.sh logs the lines draw() requires"
          + (f"; absent {absent}" if absent else ""))
    return not absent


def self_test() -> bool:
    ok = real_evidence() & driver_literals()
    for name, kw, want in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            def identity(driver, kw=kw):
                heads = HEAD.findall(driver)
                if len(heads) != 1 or heads[0][1] != "0" or kw.get("codediff"):
                    raise Bad("code identity")
            got = evaluate(*_bundle(Path(tmp), **kw), identity=identity)
        match = got["verdict"] == want
        ok &= match
        print(f"{'ok ' if match else 'BAD'}  {name}" + ("" if match else f"  got {got['verdict']} {got.get('why', '')}"))
    print(f"{len(CASES)} synthetic cases + 2 real-evidence checks + driver literals, "
          f"{'all match' if ok else 'MISMATCH'}")
    return ok


def main(argv) -> int:
    if not self_test():
        return 2
    if argv[:1] == ["--self-test"]:
        return 0
    if argv[:1] == ["--preflight"]:
        freeze = git("log", "-1", "--format=%H", "--", f"{HOLDOUT}/README.md")
        frozen, m3_freeze = m3_frozen()
        ok = unchanged(freeze, None, HOLDOUT_FROZEN) and unchanged(m3_freeze, None, frozen)
        print(json.dumps({"freeze": freeze, "m3_freeze": m3_freeze, "unchanged": ok}))
        return 0 if ok else 1
    out = evaluate(HERE, C2, LEDGER)
    (HERE / "holdout-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out.get(k) for k in ("verdict", "why", "faster", "more_accurate")}))
    return 0 if out["verdict"] != "NO_RESULT" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
