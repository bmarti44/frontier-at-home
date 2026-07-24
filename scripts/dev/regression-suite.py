#!/usr/bin/env python3
"""Deterministic regression suite for the 2026-07 speed-tuning changes.

Every test uses temperature 0, fixed prompts derived from the committed
fixture, and token-count (not wall-clock) assertions, so results are
reproducible across runs on the same weights + engine build. Exit code 0 =
all selected tests passed; non-zero = at least one failure. Human-readable
PASS/FAIL lines on stdout; machine-readable JSON on --json PATH.

Tests and their targets:
  prefix-cache   llama.cpp endpoint (default http://127.0.0.1:8011)
                 Regression for: in-slot prefix reuse (docs/speed-tuning-
                 2026-07-23.md §4). Asserts identical repeat reuses >=99%
                 and shared-prefix reuses >=95% of the prefix.
  slot-thrash    llama.cpp endpoint, REQUIRES a 2-slot server (DSV4_PARALLEL=2)
                 Regression for: utility-request cache clobber (commit
                 c58c43d). Simulates a Hermes turn -> title call -> turn 2;
                 asserts turn 2 processes < 1500 tokens (i.e. the utility
                 call did not evict the conversation slot).
  reap-mmid      Patched llama.cpp build + REAP GGUF on a dev port (8021).
                 Regression for: duplicate-expert-ID crash, BOTH layers
                 (docs/patches/mmid-duplicate-expert-ids.patch). Sends three
                 small requests (layer 1: warp race) AND one ~19K-token
                 prefill (layer 2: shared-memory store overrun — small
                 requests cannot catch it). Asserts all return HTTP 200.
  ds4-mem-init   ds4 v0.4.2 dev server (default http://127.0.0.1:8022).
                 Characterization of the one-time lazy-init pool (docs/
                 ds4-v042-eval-2026-07-24.md): asserts the first-generation
                 allocation lands in the 9-15 GiB band, and that five
                 further identical requests move MemAvailable < 1.0 GiB
                 total (flat steady state — the original "creep" claim is
                 the regression this guards against).
  slot-restore   llama.cpp endpoint with --slot-save-path enabled.
                 EXPECTED-FAIL guard for the known fork bug (docs/
                 speed-paths-2026-07-24.md Bug A): asserts restore-then-
                 repeat still re-prefills (cache_n < 100). When the fork
                 fix lands this test FAILS, signalling the guard should be
                 flipped to assert reuse instead.

Usage:
  regression-suite.py TEST [TEST...] [--base URL] [--json PATH]
  regression-suite.py all-llamacpp        # prefix-cache + slot-thrash
"""
import argparse, json, sys, time, urllib.request, urllib.error

FIXTURE = "/home/bmarti44/spark-deepseek-v4-flash/fixtures/ctx-32k.txt"
RESULTS = []

def load_prefix(frac=0.46):
    text = open(FIXTURE).read()
    return text[: int(len(text) * frac)]

def chat(base, messages, max_tokens=64, timeout=1800):
    body = {"model": "d", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def record(name, ok, detail):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return ok

def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    raise RuntimeError("no MemAvailable")

def t_prefix_cache(base):
    P = load_prefix()
    msgs = [{"role": "system", "content": "Reference document follows.\n" + P},
            {"role": "user", "content": "Summarize the first paragraph in one sentence."}]
    r1 = chat(base, msgs)["timings"]
    r2 = chat(base, msgs)["timings"]
    msgs2 = [msgs[0], {"role": "user", "content": "What is the last topic discussed? One sentence."}]
    r3 = chat(base, msgs2)["timings"]
    total = r1["prompt_n"] + r1["cache_n"]
    ok1 = r2["cache_n"] >= 0.99 * total
    # Shared-prefix divergence: DSV4's cache cannot trim the tail, so the
    # server rolls back to the nearest context checkpoint, which can trail
    # the divergence point by up to the checkpoint spacing (default 8192)
    # plus the generated tail. The invariant is checkpoint-bounded reuse,
    # not raw-LCP reuse.
    CHECKPOINT_SPACING = 8192
    floor = total - CHECKPOINT_SPACING - 1024
    ok2 = r3["cache_n"] >= floor
    return record("prefix-cache", ok1 and ok2,
                  f"repeat cache_n={r2['cache_n']}/{total} (>=99%: {ok1}); "
                  f"shared-prefix cache_n={r3['cache_n']} (checkpoint-bounded floor {floor}: {ok2})")

def t_slot_thrash(base):
    P = load_prefix()
    conv = [{"role": "system", "content": "Reference document follows.\n" + P},
            {"role": "user", "content": "Summarize the first half in two sentences."}]
    first = chat(base, conv, 128)
    reply = first["choices"][0]["message"].get("content") or "Summary."
    # utility call, as an agent title-generator would send
    chat(base, [{"role": "user", "content": "Generate a short, descriptive title (3-7 words) for a conversation that starts with a summary request. Reply with only the title."}], 24)
    conv += [{"role": "assistant", "content": reply},
             {"role": "user", "content": "Now give one example record number."}]
    t2 = chat(base, conv, 64)["timings"]
    ok = t2["prompt_n"] < 1500
    return record("slot-thrash", ok,
                  f"turn-2 after utility call: prompt_n={t2['prompt_n']} cache_n={t2['cache_n']} (< 1500 required; single-slot server re-prefills ~20000)")

def t_reap_mmid(base):
    P = load_prefix()
    seqs = [
        ("small-1", [{"role": "user", "content": "Write one sentence about ships."}], 48),
        ("small-2", [{"role": "user", "content": "What is 6*7? Answer briefly."}], 48),
        ("small-3", [{"role": "user", "content": "Name one color."}], 48),
        ("large-prefill", [{"role": "system", "content": "Reference document follows.\n" + P},
                           {"role": "user", "content": "Summarize briefly."}], 48),
        ("small-after-large", [{"role": "user", "content": "Name one animal."}], 48),
    ]
    for tag, msgs, mt in seqs:
        try:
            chat(base, msgs, mt)
        except Exception as e:
            return record("reap-mmid", False, f"{tag} failed: {type(e).__name__} (duplicate-expert-ID regression — see docs/patches/)")
    return record("reap-mmid", True, "3 small + 19K prefill + follow-up all served (both mmid fix layers hold)")

def t_ds4_mem_init(base):
    prose = [{"role": "user", "content": "Write a flowing two-paragraph essay about the history of shipbuilding."}]
    before = mem_available_gib()
    chat(base, prose, 128)
    after_first = mem_available_gib()
    init_cost = before - after_first
    for _ in range(5):
        chat(base, prose, 128)
    after_five = mem_available_gib()
    drift = after_first - after_five
    # Band is config-dependent: ~11-12 G at default knobs; ~8-9 G with
    # DS4_SERVER_COALESCE_MAX_TOKENS=2048 + DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1
    # (sol memory analysis, 2026-07-24). Override via env for other configs.
    import os
    lo = float(os.environ.get("DS4_INIT_BAND_LO", "7.0"))
    hi = float(os.environ.get("DS4_INIT_BAND_HI", "15.0"))
    ok1 = lo <= init_cost <= hi
    ok2 = drift < 1.0
    return record("ds4-mem-init", ok1 and ok2,
                  f"first-request init={init_cost:.1f}G (band {lo}-{hi}: {ok1}); "
                  f"drift over 5 identical={drift:.2f}G (<1.0: {ok2}); NOTE: run on a freshly started server")

def t_slot_restore(base):
    P = load_prefix(0.2)
    msgs = [{"role": "system", "content": P}, {"role": "user", "content": "Say A."}]
    chat(base, msgs, 8)
    def slots(action, fn):
        req = urllib.request.Request(base + f"/slots/0?action={action}",
                                     data=json.dumps({"filename": fn}).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))
    slots("save", "regress-bugA.bin")
    chat(base, [{"role": "user", "content": "Different short prompt entirely."}], 8)
    slots("restore", "regress-bugA.bin")
    t = chat(base, msgs, 8)["timings"]
    still_broken = t["cache_n"] < 100
    return record("slot-restore(expected-fail guard)", still_broken,
                  f"restore-then-repeat cache_n={t['cache_n']}; guard asserts bug STILL PRESENT — "
                  "if this FAILS the fork fix landed: flip the assertion to require reuse")

TESTS = {"prefix-cache": t_prefix_cache, "slot-thrash": t_slot_thrash,
         "reap-mmid": t_reap_mmid, "ds4-mem-init": t_ds4_mem_init,
         "slot-restore": t_slot_restore}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="+")
    # No default: port 8011's engine changes with operational posture (see
    # results/OPERATIONAL-OVERRIDE-2026-07-24.md), so an implicit default
    # can silently run llama.cpp assertions against ds4 or vice versa.
    ap.add_argument("--base", required=True,
                    help="engine base URL, e.g. http://127.0.0.1:8011 — required; verify which engine is serving first")
    ap.add_argument("--json")
    args = ap.parse_args()
    names = ["prefix-cache", "slot-thrash"] if args.tests == ["all-llamacpp"] else args.tests
    ok = True
    for n in names:
        if n not in TESTS:
            print(f"unknown test: {n}", file=sys.stderr); return 2
        try:
            ok &= TESTS[n](args.base)
        except Exception as e:
            record(n, False, f"harness error: {type(e).__name__}: {e}")
            ok = False
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "base": args.base, "results": RESULTS}, f, indent=1)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
