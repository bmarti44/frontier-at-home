#!/usr/bin/env python3
"""T0 for task 49: the strict-FP8 KV cache MUST fit alongside the model and the
expert cache. Written before the engine change, and run before every attempt.

The user's constraint, made executable: "go strict, so long as it all still fits
in memory with the model."

Every input is DERIVED FROM MEASURED ENGINE OUTPUT, never hardcoded:
  - per-token and fixed KV cost come from a two-point fit of the engine's own
    "context buffers ... MiB (ctx=N)" lines
  - the expert-cache footprint comes from the safe-run sampler's peak RSS
  - the headroom comes from the sampler's minimum MemAvailable
If an input cannot be found, the test FAILS rather than assuming a default.

Usage: test_kv_budget.py [--ctx N] [--margin-gib G]
"""
import argparse, re, subprocess, sys

GIB = 1024 ** 3
KIB = 1024

# strict layout: 512 non-RoPE @1B + 64 RoPE @4B + 8 block scales @4B = 800 B/row
STRICT_ROW_BYTES = 512 * 1 + 64 * 4 + 8 * 4
F32_ROW_BYTES = 576 * 4 + 8 * 4          # what the cache costs today, + scales
N_LAYER = 79

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout
    except Exception:
        return ""

def measured_context_points():
    """(ctx, MiB) from the engine's own reporting, across every run log."""
    out = sh("sudo -n -u dsv4 bash -c \"grep -h 'context buffers' "
             "/home/dsv4/ds4-project/glm52-*/server-*.log 2>/dev/null\"")
    pts = {}
    for m in re.finditer(r"context buffers ([0-9.]+) MiB \(ctx=(\d+)", out):
        pts[int(m.group(2))] = float(m.group(1))
    return sorted(pts.items())

def _sampler_text():
    """Only runs of the SAME configuration we are predicting for.

    Pooling every crashlog was a TEST BUG, not a conservative choice: the worst
    case it found (peak RSS 93.4 GiB / min free 5.6 GiB) came from
    'reap-bench', a llama.cpp --cpu-moe run on the REAP model -- a different
    engine, a different model, and a different memory profile entirely -- and
    from 'railtest-ulimit', a deliberate memory stress test. Budgeting the ds4
    streaming config against those numbers is comparing to the wrong system.

    main.log records the safe-run WRAPPER command (e.g. "bash /tmp/slru_ab_v1.sh"),
    not the engine argv, so the qualifying signal is the recorded binary sha /
    tree line plus an explicit exclusion of the two known-incomparable runs:
      reap-bench     -- llama.cpp --cpu-moe on the REAP model (different engine,
                        different model; those shards have since been deleted)
      railtest-*     -- a deliberate memory stress test, not a serving config
    If the filter leaves nothing, the test FAILS rather than silently widening."""
    EXCLUDE = ("reap-bench", "railtest")
    dirs = sh("sudo -n -u dsv4 bash -c \"ls -d "
              "/home/dsv4/ds4-project/glm52-crashlog/*/ 2>/dev/null\"").split()
    keep = []
    for d in dirs:
        if any(x in d for x in EXCLUDE):
            continue
        # a real ds4 serving run records the engine tree/binary sha
        if "binary_sha12" in sh("sudo -n -u dsv4 cat %smain.log" % d):
            keep.append(d + "main.log")
    if not keep:
        return ""
    print("budget baseline: %d comparable ds4-streaming runs" % len(keep))
    return sh("sudo -n -u dsv4 bash -c \"cat %s\""
              % " ".join(d.replace("main.log", "samples.log") for d in keep))

def measured_peak_rss_gib(txt):
    peak = 0
    for m in re.finditer(r"rss_kb=(\d+)", txt):
        peak = max(peak, int(m.group(1)))
    return peak / 1048576.0

def measured_min_avail_gib(txt):
    lo = None
    for m in re.finditer(r"mem_avail_kb=(\d+)", txt):
        v = int(m.group(1))
        if v > 0 and (lo is None or v < lo):
            lo = v
    return (lo / 1048576.0) if lo else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--margin-gib", type=float, default=6.0)
    a = ap.parse_args()

    fails = []
    print("T0: strict-FP8 KV must fit alongside the model and expert cache\n")

    pts = measured_context_points()
    if len(pts) < 2:
        print("FAIL: need >=2 engine-reported 'context buffers' points, found %d" % len(pts))
        return 1
    (c1, m1), (c2, m2) = pts[0], pts[-1]
    per_f32_kib = (m2 - m1) * 1024.0 / (c2 - c1)
    fixed_mib = m1 - per_f32_kib * c1 / 1024.0
    print("measured points: %s" % [(c, round(m, 1)) for c, m in pts])
    print("  -> F32 %.1f KiB/token + %.0f MiB fixed" % (per_f32_kib, fixed_mib))

    # only the compact KV row shrinks; indexer/misc per-token cost is unchanged
    compact_f32_kib = F32_ROW_BYTES * N_LAYER / 1024.0
    other_kib = per_f32_kib - compact_f32_kib
    if other_kib < 0:
        fails.append("fit implies negative non-compact per-token cost (%.1f KiB) -- "
                     "the row model does not match the engine" % other_kib)
        other_kib = 0.0
    per_strict_kib = STRICT_ROW_BYTES * N_LAYER / 1024.0 + other_kib
    print("  -> compact %.1f KiB/tok, other %.1f KiB/tok (unchanged by this work)"
          % (compact_f32_kib, other_kib))
    print("  -> STRICT %.1f KiB/token  (%.2fx less)\n"
          % (per_strict_kib, per_f32_kib / per_strict_kib))

    txt = _sampler_text()
    rss = measured_peak_rss_gib(txt)
    avail = measured_min_avail_gib(txt)
    if rss <= 0:
        fails.append("no peak RSS found in COMPARABLE ds4-streaming runs -- refusing "
                     "to fall back to unrelated configurations")
    if avail is None:
        fails.append("no MemAvailable found in comparable ds4-streaming runs")
    print("measured peak RSS        : %.1f GiB  (expert cache + runtime)" % rss)
    print("measured min MemAvailable: %s GiB" % ("%.1f" % avail if avail else "?"))

    kv_now = (fixed_mib * 1024 * 1024 + per_f32_kib * KIB * c2) / GIB
    envelope = kv_now + (avail or 0)
    kv_strict = (fixed_mib * 1024 * 1024 + per_strict_kib * KIB * a.ctx) / GIB
    print("\nKV envelope = current KV at ctx=%d (%.1f GiB) + free (%.1f GiB) = %.1f GiB"
          % (c2, kv_now, avail or 0, envelope))
    print("strict KV at ctx=%d = %.1f GiB" % (a.ctx, kv_strict))

    need = kv_strict + a.margin_gib
    ok = need <= envelope
    print("\nrequire strict_KV + %.1f GiB margin <= envelope:  %.1f <= %.1f  -> %s"
          % (a.margin_gib, need, envelope, "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("ctx=%d does not fit with a %.1f GiB margin; max safe ctx is ~%d"
                     % (a.ctx, a.margin_gib,
                        int(((envelope - a.margin_gib) * GIB - fixed_mib * 1024 * 1024)
                            / (per_strict_kib * KIB))))

    # the same context under F32 must NOT fit -- otherwise this work buys nothing
    kv_f32_at_target = (fixed_mib * 1024 * 1024 + per_f32_kib * KIB * a.ctx) / GIB
    if kv_f32_at_target + a.margin_gib <= envelope:
        print("\nNOTE: ctx=%d already fits at F32 (%.1f GiB) -- the FP8 work is not "
              "required for THIS context size." % (a.ctx, kv_f32_at_target))
    else:
        print("\nctx=%d needs %.1f GiB at F32, which does NOT fit -- so strict FP8 is "
              "what unlocks it." % (a.ctx, kv_f32_at_target))

    for f in fails:
        print("FAIL: %s" % f)
    print("\n%s" % ("T0 PASSED" if not fails else "T0 FAILED (%d)" % len(fails)))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
