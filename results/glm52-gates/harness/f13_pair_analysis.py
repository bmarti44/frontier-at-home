#!/usr/bin/env python3
"""F13 logit comparison, corrected pairing.

The regime harness demanded a COLD dump with start==0 at the append length.
That can never exist: the engine always chunks a 5000+ token prompt (the GLM
indexed-prefill path is clamped to the 2048 indexer-top-k boundary), so even a
fresh process with a wiped KV dir evaluates 5066 tokens as 5064 then +2.

The valid pair is each arm's FINAL-position dump at the append length:
  X: start=5044 prompt=5066 suffix=22  (resumed from a disk-loaded checkpoint
                                        after a BPE re-merge junction miss)
  Y: start=5064 prompt=5066 suffix=2   (fresh process, wiped KV, minimal resume)
Both evaluate the same 5066-token prompt to the same final position, so their
last-token logits are directly comparable. They differ only in how much state
was restored rather than recomputed -- which is exactly the variable under test.

Reads dumps already on disk; no GPU needed.
"""
import os, re, struct, sys, hashlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/dsv4/ds4-project/glm52-f13-regime"

def parse(fn):
    m = re.match(r'lg-([XY])\.(\d+)\.s(-?\d+)_p(-?\d+)_x(-?\d+)$', fn)
    if not m:
        return None
    return {"arm": m.group(1), "seq": int(m.group(2)), "start": int(m.group(3)),
            "prompt": int(m.group(4)), "suffix": int(m.group(5)), "file": fn}

dumps = sorted((d for d in (parse(f) for f in os.listdir(OUT)) if d),
               key=lambda d: (d["arm"], d["seq"]))
if not dumps:
    print("no dumps found in %s" % OUT)
    raise SystemExit(1)

ap_len = max(d["prompt"] for d in dumps)
print("append length = %d" % ap_len)
for d in dumps:
    print("  %-28s arm=%s start=%-5d prompt=%-5d suffix=%d" %
          (d["file"], d["arm"], d["start"], d["prompt"], d["suffix"]))

# final-position dump at the append length, per arm; take the LAST such dump
def final_dump(arm):
    c = [d for d in dumps if d["arm"] == arm and d["prompt"] == ap_len]
    return c[-1] if c else None

X, Y = final_dump("X"), final_dump("Y")
if not X or not Y:
    print("\nNO PAIR: X=%s Y=%s" % (X and X["file"], Y and Y["file"]))
    raise SystemExit(0)

def load(fn):
    b = open(os.path.join(OUT, fn), 'rb').read()
    n = len(b) // 4
    return struct.unpack('<%df' % n, b[:n * 4])

r, c = load(X["file"]), load(Y["file"])
n = min(len(r), len(c))
if n == 0:
    print("\nempty dumps")
    raise SystemExit(0)

deltas = [abs(r[i] - c[i]) for i in range(n)]
mx = max(deltas)
mean = sum(deltas) / n
ir = max(range(n), key=lambda i: r[i])
ic = max(range(n), key=lambda i: c[i])
sr = sorted(r, reverse=True)[:2]
sc = sorted(c, reverse=True)[:2]

print("\nPAIR (both final-position at prompt=%d):" % ap_len)
print("  resumed : %s  (start=%d suffix=%d)" % (X["file"], X["start"], X["suffix"]))
print("  cold    : %s  (start=%d suffix=%d)" % (Y["file"], Y["start"], Y["suffix"]))
print("\nlogits=%d  max|delta|=%.6g  mean|delta|=%.6g" % (n, mx, mean))
print("argmax resumed=%d top1=%.5f margin=%.5f" % (ir, sr[0], sr[0] - sr[1]))
print("argmax cold   =%d top1=%.5f margin=%.5f" % (ic, sc[0], sc[0] - sc[1]))
print("same argmax: %s" % (ir == ic))

# how far apart are they in units of the model's own logit scale?
spread = max(max(r), max(c)) - min(min(r), min(c))
print("logit range=%.4f, so max|delta| is %.2f%% of full scale" %
      (spread, 100.0 * mx / spread if spread else 0.0))

print()
if mx == 0:
    print("VERDICT: BIT-IDENTICAL. The resumed evaluation is not the divergence source.")
elif mx < 1e-3:
    print("VERDICT: NUMERICS. Deltas are at chunk-shape/FP-reassociation scale.")
    print("Greedy decoding can still flip at a near-tie; check the argmax margin above.")
elif ir != ic:
    print("VERDICT: DIVERGENT STATE. The resumed path selects a DIFFERENT top token")
    print("with large logit deltas -- consistent with restored state that does not")
    print("match what a fresh evaluation of the same prompt produces.")
else:
    print("VERDICT: LARGE DELTAS, same argmax. State differs materially even though")
    print("this position happens to agree; divergence would surface downstream.")
