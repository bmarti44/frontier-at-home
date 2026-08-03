import re, sys
from collections import defaultdict, Counter, OrderedDict

path = sys.argv[1]
# decode steps: N8 lines (one per layer per token). batch prefill: N>8.
decode = defaultdict(list)   # layer -> [set(ids) per step in order]
batch  = defaultdict(list)
pat = re.compile(r"^XTRACE L(\d+) N(\d+): (.*)$")
for line in open(path, errors="replace"):
    m = pat.match(line)
    if not m: continue
    l, n, rest = int(m.group(1)), int(m.group(2)), m.group(3)
    ids = [int(x) for x in rest.replace(" ...", "").split()]
    (decode if n == 8 else batch)[l].append((n, ids))

layers = sorted(decode)
steps = min(len(decode[l]) for l in layers) if layers else 0
print(f"decode steps per layer: {steps}, layers: {len(layers)}, batch loads: {sum(len(v) for v in batch.values())}")

# 1. consecutive-token overlap per layer
ov = []
for l in layers:
    seq = [set(ids) for _, ids in decode[l]]
    o = [len(a & b) / 8 for a, b in zip(seq, seq[1:])]
    ov.append(sum(o) / len(o))
print(f"consecutive-token overlap: mean {sum(ov)/len(ov):.3f}, min {min(ov):.3f}, max {max(ov):.3f}")

# Future-token expert-set expansion. This is the exact oracle target for the
# union probe, not a predictor result: at token t, union selections from
# t+1..t+K within the same routed layer.
for horizon in (2, 4, 8):
    union_sizes = []
    for layer in layers:
        sequence = [set(ids) for _, ids in decode[layer]]
        for token in range(len(sequence) - horizon):
            future = set()
            for offset in range(1, horizon + 1):
                future.update(sequence[token + offset])
            union_sizes.append(len(future))
    if not union_sizes:
        print(f"future-union K={horizon} samples=0 status=NO_RESULT")
        continue
    ordered = sorted(union_sizes)
    p95_index = (95 * len(ordered) + 99) // 100 - 1
    print(
        f"future-union K={horizon} samples={len(ordered)} "
        f"mean={sum(ordered) / len(ordered):.3f} "
        f"p95={ordered[p95_index]}"
    )

# Flat-router null hypothesis. Fit each layer's popularity prior on the first
# 70% of its trace and evaluate only future unions wholly in the final 30%.
# This old trace lacks request boundaries, so these are bootstrap diagnostics;
# the P0 corpus must use fixture-grouped frozen splits.
for horizon in (2, 4, 8):
    for budget in (8, 16, 32, 48, 64):
        hit_experts = 0
        target_experts = 0
        contained_sets = 0
        samples = 0
        for layer in layers:
            sequence = [set(ids) for _, ids in decode[layer]]
            split = max(1, len(sequence) * 7 // 10)
            frequency = Counter(
                expert
                for selected in sequence[:split]
                for expert in selected
            )
            ordered_experts = sorted(
                frequency,
                key=lambda expert: (-frequency[expert], expert),
            )
            predicted = set(ordered_experts[:budget])
            for token in range(split, len(sequence) - horizon):
                future = set()
                for offset in range(1, horizon + 1):
                    future.update(sequence[token + offset])
                hit_experts += len(predicted & future)
                target_experts += len(future)
                contained_sets += int(future <= predicted)
                samples += 1
        if samples == 0 or target_experts == 0:
            print(
                f"frequency-prior K={horizon} budget={budget} "
                "samples=0 status=NO_RESULT"
            )
            continue
        print(
            f"frequency-prior K={horizon} budget={budget} samples={samples} "
            f"recall={hit_experts / target_experts:.6f} "
            f"set_coverage={contained_sets / samples:.6f}"
        )

# 2. popularity skew (decode selections)
freq = Counter()
total = 0
for l in layers:
    for _, ids in decode[l]:
        for e in ids: freq[(l, e)] += 1; total += 1
f = sorted(freq.values(), reverse=True)
cum = 0
for pct in (1, 5, 10, 20, 50):
    k = max(1, len(f) * pct // 100)
    print(f"top {pct}% of (layer,expert) pairs serve {sum(f[:k])/total*100:.1f}% of selections", end="; " if pct != 50 else "\n")
print(f"distinct (layer,expert) pairs touched: {len(freq)} of {len(layers)*256}")

# 3. LRU cache simulation over the full decode stream (global, expert=9.28MiB)
EXP_MB = 9.28
print("legacy LRU I/O ceiling below assumes 10.7 GB/s; recalibrate after fio")
stream = []  # interleave layers in temporal order: reconstruct per step
for i in range(steps):
    for l in layers:
        stream.extend(((l, e) for e in decode[l][i][1]))
for gib in (10, 20, 30, 40, 60, 80, 100):
    cap = int(gib * 1024 / EXP_MB)
    lru = OrderedDict()
    hits = miss = 0
    for key in stream:
        if key in lru:
            hits += 1; lru.move_to_end(key)
        else:
            miss += 1; lru[key] = 1
            if len(lru) > cap: lru.popitem(last=False)
    hr = hits / (hits + miss) * 100
    tok_s = min(10.7 / (miss/(hits+miss) * 75 * 8 * EXP_MB/1024), 999)
    print(f"LRU {gib:3d} GiB ({cap:5d} slots): hit {hr:5.1f}%  -> I/O-ceiling ~{tok_s:.1f} tok/s")
