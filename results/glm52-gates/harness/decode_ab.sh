#!/bin/bash
# Decode-isolated A/B for the two decode levers, one binary, ABBA order.
#
# WHY THIS HARNESS EXISTS. The previous keep-N speed attempt measured wall
# time of a 5047-token-prompt request, which is ~95% PREFILL -- it could not
# have seen a decode-only lever. Here decode is isolated arithmetically:
# fire the same short prompt at max_tokens=1 (t1: prompt eval + 1 token) and
# at max_tokens=65 (t65), so decode t/s = 64 / (t65 - t1). Both requests hit
# the same warm server, so prefill cost cancels.
#
# Arms (same hashed binary; only env differs):
#   bug     DS4_CUDA_MODEL_GEN_ALWAYS_BUMP=1 -- reproduces the pre-fix
#           behaviour where every model-span install bumped the load
#           generation and wiped the whole persistent expert cache
#   fix     default: generation only bumps when the mapping really changes
#   keep7   fix + DS4_GLM_TOPK_KEEP=7 + SKIP_LOAD (7/8 unique experts fetched)
#   keep6   fix + DS4_GLM_TOPK_KEEP=6 + SKIP_LOAD (6/8 unique experts fetched)
#
# Run in ABBA order (bug fix keep7 keep6 | keep6 keep7 fix bug) so a
# monotone drift in machine state cannot masquerade as an arm effect --
# sol's "fixed arm order, one run per arm" objection.
#
# Deterministic verification recorded per arm, not inferred from timings:
#   - flush count ("persistent expert cache flushed")
#   - final expert-cache stats line (hit%, resident slots)
#   - histogram of unique experts per decode load (must equal keep-N)
#   - HTTP status of every request (sol: harness must not treat a failed
#     curl as a fast response) and output sha for fidelity tracking
set -u
OUT=/home/dsv4/ds4-project/glm52-decode-ab
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
PORT=8016
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
note "decode A/B start binary_sha12=$(sha256sum $SRC/ds4-server | cut -c1-12)"

# sol: sanitize inherited environment so no stray lever leaks into an arm
unset DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL DS4_GLM_TOPK_KEEP \
      DS4_GLM_TOPK_NORENORM DS4_GLM_TOPK_SKIP_LOAD \
      DS4_CUDA_MODEL_GEN_ALWAYS_BUMP DS4_GLM_MTP 2>/dev/null || true

wait_gone() {
  for i in $(seq 1 90); do pgrep -x ds4-server > /dev/null || return 0; sleep 2; done
  pkill -KILL -x ds4-server; sleep 5
}
pkill -TERM -f "llama-server.*8011" 2>/dev/null
pkill -TERM -x ds4-server 2>/dev/null; wait_gone

python3 - "$OUT" <<'PYEOF'
import json, sys
p = ("Explain, in careful technical detail, how a write-back cache decides "
     "which line to evict and why that policy matters for throughput.")
for n in (1, 65):
    json.dump({"model": "default", "prompt": p, "max_tokens": n,
               "temperature": 0}, open("%s/q%d.json" % (sys.argv[1], n), "w"))
PYEOF

# $1 tag  $2 pass(1|2)  $3.. env assignments
run_arm() {
  local tag=$1 pass=$2; shift 2
  local envs=("$@")
  local key="${tag}-p${pass}"
  note "arm $key env=[${envs[*]:-none}]"
  wait_gone
  env "${envs[@]}" \
    DS4_GLM_TP_DEBUG=1 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
    DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
    DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "$SRC/ds4-server" --cuda -m "$GGUF" -c 8192 --host 127.0.0.1 --port $PORT \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    > "$OUT/server-$key.log" 2>&1 &
  SPID=$!
  local up=0
  for i in $(seq 1 300); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && { up=1; break; }
    kill -0 $SPID 2>/dev/null || { note "$key died during load"; return 1; }
    sleep 2
  done
  [[ $up == 1 ]] || { note "$key never came up"; kill -TERM $SPID; return 1; }

  fire() { # $1 label, $2 request file -> records ms + http + sha
    local t0=$(date +%s%3N)
    local code=$(curl -s -o "$OUT/$key-$1.json" -w '%{http_code}' --max-time 3600 \
      -H 'Content-Type: application/json' -d @"$2" \
      http://127.0.0.1:$PORT/v1/completions)
    echo "$key $1 ms=$(( $(date +%s%3N) - t0 )) http=$code" >> "$OUT/timings"
  }
  fire warm "$OUT/q65.json"                  # prime cache + KV for this prompt
  echo "$key mark=$(wc -l < "$OUT/server-$key.log")" >> "$OUT/timings"
  fire t1_a  "$OUT/q1.json"
  fire t65_a "$OUT/q65.json"
  fire t1_b  "$OUT/q1.json"
  fire t65_b "$OUT/q65.json"
  {
    echo "$key flushes=$(grep -c 'expert cache flushed' "$OUT/server-$key.log")"
    echo "$key stats=$(grep 'expert-cache stats' "$OUT/server-$key.log" | tail -1)"
  } >> "$OUT/timings"
  kill -TERM $SPID; wait_gone
  return 0
}
trap 'kill -TERM ${SPID:-0} 2>/dev/null' EXIT
BUG=(DS4_CUDA_MODEL_GEN_ALWAYS_BUMP=1)
FIX=()
K7=(DS4_GLM_TOPK_KEEP=7 DS4_GLM_TOPK_SKIP_LOAD=1)
K6=(DS4_GLM_TOPK_KEEP=6 DS4_GLM_TOPK_SKIP_LOAD=1)
run_arm bug   1 "${BUG[@]}"
run_arm fix   1 DS4_CUDA_EXPERT_CACHE_SLRU=1
run_arm keep7 1 "${K7[@]}"
run_arm keep6 1 "${K6[@]}"
run_arm keep6 2 "${K6[@]}"
run_arm keep7 2 "${K7[@]}"
run_arm fix   2 DS4_CUDA_EXPERT_CACHE_SLRU=1
run_arm bug   2 "${BUG[@]}"
trap - EXIT

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import json, os, re, sys, hashlib, statistics
out = sys.argv[1]
arms = ["bug", "fix", "keep7", "keep6"]
ms, http, mark, flush, stats = {}, {}, {}, {}, {}
for line in open(os.path.join(out, "timings")):
    p = line.split()
    if len(p) >= 3 and p[2].startswith("ms="):
        ms[(p[0], p[1])] = int(p[2].split("=")[1])
        http[(p[0], p[1])] = p[3].split("=")[1] if len(p) > 3 else "?"
    elif len(p) >= 2 and p[1].startswith("mark="):
        mark[p[0]] = int(p[1].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("flushes="):
        flush[p[0]] = int(p[1].split("=")[1])
    elif len(p) >= 2 and p[1].startswith("stats="):
        stats[p[0]] = line.split("stats=", 1)[1].strip()
def text(key, lbl):
    try:
        raw = open(os.path.join(out, "%s-%s.json" % (key, lbl)), 'rb').read()
        d = json.loads(raw.decode('utf-8', 'replace'))
        return d["choices"][0]["text"], d["usage"]["completion_tokens"]
    except Exception:
        return "", 0
UNIQ = re.compile(r"slots=8 unique=(\d+)")
def uniq_hist(key):
    h = {}
    try:
        for i, line in enumerate(open(os.path.join(out, "server-%s.log" % key), errors="replace")):
            if i < mark.get(key, 0): continue
            m = UNIQ.search(line)
            if m: h[int(m.group(1))] = h.get(int(m.group(1)), 0) + 1
    except FileNotFoundError:
        pass
    return h
def decode_tps(key, rep):
    t1, t65 = ms.get((key, "t1_%s" % rep)), ms.get((key, "t65_%s" % rep))
    if not t1 or not t65 or t65 <= t1: return None
    _, n = text(key, "t65_%s" % rep)
    if n <= 1: return None
    return (n - 1) / ((t65 - t1) / 1000.0)
print("%-6s %-3s %9s %9s %8s %7s  %-14s %s" % (
    "arm", "ps", "dec t/s a", "dec t/s b", "flushes", "http", "uniq hist", "cache stats"))
per_arm = {}
for arm in arms:
    for ps in ("1", "2"):
        key = "%s-p%s" % (arm, ps)
        a, b = decode_tps(key, "a"), decode_tps(key, "b")
        codes = {v for (k, l), v in http.items() if k == key}
        h = uniq_hist(key)
        hs = " ".join("%d:%d" % (u, c) for u, c in sorted(h.items())) or "none"
        st = stats.get(key, "")
        m = re.search(r"hit%=([0-9.]+).*resident=(\d+)/(\d+)", st)
        stx = "hit%%=%s res=%s/%s" % m.groups() if m else "-"
        print("%-6s %-3s %9s %9s %8s %7s  %-14s %s" % (
            arm, ps,
            "%.3f" % a if a else "--", "%.3f" % b if b else "--",
            flush.get(key, "?"), ",".join(sorted(codes)) or "?", hs, stx))
        per_arm.setdefault(arm, []).extend([x for x in (a, b) if x])
print()
base = per_arm.get("bug", [])
fix = per_arm.get("fix", [])
if base and fix:
    mb, mf = statistics.mean(base), statistics.mean(fix)
    print("LEVER 1 expert-cache flush fix: bug mean=%.3f t/s (n=%d, %s)  fix mean=%.3f t/s (n=%d, %s)  -> %+.1f%%" % (
        mb, len(base), ["%.3f" % x for x in base], mf, len(fix), ["%.3f" % x for x in fix],
        100 * (mf / mb - 1)))
for arm in ("keep7", "keep6"):
    v = per_arm.get(arm, [])
    if v and fix:
        print("LEVER 2 %s vs fix: %.3f t/s (n=%d, %s) -> %+.1f%% over the fixed baseline" % (
            arm, statistics.mean(v), len(v), ["%.3f" % x for x in v],
            100 * (statistics.mean(v) / statistics.mean(fix) - 1)))
print()
print("output fidelity (t65_a text sha, greedy):")
for arm in arms:
    for ps in ("1", "2"):
        key = "%s-p%s" % (arm, ps)
        t, n = text(key, "t65_a")
        print("  %-9s n=%-3d sha=%s %r" % (
            key, n, hashlib.sha256(t.encode()).hexdigest()[:12], t[:48]))
PYEOF
chmod -R a+rX "$OUT"
note "decode A/B done"
echo DECODE_AB_DONE
