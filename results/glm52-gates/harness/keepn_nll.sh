#!/bin/bash
# FIX-B part 3: the FIDELITY gate for expert skipping.
#
# "Coherent English, no invalid UTF-8, low 3-gram repetition" is a liveness
# check, not a fidelity measurement -- it cannot see factual degradation,
# reasoning collapse, or broken code. This runs the committed 100-case
# reference suite (glm52-openrouter-100) and reports average NLL against the
# official reference continuations plus first-token agreement, at keep-8
# (control) vs the byte-reducing keep-7 and keep-6 arms.
#
# NLL is computed by score_official, which links the same core objects as
# ds4-server, so DS4_GLM_TOPK_KEEP / DS4_GLM_TOPK_SKIP_LOAD apply identically.
# The binary MUST be relinked after any ds4.c change -- verified below by
# comparing its mtime against ds4.o, and the sha is recorded per arm.
set -u
OUT=/home/dsv4/ds4-project/glm52-keepn-nll
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
MANI=gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv
SCORER=gguf-tools/quality-testing/score_official
# Teacher-forced scoring runs at prefill rate (~23 tok/s on ~2300-token cases),
# so the full 100-case suite costs ~2.8 h PER ARM. NLL_CASES caps the case
# count; the arms are paired on the SAME subset, so the paired delta stays
# valid -- only the width of the confidence interval changes. The count is
# printed in the summary so no result can quietly claim n=100.
NLL_CASES=${NLL_CASES:-30}
rm -rf "$OUT"; mkdir -p "$OUT"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
pkill -TERM -x ds4-server 2>/dev/null
for i in $(seq 1 90); do pgrep -x ds4-server > /dev/null || break; sleep 2; done

if [[ "$SRC/$SCORER" -ot "$SRC/ds4.o" ]]; then
  note "FATAL scorer older than ds4.o -- relink required, refusing to score stale code"
  echo KEEPN_NLL_STALE; exit 1
fi
note "scorer sha12=$(sha256sum "$SRC/$SCORER" | cut -c1-12) ds4.o sha12=$(sha256sum "$SRC/ds4.o" | cut -c1-12)"

# Build the capped manifest once and point every arm at it, so the arms are
# paired case-for-case.
SUB="$OUT/manifest-$NLL_CASES.tsv"
{ grep '^#' "$SRC/$MANI" || true; grep -v '^#' "$SRC/$MANI" | head -n "$NLL_CASES"; } > "$SUB"
note "cases=$(grep -vc '^#' "$SUB") of $(grep -vc '^#' "$SRC/$MANI") available"

run_arm() { # $1 tag, $2 keepN (0=off), $3 skip_load(1/0)
  local envs=()
  [[ "$2" != "0" ]] && envs+=("DS4_GLM_TOPK_KEEP=$2")
  [[ "$3" == "1" ]] && envs+=("DS4_GLM_TOPK_SKIP_LOAD=1")
  note "NLL arm $1 keep=$2 skip_load=$3"
  local t0=$(date +%s)
  (cd "$SRC" && env "${envs[@]}" \
     DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
     DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
     DS4_CUDA_EXPERT_CACHE_SLRU=1 \
     ./$SCORER "$GGUF" "$SUB" "$OUT/q100-$1.tsv" 8192 \
     --ssd-streaming --ssd-streaming-cache-experts 40GB) \
     > "$OUT/q100-$1.log" 2>&1
  note "arm $1 exit=$? elapsed_s=$(( $(date +%s) - t0 ))"
}
run_arm keep8  0 0
run_arm keep7s 7 1
run_arm keep6s 6 1

python3 - "$OUT" <<'PYEOF' | tee "$OUT/summary"
import os, statistics, sys
out = sys.argv[1]
def load(tag):
    p = os.path.join(out, "q100-%s.tsv" % tag)
    try:
        rows = [l.rstrip("\n").split("\t") for l in open(p) if not l.startswith("#")]
    except FileNotFoundError:
        return None
    nll = [float(r[4]) for r in rows if len(r) > 4]
    first = [int(r[5]) for r in rows if len(r) > 5]
    return {"n": len(nll), "nll": nll, "first": first}
base = load("keep8")
print("%-7s %4s %9s %9s %9s %9s" % ("arm", "n", "mean_nll", "median", "p90", "first_tok%"))
print("(n is the number of scored cases -- the suite has 100 available; see run.log)")
for tag in ("keep8", "keep7s", "keep6s"):
    d = load(tag)
    if not d or not d["n"]:
        print("%-7s   -- no data (scorer failed; see q100-%s.log)" % (tag, tag)); continue
    s = sorted(d["nll"])
    print("%-7s %4d %9.4f %9.4f %9.4f %9.1f" % (
        tag, d["n"], statistics.mean(d["nll"]), statistics.median(d["nll"]),
        s[int(len(s) * 0.9)],
        100.0 * sum(d["first"]) / len(d["first"]) if d["first"] else float("nan")))
if base and base["n"]:
    b = statistics.mean(base["nll"])
    print("\nfidelity cost vs keep-8 control (positive = worse):")
    for tag in ("keep7s", "keep6s"):
        d = load(tag)
        if not d or not d["n"] or d["n"] != base["n"]:
            print("  %-7s not comparable (n mismatch or missing)" % tag); continue
        # paired, since the same 100 prompts are scored in every arm
        deltas = [x - y for x, y in zip(d["nll"], base["nll"])]
        mean_d = statistics.mean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        se = sd / (len(deltas) ** 0.5) if deltas else 0.0
        worse = sum(1 for x in deltas if x > 0)
        print("  %-7s mean_delta_nll=%+.4f (%+.2f%%) 95%%CI=[%+.4f,%+.4f] worse_on=%d/%d" % (
            tag, mean_d, 100.0 * mean_d / b, mean_d - 1.96 * se, mean_d + 1.96 * se,
            worse, len(deltas)))
    print("\nPaired 95%% CI excluding 0 means the fidelity change is real, not noise.")
PYEOF
chmod -R a+rX "$OUT"
note "keep-N NLL done"
echo KEEPN_NLL_DONE
