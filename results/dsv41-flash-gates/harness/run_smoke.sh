#!/usr/bin/env bash
# Contained DeepSeek V4.1 Flash CUDA smoke runner (fidelity/correctness only).
# usage: run_smoke.sh text|fidelity OUTDIR [CACHE]   (expert cache, default 42gb)
#        run_smoke.sh diag OUTDIR CACHE   (diagnostic only: -n 16, expert cache CACHE e.g. 4gb)
# Preregistration: ../smoke-2026-09-25/PREREGISTRATION.md
set -euo pipefail
readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SRC=/home/bmarti44/.cache/ds4-v41-0aaea5a2
readonly MODEL=/home/bmarti44/models/deepseek-v4.1-flash/DeepSeek-V4.1-Flash-Q2.gguf
readonly FIX=$SRC/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router
readonly WRAPPER=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly LOCK=/run/lock/frontier-at-home/inference.lock
readonly FROZEN=$REPO/results/dsv41-flash-gates/smoke-2026-09-25/frozen-inputs-c8b.json
readonly BUNDLER=$REPO/results/dsv41-flash-gates/harness/bundle_attempt.py
readonly PROMPT='Explain in three sentences why the sky is blue.'
readonly LONG_PROMPT='Write a detailed, multi-paragraph explanation of how a CPU pipeline works, covering fetch, decode, execute, memory access, write-back, hazards, and branch prediction.'

arm=${1:?arm}; out=${2:?outdir}
# Expert-cache size: the fidelity arm at 42gb logged an NVRM
# NV_ERR_NO_MEMORY line (fidelity-c8-20260926T163851/NOTE.md).
cache=${3:-42gb}
[[ $cache =~ ^[0-9]{1,3}gb$ ]] || { echo "bad cache $cache" >&2; exit 2; }
# A fresh, empty output directory per attempt (review round 2, finding 1).
if [[ -e $out && -n $(ls -A "$out" 2>/dev/null) ]]; then
  echo "output directory is not empty: $out" >&2; exit 4
fi
mkdir -p "$out"
out=$(realpath -e "$out")
# Match real engine executables (exe basename) or python vLLM servers; never
# match shell command lines that merely mention an engine name.
engines=$(for d in /proc/[0-9]*; do
  exe=$(readlink "$d/exe" 2>/dev/null) || continue
  case ${exe##*/} in
    ds4|ds4-server|ds4-agent|ds4-bench|ds4-eval|score_official|llama-server) echo "${d#/proc/} $exe" ;;
    python3*) tr '\0' ' ' < "$d/cmdline" 2>/dev/null | grep -q 'vllm' && echo "${d#/proc/} vllm" ;;
  esac
done)
[[ -z $engines ]] || { echo "another engine is running: $engines" >&2; exit 3; }
python3 "$REPO/scripts/03_memory_guard.py" --required-gib 110 --timeout-seconds 600

# Binary and model identity against the frozen candidate, before and after the
# run (review finding H2). Binaries are read-only; the model is checked by
# size/inode/mtime against its verified full-file sha256 record.
identity_check() {  # $1 = binary path, $2 = phase
  python3 - "$FROZEN" "$1" "$MODEL" "$2" <<'PY'
import hashlib, json, os, sys
frozen, binary, model, phase = sys.argv[1:]
f = json.load(open(frozen))
h = hashlib.sha256(open(binary, "rb").read()).hexdigest()
want = f["artifact_sha256"].get(os.path.basename(binary))
st = os.stat(model)
mi = f["model_identity"]
ok = (h == want and os.access(binary, os.W_OK) is False and
      (st.st_size, st.st_ino, st.st_mtime_ns) == (mi["bytes"], mi["inode"], mi["mtime_ns"]) and
      not os.access(model, os.W_OK))
print(json.dumps({"phase": phase, "binary": binary, "binary_sha256": h, "expected_sha256": want,
                  "model_bytes": st.st_size, "model_inode": st.st_ino, "model_mtime_ns": st.st_mtime_ns,
                  "ok": ok}, sort_keys=True))
sys.exit(0 if ok else 21)
PY
}

# Every frozen harness, scorer, fixture and reference hash must match: at
# launch, after the run and after scoring (review round 2 H3 remainder;
# round 3 finding 3). $1 = output file name. The bundler requires every
# check it expects to be present and ok. Round 4 (finding 3 remainder): the
# frozen-inputs file itself must equal its committed blob at HEAD with no
# uncommitted change to any frozen path, and every phase records HEAD and the
# actual component hashes so the bundler can require them unchanged. Round 5:
# the logic lives in frozen_check.py, which the bundler re-verifies against git.
frozen_check() {
  python3 "$REPO/results/dsv41-flash-gates/harness/frozen_check.py" "$FROZEN" "$1" > "$out/$1"
}
frozen_check frozen-check.json || { echo "frozen input mismatch" >&2; exit 22; }

case $arm in
  text)
    timeout_s=900
    cmd=("$SRC/ds4" --cuda -m "$MODEL" --ssd-streaming --ssd-streaming-cache-experts "$cache"
         -c 8192 --nothink --temp 0 -n 256 -p "$LONG_PROMPT"
         --dump-logprobs "$out/steps.json" --logprobs-top-k 5) ;;
  fidelity)
    timeout_s=5400
    cmd=("$SRC/gguf-tools/quality-testing/score_official" "$MODEL" "$FIX/manifest.tsv"
         "$out/cuda.tsv" 34816 --ssd-streaming --ssd-streaming-cache-experts "$cache") ;;
  diag)
    timeout_s=900
    [[ -n ${3:-} ]] || { echo "diag needs CACHE, e.g. 4gb" >&2; exit 2; }
    cmd=("$SRC/ds4" --cuda -m "$MODEL" --ssd-streaming --ssd-streaming-cache-experts "$cache"
         -c 8192 --nothink --temp 0 -n 16 -p "$PROMPT") ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac

pre=$(identity_check "${cmd[0]}" pre) || { echo "$pre" > "$out/identity-pre.json"; echo "identity check failed before launch" >&2; exit 21; }
echo "$pre" > "$out/identity-pre.json"
unit=glm52-dsv41-smoke-$arm-$(date -u +%Y%m%dT%H%M%S)
printf '%s\n' "$unit" > "$out/unit.txt"
printf '%q ' "${cmd[@]}" > "$out/command.txt"; echo >> "$out/command.txt"
date -u --iso-8601=ns > "$out/started_at.txt"
set +e
/usr/bin/flock -n -E 75 "$LOCK" \
  systemd-run --user --unit "$unit" --wait --collect --pipe --quiet \
    --working-directory="$SRC" \
    -p MemorySwapMax=0 -p OOMPolicy=kill -p KillMode=control-group \
    -p MemoryHigh=72G -p MemoryMax=76G \
    -E GLM_SAFE_RUN_AS_CURRENT_USER=1 -E GLM_SAFE_REQUIRE_CGROUP=1 \
    -E GLM_SAFE_CGROUP_UNIT="$unit" -E GLM_SAFE_KILL_FLOOR_GIB=40 \
    -E GLM_SAFE_MIN_START_GIB=110 -E GLM_SAFE_TIMEOUT_S="$timeout_s" \
    /bin/bash "$WRAPPER" --tag "$unit" -- "${cmd[@]}" \
    > "$out/stdout.txt" 2> "$out/stderr.txt"
rc=$?
set -e
date -u --iso-8601=ns > "$out/finished_at.txt"
echo "$rc" > "$out/exit_code.txt"
crash=$(ls -d /home/bmarti44/.local/state/glm52-crashlog/*-"$unit" 2>/dev/null | tail -1 || true)
[[ -n $crash ]] && cp -a "$crash" "$out/containment"
journalctl -k --since "$(date -d "$(sed 's/,/./' "$out/started_at.txt")" '+%Y-%m-%d %H:%M:%S')" --no-pager 2>/dev/null | grep -E 'NVRM|Xid|oom' > "$out/kernel-events.txt" || true
post_rc=0
post=$(identity_check "${cmd[0]}" post) || post_rc=$?
echo "$post" > "$out/identity-post.json"
python3 - "$out" "$post_rc" <<'PY'
import json, os, sys
d, post_rc = sys.argv[1], int(sys.argv[2])
pre = json.load(open(os.path.join(d, "identity-pre.json")))
post = json.load(open(os.path.join(d, "identity-post.json")))
json.dump({"verified": pre["ok"] and post["ok"] and post_rc == 0, "pre": pre, "post": post},
          open(os.path.join(d, "identity.json"), "w"), indent=1, sort_keys=True)
PY
# A mismatch after launch is recorded, not fatal here: the bundler FAILs it.
frozen_check frozen-check-post-run.json || true
if [[ $arm == fidelity && -f $out/cuda.tsv ]]; then
  # Score from in-bundle copies so the bundler can re-run the frozen scorer
  # on exactly these bytes (review round 3, finding 1).
  cp "$FIX/results/base-default.tsv" "$out/metal-reference.tsv"
  cp "$FIX/manifest.tsv" "$out/fixture-manifest.tsv"
  score_rc=0
  python3 "$REPO/results/dsv41-flash-gates/harness/score_fidelity.py" "$out/cuda.tsv" \
    "$out/metal-reference.tsv" "$out/fixture-manifest.tsv" "$out/fidelity-summary.json" \
    > "$out/fidelity-score.log" 2>&1 || score_rc=$?
  echo "$score_rc" > "$out/fidelity-score-exit.txt"
  frozen_check frozen-check-post-score.json || true
fi
# The runner's exit status is the bundled verdict (0 PASS, 1 FAIL,
# 2 PENDING_REVIEW); the engine's own status stays in exit_code.txt.
verdict_rc=0
python3 "$BUNDLER" "$out" "$arm" || verdict_rc=$?
case $verdict_rc in 0|1|2) ;; *) echo "bundler failed rc=$verdict_rc" >&2; verdict_rc=3 ;; esac
echo "arm=$arm engine_rc=$rc verdict_rc=$verdict_rc out=$out"
exit "$verdict_rc"
