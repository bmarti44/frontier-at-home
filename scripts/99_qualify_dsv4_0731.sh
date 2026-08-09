#!/usr/bin/env bash
# Run the DeepSeek-V4-Flash-0731 requalification suite against an already
# running staged engine. Each stage writes its own artifact and a per-stage
# status line; a failing stage is recorded and does not abort the remaining
# stages, so one run produces a complete picture instead of stopping at the
# first problem.
#
# Preconditions: scripts/98_serve_dsv4_0731.sh is serving on $PORT.
set -uo pipefail

PORT=${PORT:-8021}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}
LABEL=${LABEL:-ds4-0731-mtp}
OUT=${OUT:-results/dsv4-0731-staging}
REPO_ROOT=${REPO_ROOT:-/home/bmarti44/spark-deepseek-v4-flash}
TOKENIZER=${TOKENIZER:-$REPO_ROOT/vendor/official-encoding/tokenizer.json}
STATUS=$OUT/qualification-status.tsv

mkdir -p -- "$OUT" "$OUT/transcripts"
: >"$STATUS"

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$STATUS"; }

run_stage() {
    local name=$1; shift
    printf '\n=== %s ===\n' "$name" >&2
    if "$@"; then
        record "$name" PASS
    else
        record "$name" "FAIL(rc=$?)"
    fi
}

# ds4-server exposes no /health route (the llama.cpp stack's default); its
# readiness signal is /v1/models answering 200. HEALTH_PATH is threaded into
# the golden harness for the same reason.
HEALTH_PATH=${HEALTH_PATH:-/v1/models}
curl -sf "$BASE_URL$HEALTH_PATH" >/dev/null \
    || { echo "engine not answering $HEALTH_PATH at $BASE_URL" >&2; exit 2; }

# vendor/ is gitignored, so a fresh clone or worktree has no encoder until it
# is generated. Materialize it from the pinned revision before any stage runs;
# 14_fetch_encoder.sh verifies what it fetches against
# configs/pins/official-encoding.json, and --verify-only re-checks an existing
# tree without downloading. Without this the golden needle check and the speed
# suite fail on a missing tokenizer, which looks like a model defect and is not.
if [[ -f vendor/official-encoding/tokenizer.json ]]; then
    run_stage encoder-verify bash scripts/14_fetch_encoder.sh --verify-only
else
    run_stage encoder-fetch bash scripts/14_fetch_encoder.sh
fi

# The published baselines (results/acc-*-ds4.json) were measured with thinking
# disabled: config_digest_payload.extra_body = {"enable_thinking": false}. 0731
# emits reasoning by default, so an unset extra_body changes the generation
# contract and makes the comparison meaningless -- a first run scored GSM8K dev
# 94/100 against the baseline's 98/100 with this as the only difference. Match
# the baseline exactly; measuring 0731's reasoning modes is a separate question
# from whether it can replace the current default.
EXTRA_BODY=${EXTRA_BODY:-'{"enable_thinking": false}'}

run_stage golden python3 scripts/32_golden_tests.py \
    --base-url "$BASE_URL" --out "$OUT/golden-0731.json" --stack-label "$LABEL" \
    --health-path "$HEALTH_PATH"

run_stage speed python3 scripts/30_bench_speed.py \
    --base-url "$BASE_URL" --out "$OUT/speed-0731.json" --stack-label "$LABEL" \
    --reps 5 --warmup 1 --max-tokens 256 --seed 42 \
    --context-levels 0,4096,16384 --tokenizer-path "$TOKENIZER"

for suite in gsm8k mmlu-pro humaneval; do
    run_stage "accuracy-$suite-dev" python3 scripts/31_bench_accuracy.py \
        --base-url "$BASE_URL" --out "$OUT/acc-$suite-dev-0731.json" \
        --stack-label "$LABEL" --suite "$suite" --split dev \
        --extra-body "$EXTRA_BODY" \
        --transcripts-dir "$OUT/transcripts/$suite-dev"
done

printf '\n=== qualification summary ===\n' >&2
cat "$STATUS" >&2
