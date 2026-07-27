#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

[[ $# == 3 ]] || { echo "usage: $0 OUT LABEL SEED" >&2; exit 2; }
OUT=$1
LABEL=$2
SEED=$3
REPO=/home/bmarti44/spark-deepseek-v4-flash
SRC=${GLM_CANDIDATE_SRC:-/home/dsv4/ds4-project/src/ds4-goal-clean-0a7ad776}
MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
TOKENIZER_SHA256=19e773648cb4e65de8660ea6365e10ac\
ca112d42a854923df93db4a6f333a82d
PORT=${GLM_PORT:-8011}
PID=
START_TICKS=
EXPERT_CACHE_GB=${GLM_EXPERT_CACHE_GB:-0}
REQUIRE_TOKEN_TIMING_LOG=${GLM_REQUIRE_TOKEN_TIMING_LOG:-1}

[[ $SRC == /home/dsv4/ds4-project/src/* && -x $SRC/ds4-server ]] \
    || { echo "invalid GLM_CANDIDATE_SRC: $SRC" >&2; exit 2; }
[[ -r $TOKENIZER && $(sha256sum "$TOKENIZER" | awk '{print $1}') == "$TOKENIZER_SHA256" ]] \
    || { echo "GLM tokenizer identity mismatch: $TOKENIZER" >&2; exit 2; }
[[ $EXPERT_CACHE_GB =~ ^([0-9]|[1-6][0-9]|7[0-2])$ ]] \
    || { echo "GLM_EXPERT_CACHE_GB must be an integer from 0 through 72" >&2; exit 2; }
[[ $REQUIRE_TOKEN_TIMING_LOG =~ ^[01]$ ]] \
    || { echo "GLM_REQUIRE_TOKEN_TIMING_LOG must be 0 or 1" >&2; exit 2; }
[[ $PORT =~ ^[0-9]+$ ]] \
    || { echo "GLM_PORT must be an integer from 1024 through 65535" >&2; exit 2; }
port=$((10#$PORT))
if (( port < 1024 || port > 65535 )); then
    echo "GLM_PORT must be an integer from 1024 through 65535" >&2
    exit 2
fi
PORT=$port

stop_server() {
    [[ ${PID:-} =~ ^[0-9]+$ ]] || return 0
    local current=
    current=$(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true)
    [[ -n $current && $current == "$START_TICKS" ]] || return 0
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        [[ ! -r /proc/$PID/stat ]] && return 0
        [[ $(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true) != "$START_TICKS" ]] && return 0
        sleep 2
    done
    kill -KILL "$PID" 2>/dev/null || true
}
trap stop_server EXIT

mkdir -p -- "$OUT"
DS4_TOKEN_TIMING_LOG=1 \
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
DS4_CUDA_EXPERT_CACHE_GB="$EXPERT_CACHE_GB" \
DS4_CUDA_EXPERT_CACHE_PIN=1 \
DS4_CUDA_FETCH_THREADS=6 \
DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "$SRC/ds4-server" --cuda -m "$MODEL" -c 8192 \
    --host 127.0.0.1 --port "$PORT" --ssd-streaming \
    --ssd-streaming-cache-experts 40GB \
    >"$OUT/server.log" 2>&1 &
PID=$!
START_TICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$PID $START_TICKS $(sha256sum "$SRC/ds4-server" | awk '{print $1}')" \
    >"$OUT/process.identity"

ready=false
for _ in $(seq 1 600); do
    if [[ $(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
        "http://127.0.0.1:$PORT/v1/models" || true) == 200 ]]; then
        ready=true
        break
    fi
    [[ -r /proc/$PID/stat ]] || break
    [[ $(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true) == "$START_TICKS" ]] || break
    sleep 2
done
"$ready" || { tail -80 "$OUT/server.log" >&2; exit 1; }

timing_args=()
if [[ $REQUIRE_TOKEN_TIMING_LOG == 1 ]]; then
    timing_args=(--token-timing-log "$OUT/server.log")
fi

/home/bmarti44/spark-deepseek-v4-flash/.venv-harness/bin/python \
    "$REPO/scripts/30_bench_speed.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --out "$OUT/result.json" \
    --stack-label "$LABEL" \
    --model-id glm-5.2 \
    --output-tokenizer-path "$TOKENIZER" \
    --output-tokenizer-sha256 "$TOKENIZER_SHA256" \
    "${timing_args[@]}" \
    --reps 1 --context-levels 0 --max-tokens 160 \
    --min-completion-tokens 128 --seed "$SEED"

stop_server
trap - EXIT
