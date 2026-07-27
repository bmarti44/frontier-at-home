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
FIXTURE=$REPO/results/glm52-gates/harness/fixture-glm-short.json
PORT=${GLM_PORT:-8011}
IQ2_REFERENCE=${DS4_CUDA_IQ2_DOWN_REFERENCE:-1}
NO_EXPERT_TILES=${DS4_CUDA_MOE_NO_EXPERT_TILES:-0}
IQ2_ENV=()
TILE_ENV=()
PID=
START_TICKS=

[[ $SRC == /home/dsv4/ds4-project/src/* && -x $SRC/ds4-server ]] \
    || { echo "invalid GLM_CANDIDATE_SRC: $SRC" >&2; exit 2; }
[[ -r $TOKENIZER && $(sha256sum "$TOKENIZER" | awk '{print $1}') == "$TOKENIZER_SHA256" ]] \
    || { echo "GLM tokenizer identity mismatch: $TOKENIZER" >&2; exit 2; }
[[ $PORT =~ ^[0-9]+$ ]] \
    || { echo "GLM_PORT must be an integer from 1024 through 65535" >&2; exit 2; }
port=$((10#$PORT))
if (( port < 1024 || port > 65535 )); then
    echo "GLM_PORT must be an integer from 1024 through 65535" >&2
    exit 2
fi
PORT=$port
[[ $IQ2_REFERENCE == 0 || $IQ2_REFERENCE == 1 ]] \
    || { echo "IQ2_REFERENCE must be 0 or 1" >&2; exit 2; }
if [[ $IQ2_REFERENCE == 1 ]]; then
    IQ2_ENV+=(DS4_CUDA_IQ2_DOWN_REFERENCE=1)
fi
[[ $NO_EXPERT_TILES == 0 || $NO_EXPERT_TILES == 1 ]] \
    || { echo "NO_EXPERT_TILES must be 0 or 1" >&2; exit 2; }
if [[ $NO_EXPERT_TILES == 1 ]]; then
    TILE_ENV+=(DS4_CUDA_MOE_NO_EXPERT_TILES=1)
fi

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
/usr/bin/python3 - "$FIXTURE" "$OUT/request.json" "$SEED" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
request.update({
    "model": "glm-5.2",
    "max_tokens": 1,
    "temperature": 0,
    "seed": int(sys.argv[3]),
})
with open(sys.argv[2], "x", encoding="utf-8") as stream:
    json.dump(request, stream, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY
printf 'label=%s\niq2_reference=%s\nno_expert_tiles=%s\n' \
    "$LABEL" "$IQ2_REFERENCE" "$NO_EXPERT_TILES" \
    >"$OUT/runtime.config"
env DS4_TOKEN_TIMING_LOG=1 \
DS4_GLM_LOGIT_DUMP="$OUT/prefill.logits" \
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
DS4_CUDA_EXPERT_CACHE_GB=0 \
DS4_CUDA_EXPERT_CACHE_PIN=1 \
DS4_CUDA_FETCH_THREADS=6 \
DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "${TILE_ENV[@]}" \
    "${IQ2_ENV[@]}" \
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

curl --fail-with-body -sS --max-time 1200 \
    -H 'Content-Type: application/json' \
    --data-binary "@$OUT/request.json" \
    "http://127.0.0.1:$PORT/v1/completions" >"$OUT/response.json"

/usr/bin/python3 - "$OUT" <<'PY'
import json
import math
import pathlib
import struct
import sys

out = pathlib.Path(sys.argv[1])
response = json.loads((out / "response.json").read_text(encoding="utf-8"))
completion_tokens = response.get("usage", {}).get("completion_tokens")
if completion_tokens == 1:
    pass
else:
    raise SystemExit(f"completion_tokens != 1: {completion_tokens!r}")
raw = (out / "prefill.logits").read_bytes()
if len(raw) != 154880 * 4:
    raise SystemExit(f"wrong logit dump size: {len(raw)}")
logits = struct.unpack(f"<{len(raw) // 4}f", raw)
if not all(math.isfinite(value) for value in logits):
    raise SystemExit("non-finite logit")
PY

stop_server
trap - EXIT
