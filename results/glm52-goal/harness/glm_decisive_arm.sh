#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

[[ $# == 3 ]] || { echo "usage: $0 OUT LABEL SEED" >&2; exit 2; }
OUT=$1
LABEL=$2
SEED=$3
REPO=/home/bmarti44/spark-deepseek-v4-flash
SRC=${GLM_CANDIDATE_SRC:?GLM_CANDIDATE_SRC is required}
EXPECTED_BINARY_SHA256=${GLM_SAFE_EXPECTED_BINARY_SHA256:?GLM_SAFE_EXPECTED_BINARY_SHA256 is required}
MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
EXPECTED_MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
EXPECTED_MODEL_IDENTITY=${GLM_VERIFIED_MODEL_DEVICE_INODE_SIZE:?GLM_VERIFIED_MODEL_DEVICE_INODE_SIZE is required}
TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
TOKENIZER_SHA256=19e773648cb4e65de8660ea6365e10ac\
ca112d42a854923df93db4a6f333a82d
PORT=${GLM_PORT:-8011}
CACHE_GB=${GLM_EXPERT_CACHE_GB:-0}
IQ2_REFERENCE=${DS4_CUDA_IQ2_DOWN_REFERENCE:-1}
NO_EXPERT_TILES=${DS4_CUDA_MOE_NO_EXPERT_TILES:-0}
IQ2_ENV=()
TILE_ENV=()
PID=
START_TICKS=

[[ $SRC == /home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6 && -x $SRC/ds4-server ]] \
    || { echo "invalid GLM_CANDIDATE_SRC: $SRC" >&2; exit 2; }
actual_binary_sha256=$(sha256sum -- "$SRC/ds4-server" | awk '{print $1}')
[[ $actual_binary_sha256 == "$EXPECTED_BINARY_SHA256" ]] \
    || { echo "GLM candidate binary identity mismatch" >&2; exit 2; }
actual_model_identity=$(stat -Lc '%d:%i:%s' -- "$MODEL")
[[ $actual_model_identity == "$EXPECTED_MODEL_IDENTITY" ]] \
    || { echo "GLM model device/inode/size changed after campaign verification" >&2; exit 2; }
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
[[ $CACHE_GB =~ ^[0-9]+$ ]] \
    || { echo "CACHE_GB must be an integer from 0 through 40" >&2; exit 2; }
cache_gb=$((10#$CACHE_GB))
if (( cache_gb < 0 || cache_gb > 40 )); then
    echo "CACHE_GB must be an integer from 0 through 40" >&2
    exit 2
fi
CACHE_GB=$cache_gb
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
printf 'context_cap=32768\nexpert_cache_gib=%s\niq2_reference=%s\nno_expert_tiles=%s\nstable_model_remap=1\nmodel_sha256=%s\n' \
    "$CACHE_GB" "$IQ2_REFERENCE" "$NO_EXPERT_TILES" \
    "$EXPECTED_MODEL_SHA256" >"$OUT/runtime.config"
env DS4_TOKEN_TIMING_LOG=1 \
DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
DS4_CUDA_STABLE_MODEL_REMAP=1 \
DS4_CUDA_EXPERT_CACHE_GB="$CACHE_GB" \
DS4_CUDA_EXPERT_CACHE_PIN=1 \
DS4_CUDA_FETCH_THREADS=6 \
DS4_CUDA_EXPERT_CACHE_SLRU=1 \
    "${TILE_ENV[@]}" \
    "${IQ2_ENV[@]}" \
    "$SRC/ds4-server" --cuda -m "$MODEL" -c 32768 \
    --host 127.0.0.1 --port "$PORT" --ssd-streaming \
    --ssd-streaming-cache-experts 40GB \
    >"$OUT/server.log" 2>&1 &
PID=$!
START_TICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$PID $START_TICKS $(sha256sum "$SRC/ds4-server" | awk '{print $1}')" \
    >"$OUT/process.identity"
cp -- /proc/sys/kernel/random/boot_id "$OUT/host.boot_id"

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

python3 - "$PID" "$MODEL" "$OUT" "$actual_binary_sha256" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

pid = int(sys.argv[1])
model = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])
binary_sha256 = sys.argv[4]
names = (
    "DS4_CUDA_EXPERT_CACHE_GB",
    "DS4_CUDA_EXPERT_CACHE_PIN",
    "DS4_CUDA_EXPERT_CACHE_SLRU",
    "DS4_CUDA_FETCH_THREADS",
    "DS4_CUDA_IQ2_DOWN_REFERENCE",
    "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
    "DS4_CUDA_STABLE_MODEL_REMAP",
    "DS4_TOKEN_TIMING_LOG",
)
raw_environment = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
all_environment = {}
for entry in raw_environment:
    if not entry:
        continue
    key, separator, value = entry.partition(b"=")
    if not separator:
        raise SystemExit("malformed live environment")
    all_environment[key.decode("ascii")] = value.decode("ascii")
environment = {name: all_environment[name] for name in names}
canonical = "".join(f"{name}={environment[name]}\n" for name in sorted(environment))
(out / "process.environment").write_text(
    json.dumps(
        {"environment": environment, "sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n",
    encoding="ascii",
)
argv = [value.decode("utf-8") for value in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]
model_info = model.stat()
model_identity = f"{model_info.st_dev}:{model_info.st_ino}:{model_info.st_size}"
try:
    context_index = argv.index("-c")
    context_cap = int(argv[context_index + 1])
except (ValueError, IndexError):
    raise SystemExit("live command has no context cap")
(out / "process.command").write_text(
    json.dumps(
        {
            "argv": argv,
            "binary_sha256": binary_sha256,
            "context_cap": context_cap,
            "model_device_inode_size": model_identity,
            "stable_model_remap": environment["DS4_CUDA_STABLE_MODEL_REMAP"] == "1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n",
    encoding="ascii",
)
(out / "model.device-inode-size").write_text(model_identity + "\n", encoding="ascii")
PY

/home/bmarti44/spark-deepseek-v4-flash/.venv-harness/bin/python \
    "$REPO/scripts/30_bench_speed.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --out "$OUT/result.json" \
    --stack-label "$LABEL" \
    --model-id glm-5.2 \
    --output-tokenizer-path "$TOKENIZER" \
    --output-tokenizer-sha256 "$TOKENIZER_SHA256" \
    --token-timing-log "$OUT/server.log" \
    --prompt-count-log "$OUT/server.log" --prompt-count-format ds4 \
    --reps 2 --context-levels 0,28672 --max-tokens 160 \
    --min-completion-tokens 128 --request-timeout 2700 --seed "$SEED"

stop_server
trap - EXIT
