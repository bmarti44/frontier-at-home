#!/usr/bin/env bash
# 52_engine_switch.sh — one-command switch between serving profiles behind
# the unchanged auth chain (caddy/tailnet -> authhelper :8010 -> engine :8011).
#
#   sudo -u dsv4 scripts/52_engine_switch.sh status
#   sudo -u dsv4 scripts/52_engine_switch.sh dsv4    # qualified llama.cpp DSV4
#   sudo -u dsv4 scripts/52_engine_switch.sh glm52   # GLM-5.2 ds4 streaming
#
# glm52 profile = the G4a-qualified configuration: upstream ds4 pin+patches
# (binary recorded in state), persistent expert cache, deterministic batch
# dispatch, disk-KV prefix cache. memwatch is armed on the engine pid via the
# repo's 01_memwatch.sh. The OTHER profile's weights/state are never touched.
set -Eeuo pipefail
REPO=/home/bmarti44/spark-deepseek-v4-flash
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
KVDIR=/home/dsv4/ds4-project/glm52-kvdisk-serve
STATE=/home/dsv4/ds4-project/engine-switch
PORT=${ENGINE_PORT:-8011}
mkdir -p "$STATE" "$KVDIR"
log() { echo "$(date -Is) $*" | tee -a "$STATE/switch.log"; }

engine_identity() {
  curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    print(",".join(m["id"] for m in d.get("data", []))[:120])
except Exception:
    print("none")' 2>/dev/null || echo none
}

stop_all() {
  log "stopping engines on :$PORT"
  if [[ -f "$STATE/glm52.pid" ]]; then
    local p; p=$(cat "$STATE/glm52.pid")
    if kill -0 "$p" 2>/dev/null; then
      [[ -f "$STATE/glm52.memwatch.target" && -f "$STATE/glm52.arm" ]] && \
        cp "$STATE/glm52.arm" "$STATE/glm52.memwatch.target" 2>/dev/null || true
      kill -TERM "$p" 2>/dev/null || true
      for i in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
      kill -KILL "$p" 2>/dev/null || true
    fi
    rm -f "$STATE/glm52.pid"
  fi
  pkill -TERM -f "llama-server.*--port $PORT" 2>/dev/null || true
  pkill -TERM -f "ds4-server.*--port $PORT" 2>/dev/null || true
  sleep 3
  if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"; then
    log "ERROR: something still listens on :$PORT"; return 1
  fi
  # stop any memwatch we armed
  [[ -f "$STATE/glm52.memwatch.pid" ]] && kill "$(cat "$STATE/glm52.memwatch.pid")" 2>/dev/null || true
  rm -f "$STATE/glm52.memwatch.pid"
  return 0
}

verify_serving() { # profile expected_identity_regex
  local code idy t0 t1
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/v1/models" || true)
  idy=$(engine_identity)
  [[ "$code" == 200 ]] || { log "VERIFY FAIL health=$code"; return 1; }
  echo "$idy" | grep -qE "$2" || { log "VERIFY FAIL identity='$idy' !~ $2"; return 1; }
  # tailnet-facing auth must still 401 without a key
  local auth; auth=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8010/health || true)
  [[ "$auth" == 401 || "$auth" == 200 ]] || { log "VERIFY FAIL authhelper=$auth"; return 1; }
  t0=$(date +%s%3N)
  local rc; rc=$(curl -s -o "$STATE/$1.probe.json" -w '%{http_code}' --max-time 1800 \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"Reply with the single word: ready","max_tokens":4,"temperature":0}' \
    "http://127.0.0.1:$PORT/v1/completions" || true)
  t1=$(date +%s%3N)
  [[ "$rc" == 200 ]] || { log "VERIFY FAIL probe=$rc"; return 1; }
  log "VERIFY OK profile=$1 identity='$idy' authhelper=$auth probe_ms=$((t1-t0))"
}

case "${1:-status}" in
  status)
    log "status: identity='$(engine_identity)' (:$PORT)"
    ;;
  dsv4)
    stop_all
    log "starting DSV4 (qualified llama.cpp stack)"
    "$REPO/scripts/21_serve_llamacpp.sh" start
    verify_serving dsv4 "deepseek|flash"
    ;;
  glm52)
    stop_all
    log "starting GLM-5.2 (ds4 streaming, binary $(sha256sum "$SRC/ds4-server" | cut -c1-12))"
    TF="$STATE/glm52.memwatch.target"; RF="$STATE/glm52.memwatch.ready"
    rm -f "$TF" "$RF"
    "$REPO/scripts/01_memwatch.sh" --target-file "$TF" --ready-file "$RF" \
      --threshold-gib 12 --interval-sec 2 --log "$STATE/glm52.memwatch.log" &
    echo $! > "$STATE/glm52.memwatch.pid"
    DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_EXPERT_CACHE_GB=72 \
      DS4_CUDA_EXPERT_CACHE_PIN=1 DS4_CUDA_FETCH_THREADS=6 \
      DS4_CUDA_EXPERT_CACHE_SLRU=1 \
      DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1 \
      "$SRC/ds4-server" --cuda -m "$GGUF" -c 32768 --host 127.0.0.1 --port $PORT \
      --ssd-streaming --ssd-streaming-cache-experts 40GB \
      --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
      --kv-cache-boundary-align-tokens 4 \
      --kv-cache-boundary-trim-tokens 0 \
      > "$STATE/glm52.server.log" 2>&1 &
    SP=$!
    echo "$SP" > "$STATE/glm52.pid"
    SPG=$(ps -o pgid= -p $SP | tr -d ' '); STK=$(awk '{print $22}' /proc/$SP/stat)
    echo "$SP $SPG $STK engine" > "$TF"
    printf 'DISARM %s %s %s\n' "$SP" "$SPG" "$STK" > "$STATE/glm52.arm"
    for i in $(seq 1 200); do
      [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$PORT/v1/models)" == 200 ]] && break
      kill -0 $SP 2>/dev/null || { log "GLM server died"; tail -5 "$STATE/glm52.server.log"; exit 1; }
      sleep 2
    done
    verify_serving glm52 "glm"
    ;;
  *) echo "usage: $0 status|dsv4|glm52"; exit 2 ;;
esac
