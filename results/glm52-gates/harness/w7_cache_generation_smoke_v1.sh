#!/usr/bin/env bash
# One-arm W7.1 production-path smoke. The outer caller must use
# glm_cgroup_run.sh, which supplies glm_safe_run.sh and process containment.
set -Eeuo pipefail
umask 077

readonly BIN=/home/bmarti44/.cache/glm52-w7-stable-remap-bccf0b6/ds4-server
readonly BINARY_SHA256=eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly MODEL_BYTES=211075856448
readonly LIVE=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/live-request.json
readonly LIVE_SHA256=d1def599a8bbfcd3a49e97d3c467fe30264caa241e9fa7cf717e5550c2bb601a
readonly PRIMARY=/home/bmarti44/.local/state/glm52-w7-red/attempt-22decf741c3dafa862eb08dc28aee7e8/primary-request.json
readonly PRIMARY_SHA256=a453691312004c144474d0fc8f27c17e38aec055a353a20bb2e9946f265667f3
readonly PORT=8097
server_pid=

verify_file() {
  [[ -f $1 && ! -L $1 && $(sha256sum -- "$1" | awk '{print $1}') == "$2" ]]
}

stop_server() {
  [[ ${server_pid:-} =~ ^[0-9]+$ ]] || return 0
  kill -TERM "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 300); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=
}

verify_dependencies() {
  verify_file "$BIN" "$BINARY_SHA256"
  [[ -f $MODEL && ! -L $MODEL && $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]]
  verify_file "$LIVE" "$LIVE_SHA256"
  verify_file "$PRIMARY" "$PRIMARY_SHA256"
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]]
  verify_dependencies
  echo W7_CACHE_GENERATION_SMOKE_SELFTEST_OK
  exit 0
fi

[[ $# == 2 && ( $1 == off || $1 == on ) ]] || exit 2
readonly mode=$1
readonly out=$2
[[ $out =~ ^/home/bmarti44/\.local/state/glm52-w7-cache-generation/attempt-[0-9a-f]{32}/(off|on)$ ]]
[[ ${out##*/} == "$mode" && -d $out && ! -L $out && -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]
if [[ $mode == on ]]; then
  [[ ${DS4_CUDA_STABLE_MODEL_REMAP:-} == 1 ]]
else
  [[ ! -v DS4_CUDA_STABLE_MODEL_REMAP ]]
fi
verify_dependencies
mkdir "$out/kv"

"$BIN" --cuda -m "$MODEL" -c 8192 --host 127.0.0.1 --port "$PORT" \
  --ssd-streaming --ssd-streaming-cache-experts 40GB \
  --kv-disk-dir "$out/kv" --kv-disk-space-mb 4096 \
  --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 8 \
  >"$out/server.log" 2>&1 &
server_pid=$!
trap stop_server EXIT INT TERM HUP

code=
for _ in $(seq 1 600); do
  kill -0 "$server_pid" 2>/dev/null || exit 1
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
    "http://127.0.0.1:$PORT/v1/models" || true)
  [[ $code == 200 ]] && break
  sleep 1
done
[[ $code == 200 ]]

curl -sS --fail-with-body --max-time 900 -H 'Content-Type: application/json' \
  -o "$out/live-response.json" -w '%{http_code}\n' -d @"$LIVE" \
  "http://127.0.0.1:$PORT/v1/completions" >"$out/live-http-status"
curl -sS --fail-with-body --max-time 1200 -H 'Content-Type: application/json' \
  -o "$out/primary-response.json" -w '%{http_code}\n' -d @"$PRIMARY" \
  "http://127.0.0.1:$PORT/v1/completions" >"$out/primary-http-status"
stop_server
trap - EXIT INT TERM HUP
sync -f "$out"
