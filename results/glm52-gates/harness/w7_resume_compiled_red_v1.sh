#!/bin/bash
# Production-path RED for W7. This intentionally runs the unchanged GLM
# engine and fails its desired-behavior assertion when the strict resume guard
# converts the selected 5,028-token disk checkpoint into a full cold restart.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly SCRIPT=$(readlink -f -- "$0")
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly MEMORY_GUARD=$REPO/scripts/03_memory_guard.py
readonly STEM=$REPO/results/glm52-gates/harness/fixture-glm-long8.json
readonly POOL=$REPO/results/glm52-gates/harness/w7-production-fixture-pool-v1.json
readonly POOL_SHA256=74829fc094143b4a494e104c186e4146f2e669eaceedbb354a7ab35251083d90
readonly TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
readonly TOKENIZER_RUNTIME=/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-0.22.2
readonly CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w7-d652a9b5
readonly BIN=$CANDIDATE_SRC/ds4-server
readonly BINARY_SHA256=56263e7cda1879e0322526f34ef2c3aeacf30aa4724d22bb13562324a0e077a4
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly MODEL_BYTES=211075856448
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly OUT_PARENT=/home/bmarti44/.local/state/glm52-w7-red
readonly CRASH_ROOT=/home/bmarti44/.local/state/glm52-crashlog
readonly PORT=8097
readonly CACHE_GIB=40

fixture_check() {
  /usr/bin/python3 -I -B - "$STEM" "$POOL" "$TOKENIZER" "$TOKENIZER_RUNTIME" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[4])
from tokenizers import Tokenizer

stem_path, pool_path, tokenizer_path = sys.argv[1:4]
stem = json.load(open(stem_path, encoding="utf-8"))["prompt"]
pool = json.load(open(pool_path, encoding="utf-8"))
primary = next(item for item in pool["variants"] if item["variant"] == "primary-fixed")
suffix = "\n\n[W7 primary fixed] Explain why a restored prefix must be rewound before this appended request."
tokenizer = Tokenizer.from_file(tokenizer_path)
live_wire = stem + pool["live"]["suffix_utf8"]
primary_wire = stem + suffix
live_tokens = tokenizer.encode(live_wire, add_special_tokens=False).ids
primary_tokens = tokenizer.encode(primary_wire, add_special_tokens=False).ids
common = 0
for left, right in zip(live_tokens, primary_tokens):
    if left != right:
        break
    common += 1
observed = {
    "selected": primary["selected_tokens"],
    "common": common,
    "live": len(live_tokens),
    "prompt": len(primary_tokens),
}
expected = {"selected": 5028, "common": 5029, "live": 5037, "prompt": 5048}
if observed != expected:
    raise SystemExit(f"frozen W7 geometry mismatch: {observed!r}")
print(json.dumps(observed, sort_keys=True, separators=(",", ":")))
PY
}

write_requests() {
  local out=$1
  /usr/bin/python3 -I -B - "$STEM" "$POOL" "$out" <<'PY'
import json
import pathlib
import sys

stem_path, pool_path, out_raw = sys.argv[1:]
out = pathlib.Path(out_raw)
stem = json.load(open(stem_path, encoding="utf-8"))["prompt"]
pool = json.load(open(pool_path, encoding="utf-8"))
primary_suffix = "\n\n[W7 primary fixed] Explain why a restored prefix must be rewound before this appended request."
requests = {
    "live-request.json": {"model": "default", "prompt": stem + pool["live"]["suffix_utf8"], "max_tokens": 0, "temperature": 0},
    "primary-request.json": {"model": "default", "prompt": stem + primary_suffix, "max_tokens": 0, "temperature": 0},
}
for name, payload in requests.items():
    (out / name).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
PY
}

stop_server() {
  local pid=${server_pid:-}
  [[ $pid =~ ^[0-9]+$ ]] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  server_pid=
}

driver() {
  [[ $# == 2 ]] || exit 2
  local out=$1 port=$2 code
  write_requests "$out"
  mkdir "$out/kv"
  "$BIN" --cuda -m "$MODEL" -c 8192 --host 127.0.0.1 --port "$port" \
    --ssd-streaming --ssd-streaming-cache-experts 40GB \
    --kv-disk-dir "$out/kv" --kv-disk-space-mb 4096 \
    --kv-cache-boundary-align-tokens 4 --kv-cache-boundary-trim-tokens 8 \
    >"$out/server.log" 2>&1 &
  server_pid=$!
  trap stop_server EXIT INT TERM HUP
  for _ in $(seq 1 600); do
    kill -0 "$server_pid" 2>/dev/null || {
      echo "W7 unchanged engine died during startup" >&2
      return 1
    }
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:$port/v1/models" || true)
    [[ $code == 200 ]] && break
    sleep 2
  done
  [[ ${code:-} == 200 ]] || {
    echo "W7 unchanged engine did not become ready" >&2
    return 1
  }
  curl -sS --fail-with-body --max-time 900 -H 'Content-Type: application/json' \
    -o "$out/live-response.json" -w '%{http_code}\n' \
    -d @"$out/live-request.json" "http://127.0.0.1:$port/v1/completions" \
    >"$out/live-http-status"
  curl -sS --fail-with-body --max-time 900 -H 'Content-Type: application/json' \
    -o "$out/primary-response.json" -w '%{http_code}\n' \
    -d @"$out/primary-request.json" "http://127.0.0.1:$port/v1/completions" \
    >"$out/primary-http-status"
  stop_server
  trap - EXIT INT TERM HUP
}

score_red() {
  local out=$1
  /usr/bin/python3 -I -B - "$out" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
log = (out / "server.log").read_text(encoding="utf-8", errors="replace")
live = json.loads((out / "live-response.json").read_text(encoding="utf-8"))
primary = json.loads((out / "primary-response.json").read_text(encoding="utf-8"))
checks = {
    "live_http_200": (out / "live-http-status").read_text().strip() == "200",
    "primary_http_200": (out / "primary-http-status").read_text().strip() == "200",
    "live_prompt_tokens_5037": live.get("usage", {}).get("prompt_tokens") == 5037,
    "primary_prompt_tokens_5048": primary.get("usage", {}).get("prompt_tokens") == 5048,
    "live_miss_geometry": "live kv cache miss live=5037 prompt=5048 common=5029" in log,
    "selected_checkpoint_5028": re.search(r"kv cache hit text tokens=5028\\b", log) is not None,
    "strict_guard_cold_restart": "GLM resume guard: prompt (5048) extends/diverges past evaluated frontier 5037 (checkpoint 5028)" in log,
    "cold_sync_after_guard": re.search(r"GLM sync start=0 prompt=5048 suffix=5048\\b", log) is not None,
    "legacy_guard_bypass_absent": "DS4_GLM_RESUME_GUARD_OFF" not in log,
}
desired_resume_pass = (
    all(checks[name] for name in ("live_http_200", "primary_http_200", "live_prompt_tokens_5037", "primary_prompt_tokens_5048", "live_miss_geometry", "selected_checkpoint_5028", "legacy_guard_bypass_absent"))
    and not checks["strict_guard_cold_restart"]
    and re.search(r"GLM sync start=5028 prompt=5048 suffix=20\\b", log) is not None
)
red_confirmed = all(checks.values()) and not desired_resume_pass
summary = {
    "schema_version": 1,
    "gate": "W7-resume-bpe-lineage-v1",
    "classification": "fresh production-path reproduction",
    "geometry": {"selected": 5028, "common": 5029, "live": 5037, "prompt": 5048},
    "acceptance_formula": "HTTP/prompt/selection geometry exact; no strict cold restart; sync start=5028,prompt=5048,suffix=20",
    "checks": checks,
    "desired_resume_pass": desired_resume_pass,
    "red_confirmed": red_confirmed,
    "server_log_sha256": hashlib.sha256((out / "server.log").read_bytes()).hexdigest(),
    "verdict": "RED_CONFIRMED" if red_confirmed else ("PASS" if desired_resume_pass else "NO_RESULT"),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
raise SystemExit(1 if red_confirmed else (0 if desired_resume_pass else 2))
PY
}

if [[ ${1:-} == --self-test ]]; then
  [[ $(sha256sum -- "$POOL" | awk '{print $1}') == "$POOL_SHA256" ]]
  [[ $(sha256sum -- "$BIN" | awk '{print $1}') == "$BINARY_SHA256" ]]
  [[ $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]]
  fixture_check >/dev/null
  echo W7_RED_SELFTEST_OK
  exit 0
elif [[ ${1:-} == --driver ]]; then
  shift
  driver "$@"
  exit 0
fi

[[ $# == 0 && $(id -un) == bmarti44 ]] || {
  echo "usage: $0" >&2
  exit 2
}
[[ -x $BIN && -r $MODEL && -r $STEM && -r $POOL && -r $SAFE && -x $CGROUP ]] || exit 2
[[ $(sha256sum -- "$BIN" | awk '{print $1}') == "$BINARY_SHA256" ]] || exit 2
[[ $(sha256sum -- "$POOL" | awk '{print $1}') == "$POOL_SHA256" ]] || exit 2
[[ $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]] || exit 2
[[ -z $(git -C "$REPO" status --porcelain) ]] || {
  echo "W7 RED requires a clean committed harness" >&2
  exit 2
}
! pgrep -x ds4-server >/dev/null && ! pgrep -x fio >/dev/null || exit 75
swap_used_kib=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
(( swap_used_kib < 1048576 )) || exit 8
/usr/bin/python3 -I -B "$MEMORY_GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 0 >/dev/null
fixture_check >/dev/null

mkdir -p "$OUT_PARENT"
nonce=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
readonly nonce
readonly out=$OUT_PARENT/attempt-$nonce
readonly tag=w7red-${nonce:0:12}
mkdir "$out"
printf '%s\n' "$BINARY_SHA256" >"$out/binary.sha256"
printf '%s\n' "$MODEL_SHA256" >"$out/model-known.sha256"
printf '%s\n' "$POOL_SHA256" >"$out/fixture-pool.sha256"

set +e
/usr/bin/env -i \
  HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_RUNTIME_DIR=/run/user/1000 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  GLM_SAFE_RUN_AS_CURRENT_USER=1 GLM_SAFE_MEMORY_HIGH_GIB=78 \
  GLM_SAFE_KILL_FLOOR_GIB=24 GLM_SAFE_MIN_START_GIB=110 GLM_SAFE_TIMEOUT_S=2400 \
  GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 GLM_SAFE_EXPECTED_BINARY_SHA256=$BINARY_SHA256 \
  GLM_CANDIDATE_SRC=$CANDIDATE_SRC \
  DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GIB DS4_CUDA_EXPERT_CACHE_PIN=1 \
  DS4_CUDA_EXPERT_CACHE_SLRU=1 DS4_CUDA_FETCH_THREADS=6 \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_GLM_SYNC_TRACE=1 \
    "$CGROUP" --tag "$tag" -- /usr/bin/bash "$SCRIPT" --driver "$out" "$PORT" \
    >"$out/containment.stdout" 2>"$out/containment.stderr"
containment_rc=$?
set -e
printf '%s\n' "$containment_rc" >"$out/containment.rc"
(( containment_rc == 0 )) || {
  echo "W7 RED containment failed rc=$containment_rc evidence=$out" >&2
  exit 2
}

set +e
score_red "$out"
score_rc=$?
set -e
echo "W7 RED evidence: $out"
exit "$score_rc"
