#!/bin/bash
# Contained production-path eligibility probe for the default-off W3 direct
# expert-slot path. This is not the W3 performance campaign: it proves that
# the frozen production server reaches the path and remains byte-identical.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly MEMORY_GUARD=$REPO/scripts/03_memory_guard.py
readonly CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w3-0d855b2
readonly BIN=$CANDIDATE_SRC/ds4-server
readonly BINARY_SHA256=2de667d928f376d69f07ca252a9890e899ec425567633e742e505dff916c94cb
readonly ENGINE_COMMIT=0d855b2dc3067bc15f907ecfb022d0c18ec37185
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly MODEL_BYTES=211075856448
readonly CACHE_GIB=68
readonly CTX=8192
readonly TOKENS=64
readonly STATE_PARENT=/home/bmarti44/.local/state
readonly CRASH_ROOT=$STATE_PARENT/glm52-crashlog
readonly ENV_NAMES=DS4_CUDA_EXPERT_CACHE_GB,DS4_CUDA_EXPERT_CACHE_PIN,DS4_CUDA_EXPERT_CACHE_SLRU,DS4_CUDA_FETCH_THREADS,DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS,DS4_CUDA_MOE_NO_ATOMIC_DOWN,DS4_GLM_TP_DEBUG

[[ $(id -un) == bmarti44 ]] || {
  echo "W3 probe must run as bmarti44" >&2
  exit 2
}
[[ -x $BIN && -x $CGROUP && -r $MODEL ]] || {
  echo "W3 probe inputs are unavailable" >&2
  exit 2
}
[[ $(sha256sum -- "$BIN" | awk '{print $1}') == "$BINARY_SHA256" ]] || {
  echo "frozen W3 binary digest mismatch" >&2
  exit 2
}
[[ $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]] || {
  echo "GLM model size mismatch" >&2
  exit 2
}
if pgrep -x ds4-server >/dev/null || pgrep -x fio >/dev/null; then
  echo "exclusive W3 probe preflight found an active engine or fio" >&2
  exit 75
fi
swap_used_kib=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
(( swap_used_kib < 1048576 )) || {
  echo "W3 probe refuses to start with >=1 GiB swap in use" >&2
  exit 8
}
python3 "$MEMORY_GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 0

OUT=$(mktemp -d "$STATE_PARENT/glm52-w3-direct-slot-probe-v1.XXXXXX")
readonly OUT
printf '%s\n' "$OUT" >"$OUT/output-directory.txt"
python3 - "$OUT/request.json" <<'PY'
import json
import sys

request = {
    "model": "glm-5.2",
    "messages": [{
        "role": "user",
        "content": (
            "Generate a deterministic sequence of exactly 200 lowercase letters "
            "by repeating the alphabet in order. Do not stop early."
        ),
    }],
    "max_tokens": 64,
    "temperature": 0,
    "seed": 424242,
}
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump(request, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
readonly REQUEST_SHA256=$(sha256sum -- "$OUT/request.json" | awk '{print $1}')

active_pid=
cleanup() {
  if [[ -n ${active_pid:-} && -r /proc/$active_pid/exe &&
        $(readlink -f -- "/proc/$active_pid/exe" 2>/dev/null || true) == "$BIN" ]]; then
    kill -TERM "$active_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

environment_sha256() {
  python3 - "$ENV_NAMES" <<'PY'
import hashlib
import os
import sys

names = sys.argv[1].split(",")
canonical = b"".join(
    name.encode("ascii") + b"=" +
    os.environ.get(name, "<UNSET>").encode("ascii") + b"\n"
    for name in names
)
print(hashlib.sha256(canonical).hexdigest())
PY
}

wait_for_exact_engine() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    mapfile -t candidates < <(pgrep -x ds4-server || true)
    if (( ${#candidates[@]} == 1 )); then
      local candidate=${candidates[0]}
      if [[ $(readlink -f -- "/proc/$candidate/exe" 2>/dev/null || true) == "$BIN" ]]; then
        active_pid=$candidate
        return 0
      fi
    elif (( ${#candidates[@]} > 1 )); then
      echo "multiple ds4-server processes appeared" >&2
      return 1
    fi
    sleep 1
  done
  echo "frozen W3 engine did not appear" >&2
  return 1
}

wait_for_ready() {
  local port=$1 deadline=$((SECONDS + 900)) code=
  while (( SECONDS < deadline )); do
    kill -0 "$active_pid" 2>/dev/null || return 1
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:$port/v1/models" || true)
    [[ $code == 200 ]] && return 0
    sleep 2
  done
  return 1
}

run_arm() {
  local arm=$1 port=$2 direct=$3 tag="w3slot-${arm}-v1" arm_dir="$OUT/$arm"
  mkdir "$arm_dir"
  python3 "$MEMORY_GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 900 \
    >"$arm_dir/memory-preflight.json"
  if curl -sS -o /dev/null --max-time 2 "http://127.0.0.1:$port/v1/models"; then
    echo "probe port $port is already occupied" >&2
    return 1
  fi

  export DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GIB
  export DS4_CUDA_EXPERT_CACHE_PIN=1
  export DS4_CUDA_EXPERT_CACHE_SLRU=1
  export DS4_CUDA_FETCH_THREADS=6
  export DS4_CUDA_MOE_NO_ATOMIC_DOWN=1
  export DS4_GLM_TP_DEBUG=1
  if [[ $direct == 1 ]]; then
    export DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS=1
  else
    unset DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS
  fi
  local env_sha
  env_sha=$(environment_sha256)

  set +e
  GLM_SAFE_RUN_AS_CURRENT_USER=1 \
  GLM_SAFE_MEMORY_HIGH_GIB=95 \
  GLM_SAFE_KILL_FLOOR_GIB=18 \
  GLM_SAFE_MIN_START_GIB=110 \
  GLM_SAFE_TIMEOUT_S=1800 \
  GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1 \
  GLM_SAFE_EXPECTED_BINARY_SHA256=$BINARY_SHA256 \
  GLM_CANDIDATE_SRC=$CANDIDATE_SRC \
  GLM_SAFE_PROVENANCE_ENV_ALLOWLIST=$ENV_NAMES \
  GLM_SAFE_EXPECTED_ENV_SHA256=$env_sha \
    "$CGROUP" --tag "$tag" -- \
      "$BIN" --cuda -m "$MODEL" -c "$CTX" \
      --host 127.0.0.1 --port "$port" \
      --ssd-streaming --ssd-streaming-cache-experts 40GB \
      >"$arm_dir/containment.stdout" 2>"$arm_dir/containment.stderr" &
  local runner_pid=$!
  set -e

  if ! wait_for_exact_engine || ! wait_for_ready "$port"; then
    cleanup
    wait "$runner_pid" || true
    echo "W3 $arm arm did not become ready" >&2
    return 1
  fi

  local warm_meta measured_meta
  warm_meta=$(curl -sS -o "$arm_dir/warm.json" \
    -w '%{http_code} %{time_total}' --max-time 900 \
    -H 'Content-Type: application/json' -d @"$OUT/request.json" \
    "http://127.0.0.1:$port/v1/chat/completions")
  measured_meta=$(curl -sS -o "$arm_dir/measured.json" \
    -w '%{http_code} %{time_total}' --max-time 900 \
    -H 'Content-Type: application/json' -d @"$OUT/request.json" \
    "http://127.0.0.1:$port/v1/chat/completions")
  printf '%s\n' "$warm_meta" >"$arm_dir/warm.http"
  printf '%s\n' "$measured_meta" >"$arm_dir/measured.http"

  kill -TERM "$active_pid"
  active_pid=
  set +e
  wait "$runner_pid"
  local safe_rc=$?
  set -e

  local crash_dir
  crash_dir=$(find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name "*-$tag" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n $crash_dir && -f $crash_dir/main.log && -f $crash_dir/cmd.log ]] || {
    echo "W3 $arm arm has no complete safe-run evidence" >&2
    return 1
  }

  python3 - "$arm_dir/arm.json" "$arm" "$direct" "$safe_rc" \
      "$env_sha" "$crash_dir" "$REQUEST_SHA256" "$BINARY_SHA256" \
      "$MODEL_SHA256" "$ENGINE_COMMIT" "$arm_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

(out_path, arm, direct, safe_rc, env_sha, crash_dir, request_sha,
 binary_sha, model_sha, engine_commit, arm_dir) = sys.argv[1:]
arm_path = Path(arm_dir)
crash = Path(crash_dir)
cmd = (crash / "cmd.log").read_text(encoding="utf-8", errors="replace")
main = (crash / "main.log").read_text(encoding="utf-8", errors="replace")
kernel = (crash / "kernel.log").read_text(encoding="utf-8", errors="replace")
warm_http = (arm_path / "warm.http").read_text().split()
measured_http = (arm_path / "measured.http").read_text().split()
payload = json.loads((arm_path / "measured.json").read_text(encoding="utf-8"))
choice = payload["choices"][0]
message = choice["message"]
generated = {
    "content": message.get("content", ""),
    "reasoning_content": message.get("reasoning_content", ""),
}
canonical = json.dumps(generated, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
fault_re = re.compile(r"FATAL|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|"
                      r"Out of memory|NVRM.*Xid", re.I)
record = {
    "schema_version": 1,
    "arm": arm,
    "direct_requested": direct == "1",
    "safe_returncode": int(safe_rc),
    "warm_http_code": int(warm_http[0]),
    "warm_wall_seconds": float(warm_http[1]),
    "measured_http_code": int(measured_http[0]),
    "measured_wall_seconds": float(measured_http[1]),
    "completion_tokens": payload["usage"]["completion_tokens"],
    "generated_sha256": hashlib.sha256(canonical).hexdigest(),
    "generated_bytes": len(canonical),
    "mapping_markers": cmd.count("direct expert-slot arena mapping enabled"),
    "direct_hit_markers": cmd.count("direct expert-slot hit layer="),
    "clean_exit_attestation": (
        "verified alive at least once; no identity contradiction observed" in main
    ),
    "fault_markers": len(fault_re.findall(cmd + "\n" + main + "\n" + kernel)),
    "environment_sha256": env_sha,
    "request_sha256": request_sha,
    "binary_sha256": binary_sha,
    "model_sha256": model_sha,
    "engine_commit": engine_commit,
    "crash_evidence": str(crash),
    "crash_artifact_sha256": {
        name: hashlib.sha256((crash / name).read_bytes()).hexdigest()
        for name in ("main.log", "cmd.log", "samples.log", "kernel.log")
    },
}
Path(out_path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
PY
}

run_arm off 18163 0
run_arm on 18164 1

python3 - "$OUT" "$REQUEST_SHA256" "$BINARY_SHA256" "$MODEL_SHA256" \
    "$ENGINE_COMMIT" "$0" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
request_sha, binary_sha, model_sha, engine_commit, harness_path = sys.argv[2:]
arms = {name: json.loads((out / name / "arm.json").read_text())
        for name in ("off", "on")}
responses = {name: json.loads((out / name / "measured.json").read_text())
             for name in ("off", "on")}

def generated(payload):
    message = payload["choices"][0]["message"]
    return {
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
    }

off_generated = generated(responses["off"])
on_generated = generated(responses["on"])
checks = {
    "same_frozen_binary": all(a["binary_sha256"] == binary_sha for a in arms.values()),
    "same_model": all(a["model_sha256"] == model_sha for a in arms.values()),
    "same_request": all(a["request_sha256"] == request_sha for a in arms.values()),
    "safe_returncodes_zero": all(a["safe_returncode"] == 0 for a in arms.values()),
    "http_200": all(a["warm_http_code"] == 200 and a["measured_http_code"] == 200
                    for a in arms.values()),
    "full_64_token_measured_outputs": all(a["completion_tokens"] >= 64
                                           for a in arms.values()),
    "generated_output_nonempty": bool(
        off_generated["content"] or off_generated["reasoning_content"]
    ),
    "generated_output_byte_identical": off_generated == on_generated,
    "off_path_not_mapped": arms["off"]["mapping_markers"] == 0,
    "off_path_has_no_direct_hits": arms["off"]["direct_hit_markers"] == 0,
    "on_path_mapped": arms["on"]["mapping_markers"] >= 1,
    "on_path_executed": arms["on"]["direct_hit_markers"] >= 1,
    "clean_exit_attested": all(a["clean_exit_attestation"] for a in arms.values()),
    "no_fault_markers": all(a["fault_markers"] == 0 for a in arms.values()),
}
passed = all(checks.values())
summary = {
    "schema_version": 1,
    "gate": "W3-production-direct-slot-contained-probe-v1",
    "status": "PASS" if passed else "FAIL",
    "scope": "runtime eligibility only; no W3 completed-time performance credit",
    "acceptance_formula": "PASS iff every named boolean check is true",
    "checks": checks,
    "engine_commit": engine_commit,
    "binary_sha256": binary_sha,
    "model_sha256": model_sha,
    "request_sha256": request_sha,
    "arms": arms,
}
(out / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
manifest = {
    "schema_version": 1,
    "engine_commit": engine_commit,
    "binary_sha256": binary_sha,
    "model_sha256": model_sha,
    "request_sha256": request_sha,
    "harness_sha256": hashlib.sha256(Path(harness_path).read_bytes()).hexdigest(),
    "artifact_sha256": {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"
    },
}
(out / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"output": str(out), "status": summary["status"],
                  "checks": checks}, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY

trap - EXIT INT TERM HUP
echo "W3_DIRECT_SLOT_PROBE_DONE output=$OUT"
