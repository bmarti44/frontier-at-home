#!/bin/bash
# Contained production-path eligibility probe for the default-off W3 direct
# expert-slot path. This is not the W3 performance campaign: it proves that
# the frozen production server reaches the path and remains byte-identical.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly CGROUP=$REPO/results/glm52-gates/harness/glm_cgroup_run.sh
readonly SAFE=$REPO/results/glm52-gates/harness/glm_safe_run.sh
readonly MEMORY_GUARD=$REPO/scripts/03_memory_guard.py
readonly CANDIDATE_SRC=/home/bmarti44/.cache/glm52-w3-cc5e674
readonly BIN=$CANDIDATE_SRC/ds4-server
readonly BINARY_SHA256=e779b83a50da0c820f2eef8ebddb566c2c367d31025042235282af9d4817ea13
readonly ENGINE_COMMIT=cc5e6744718271a151593dc380c0e396229ecfc2
readonly ENGINE_SOURCE=/tmp/glm52-slot-prod-v1
readonly MODEL=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
readonly MODEL_SHA256=a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0
readonly MODEL_BYTES=211075856448
readonly TOKENIZER=/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json
readonly TOKENIZERS_INIT=/home/bmarti44/.local/lib/python3.12/site-packages/tokenizers/__init__.py
readonly TOKENIZERS_SO=/home/bmarti44/.local/lib/python3.12/site-packages/tokenizers/tokenizers.abi3.so
readonly CACHE_GIB=68
readonly CTX=8192
readonly TOKENS=64
readonly STATE_PARENT=/home/bmarti44/.local/state
readonly CRASH_ROOT=$STATE_PARENT/glm52-crashlog
readonly ENV_NAMES=DS4_CUDA_EXPERT_CACHE_GB,DS4_CUDA_EXPERT_CACHE_PIN,DS4_CUDA_EXPERT_CACHE_SLRU,DS4_CUDA_FETCH_THREADS,DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS,DS4_CUDA_MOE_NO_ATOMIC_DOWN,DS4_GLM_TP_DEBUG

[[ $# == 2 ]] || {
  echo "usage: $0 FREEZE_JSON DRAND_JSON" >&2
  exit 2
}
readonly FREEZE=$1
readonly DRAND=$2

[[ $(id -un) == bmarti44 ]] || {
  echo "W3 probe must run as bmarti44" >&2
  exit 2
}
[[ -x $BIN && -x $CGROUP && -x $SAFE && -r $MODEL && -r $TOKENIZER &&
   -r $TOKENIZERS_INIT && -r $TOKENIZERS_SO && -r $FREEZE && -r $DRAND ]] || {
  echo "W3 probe inputs are unavailable" >&2
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

# Git is the authority root for this bounded evidence harness. The committed
# freeze manifest binds the harness, transitive safety helpers, compiled engine
# source and binary. No self-reported digest can substitute for these checks.
python3 - "$REPO" "$FREEZE" "$0" "$CGROUP" "$SAFE" "$MEMORY_GUARD" \
    "$BIN" "$TOKENIZER" "$TOKENIZERS_INIT" "$TOKENIZERS_SO" \
    "$ENGINE_SOURCE" "$ENGINE_COMMIT" "$BINARY_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

(repo_raw, freeze_raw, harness_raw, cgroup_raw, safe_raw, guard_raw,
 binary_raw, tokenizer_raw, tokenizers_init_raw, tokenizers_so_raw,
 source_raw, engine_commit, binary_sha) = sys.argv[1:]
repo = Path(repo_raw).resolve()
freeze = Path(freeze_raw).resolve()
paths = {
    "harness": Path(harness_raw).resolve(),
    "cgroup": Path(cgroup_raw).resolve(),
    "safe": Path(safe_raw).resolve(),
    "memory_guard": Path(guard_raw).resolve(),
    "binary": Path(binary_raw).resolve(),
    "tokenizer": Path(tokenizer_raw).resolve(),
    "tokenizers_init": Path(tokenizers_init_raw).resolve(),
    "tokenizers_so": Path(tokenizers_so_raw).resolve(),
}
source = Path(source_raw).resolve()

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

if subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True,
                  capture_output=True, check=True).stdout:
    raise SystemExit("repository is not clean at probe execution")
relative = freeze.relative_to(repo)
committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo,
                           capture_output=True, check=True).stdout
if committed != freeze.read_bytes():
    raise SystemExit("freeze manifest differs from HEAD")
record = json.loads(committed)
required = {"schema_version", "repository_parent_commit", "engine_commit",
            "binary_sha256", "drand_floor_round", "artifacts",
            "engine_source_sha256"}
if set(record) != required or record["schema_version"] != 1:
    raise SystemExit("freeze manifest schema is invalid")
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                      capture_output=True, check=True).stdout.strip()
parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=repo, text=True,
                        capture_output=True, check=True).stdout.strip()
if parent != record["repository_parent_commit"]:
    raise SystemExit("HEAD is not the exact reviewed freeze commit")
changed = subprocess.run(["git", "diff", "--name-only", "HEAD^", "HEAD"],
                         cwd=repo, text=True, capture_output=True,
                         check=True).stdout.splitlines()
if changed != [str(relative)]:
    raise SystemExit("freeze commit changed files other than its receipt")
if record["engine_commit"] != engine_commit or record["binary_sha256"] != binary_sha:
    raise SystemExit("freeze candidate identity is inconsistent")
if set(record["artifacts"]) != set(paths):
    raise SystemExit("freeze artifact inventory is incomplete")
for name, path in paths.items():
    if digest(path) != record["artifacts"][name]:
        raise SystemExit(f"frozen artifact changed: {name}")
observed_engine = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                                 text=True, capture_output=True,
                                 check=True).stdout.strip()
tracked_dirty = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source,
    text=True, capture_output=True, check=True).stdout
if observed_engine != engine_commit or tracked_dirty:
    raise SystemExit("compiled engine source lineage changed")
for relative_path, expected in record["engine_source_sha256"].items():
    if digest(source / relative_path) != expected:
        raise SystemExit(f"compiled engine source changed: {relative_path}")
PY

readonly OBSERVED_REPOSITORY_HEAD=$(git -C "$REPO" rev-parse HEAD)
readonly OBSERVED_FREEZE_SHA256=$(sha256sum -- "$FREEZE" | awk '{print $1}')
readonly OBSERVED_TOKENIZER_SHA256=$(sha256sum -- "$TOKENIZER" | awk '{print $1}')

readonly MODEL_IDENTITY_BEFORE=$(stat -Lc '%d:%i:%s:%Y:%Z' -- "$MODEL")
[[ $(stat -Lc '%s' -- "$MODEL") == "$MODEL_BYTES" ]] || {
  echo "GLM model size mismatch" >&2
  exit 2
}
readonly OBSERVED_MODEL_SHA256=$(sha256sum -- "$MODEL" | awk '{print $1}')
[[ $OBSERVED_MODEL_SHA256 == "$MODEL_SHA256" ]] || {
  echo "GLM model digest mismatch" >&2
  exit 2
}

# Authenticate a beacon obtained after the committed freeze against three
# registered public relays. The request nonce and seed are derived from it.
read -r DRAND_ROUND DRAND_RANDOMNESS DRAND_SIGNATURE DRAND_FLOOR < <(
  python3 - "$FREEZE" "$DRAND" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

freeze = json.loads(Path(sys.argv[1]).read_text())
beacon = json.loads(Path(sys.argv[2]).read_text())
if set(beacon) != {"round", "randomness", "signature"}:
    raise SystemExit("drand receipt schema is invalid")
round_number = beacon["round"]
randomness = beacon["randomness"]
signature = beacon["signature"]
if (not isinstance(round_number, int) or round_number <= freeze["drand_floor_round"] or
        not isinstance(randomness, str) or not re.fullmatch(r"[0-9a-f]{64}", randomness) or
        not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{192}", signature) or
        hashlib.sha256(bytes.fromhex(signature)).hexdigest() != randomness):
    raise SystemExit("drand receipt is invalid or predates the freeze")
expected = {"round": round_number, "randomness": randomness, "signature": signature}
for host in ("api.drand.sh", "api2.drand.sh", "api3.drand.sh"):
    response = subprocess.run([
        "/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail",
        "--max-time", "15", "--proto", "=https",
        f"https://{host}/public/{round_number}",
    ], capture_output=True, check=True,
       env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    published = json.loads(response.stdout)
    if any(published.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"drand relay disagreement: {host}")
print(round_number, randomness, signature, freeze["drand_floor_round"])
PY
)
readonly DRAND_ROUND DRAND_RANDOMNESS DRAND_SIGNATURE DRAND_FLOOR

OUT=$(mktemp -d "$STATE_PARENT/glm52-w3-direct-slot-probe-v3.XXXXXX")
readonly OUT
printf '%s\n' "$OUT" >"$OUT/output-directory.txt"
python3 - "$OUT/request.json" "$DRAND_RANDOMNESS" <<'PY'
import json
import sys

randomness = sys.argv[2]
request = {
    "model": "glm-5.2",
    "messages": [{
        "role": "user",
        "content": (
            "Generate a deterministic sequence of exactly 200 lowercase letters "
            "by repeating the alphabet in order. Do not stop early. "
            f"Confirmation nonce: {randomness[:24]}."
        ),
    }],
    "max_tokens": 64,
    "temperature": 0,
    "seed": int(randomness[:16], 16) % 2147483647,
    "thinking_enabled": False,
}
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump(request, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
readonly REQUEST_SHA256=$(sha256sum -- "$OUT/request.json" | awk '{print $1}')

active_pid=
runner_pid=
cleanup() {
  if [[ -n ${active_pid:-} && -r /proc/$active_pid/exe &&
        $(readlink -f -- "/proc/$active_pid/exe" 2>/dev/null || true) == "$BIN" ]]; then
    kill -TERM "$active_pid" 2>/dev/null || true
  fi
  if [[ -n ${runner_pid:-} ]] && kill -0 "$runner_pid" 2>/dev/null; then
    kill -TERM "$runner_pid" 2>/dev/null || true
    for _ in $(seq 1 180); do
      kill -0 "$runner_pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -0 "$runner_pid" 2>/dev/null && kill -KILL "$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
  fi
  active_pid=
  runner_pid=
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  cleanup
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

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
  local arm=$1 port=$2 direct=$3 tag="w3s${DRAND_ROUND}-${arm}" arm_dir="$OUT/$arm"
  mkdir "$arm_dir"
  python3 "$MEMORY_GUARD" --required-gib 110 --stable-samples 3 --timeout-seconds 900 \
    >"$arm_dir/memory-preflight.json"
  if curl -sS -o /dev/null --max-time 2 "http://127.0.0.1:$port/v1/models"; then
    echo "probe port $port is already occupied" >&2
    return 1
  fi
  if find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*-$tag" \
      -print -quit | grep -q .; then
    echo "unique W3 evidence tag was already consumed: $tag" >&2
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

  local -a direct_environment=()
  [[ $direct == 1 ]] && direct_environment=(DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS=1)
  set +e
  /usr/bin/env -i \
  HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  XDG_RUNTIME_DIR=/run/user/1000 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
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
  DS4_CUDA_EXPERT_CACHE_GB=$CACHE_GIB \
  DS4_CUDA_EXPERT_CACHE_PIN=1 \
  DS4_CUDA_EXPERT_CACHE_SLRU=1 \
  DS4_CUDA_FETCH_THREADS=6 \
  DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
  DS4_GLM_TP_DEBUG=1 \
  "${direct_environment[@]}" \
    "$CGROUP" --tag "$tag" -- \
      "$BIN" --cuda -m "$MODEL" -c "$CTX" \
      --host 127.0.0.1 --port "$port" \
      --ssd-streaming --ssd-streaming-cache-experts 40GB \
      >"$arm_dir/containment.stdout" 2>"$arm_dir/containment.stderr" &
  runner_pid=$!
  set -e

  if ! wait_for_exact_engine || ! wait_for_ready "$port"; then
    cleanup
    wait "$runner_pid" || true
    echo "W3 $arm arm did not become ready" >&2
    return 1
  fi

  local crash_dir crash_identity
  mapfile -t crash_matches < <(find "$CRASH_ROOT" -mindepth 1 -maxdepth 1 \
    -type d -name "*-$tag" -print)
  (( ${#crash_matches[@]} == 1 )) || {
    echo "W3 $arm arm did not create exactly one evidence directory" >&2
    return 1
  }
  crash_dir=${crash_matches[0]}
  crash_identity=$(stat -Lc '%d:%i' -- "$crash_dir")
  [[ -f $crash_dir/cmd.log && -f $crash_dir/main.log ]] || {
    echo "W3 $arm arm evidence logs are absent after readiness" >&2
    return 1
  }

  local warm_meta measured_meta warm_dispatch_before warm_dispatch_after
  local measured_dispatch_before measured_dispatch_after
  warm_dispatch_before=$(grep -c 'direct expert-slot dispatch layer=' \
    "$crash_dir/cmd.log" || true)
  warm_meta=$(curl -sS -o "$arm_dir/warm.json" \
    -w '%{http_code} %{time_total}' --max-time 900 \
    -H 'Content-Type: application/json' -d @"$OUT/request.json" \
    "http://127.0.0.1:$port/v1/chat/completions")
  warm_dispatch_after=$(grep -c 'direct expert-slot dispatch layer=' \
    "$crash_dir/cmd.log" || true)
  measured_dispatch_before=$warm_dispatch_after
  measured_meta=$(curl -sS -o "$arm_dir/measured.json" \
    -w '%{http_code} %{time_total}' --max-time 900 \
    -H 'Content-Type: application/json' -d @"$OUT/request.json" \
    "http://127.0.0.1:$port/v1/chat/completions")
  measured_dispatch_after=$(grep -c 'direct expert-slot dispatch layer=' \
    "$crash_dir/cmd.log" || true)
  printf '%s\n' "$warm_meta" >"$arm_dir/warm.http"
  printf '%s\n' "$measured_meta" >"$arm_dir/measured.http"
  printf '%s %s\n' "$warm_dispatch_before" "$warm_dispatch_after" \
    >"$arm_dir/warm.dispatch-counts"
  printf '%s %s\n' "$measured_dispatch_before" "$measured_dispatch_after" \
    >"$arm_dir/measured.dispatch-counts"

  kill -TERM "$active_pid"
  active_pid=
  set +e
  wait "$runner_pid"
  local safe_rc=$?
  set -e
  runner_pid=

  [[ $(stat -Lc '%d:%i' -- "$crash_dir") == "$crash_identity" &&
      -f $crash_dir/main.log && -f $crash_dir/cmd.log ]] || {
    echo "W3 $arm arm has no complete safe-run evidence" >&2
    return 1
  }
  local receipt_dir
  receipt_dir=$(sed -n 's/^SAFE_RUN_DONE rc=[^ ]* killed=[^ ]* dir=//p' \
    "$arm_dir/containment.stdout")
  [[ $receipt_dir == "$crash_dir" ]] || {
    echo "W3 $arm arm receipt does not bind its evidence directory" >&2
    return 1
  }

  python3 - "$arm_dir/arm.json" "$arm" "$direct" "$safe_rc" \
      "$env_sha" "$crash_dir" "$REQUEST_SHA256" "$BINARY_SHA256" \
      "$OBSERVED_MODEL_SHA256" "$ENGINE_COMMIT" "$arm_dir" \
      "$crash_identity" "$TOKENIZER" "$OBSERVED_TOKENIZER_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

(out_path, arm, direct, safe_rc, env_sha, crash_dir, request_sha,
 binary_sha, model_sha, engine_commit, arm_dir, crash_identity, tokenizer_path,
 tokenizer_sha) = sys.argv[1:]
from tokenizers import Tokenizer
arm_path = Path(arm_dir)
crash = Path(crash_dir)
cmd = (crash / "cmd.log").read_text(encoding="utf-8", errors="replace")
main = (crash / "main.log").read_text(encoding="utf-8", errors="replace")
kernel = (crash / "kernel.log").read_text(encoding="utf-8", errors="replace")
warm_http = (arm_path / "warm.http").read_text().split()
measured_http = (arm_path / "measured.http").read_text().split()
warm_counts = [int(value) for value in
               (arm_path / "warm.dispatch-counts").read_text().split()]
measured_counts = [int(value) for value in
                   (arm_path / "measured.dispatch-counts").read_text().split()]
warm_payload = json.loads((arm_path / "warm.json").read_text(encoding="utf-8"))
payload = json.loads((arm_path / "measured.json").read_text(encoding="utf-8"))
tokenizer = Tokenizer.from_file(tokenizer_path)
choice = payload["choices"][0]
message = choice["message"]
warm_message = warm_payload["choices"][0]["message"]
generated = {
    "content": message.get("content", ""),
    "reasoning_content": message.get("reasoning_content", ""),
}
canonical = json.dumps(generated, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
measured_text = message.get("content", "")
warm_text = warm_message.get("content", "")
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
    "warm_completion_tokens": warm_payload["usage"]["completion_tokens"],
    "independent_completion_tokens": len(
        tokenizer.encode(measured_text, add_special_tokens=False).ids
    ),
    "independent_warm_completion_tokens": len(
        tokenizer.encode(warm_text, add_special_tokens=False).ids
    ),
    "measured_reasoning_bytes": len(
        message.get("reasoning_content", "").encode("utf-8")
    ),
    "warm_reasoning_bytes": len(
        warm_message.get("reasoning_content", "").encode("utf-8")
    ),
    "generated_sha256": hashlib.sha256(canonical).hexdigest(),
    "generated_bytes": len(canonical),
    "mapping_markers": cmd.count("direct expert-slot arena mapping enabled"),
    "admission_markers": cmd.count("direct expert-slot hit layer="),
    "dispatch_markers": cmd.count("direct expert-slot dispatch layer="),
    "warm_dispatch_delta": warm_counts[1] - warm_counts[0],
    "measured_dispatch_delta": measured_counts[1] - measured_counts[0],
    "clean_exit_attestation": (
        "verified alive at least once; no identity contradiction observed" in main
    ),
    "fault_markers": len(fault_re.findall(cmd + "\n" + main + "\n" + kernel)),
    "environment_sha256": env_sha,
    "request_sha256": request_sha,
    "binary_sha256": binary_sha,
    "model_sha256": model_sha,
    "tokenizer_sha256": tokenizer_sha,
    "engine_commit": engine_commit,
    "crash_evidence": str(crash),
    "crash_evidence_identity": crash_identity,
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
[[ $(stat -Lc '%d:%i:%s:%Y:%Z' -- "$MODEL") == "$MODEL_IDENTITY_BEFORE" ]] || {
  echo "GLM model identity changed after OFF arm" >&2
  exit 2
}
run_arm on 18164 1
[[ $(stat -Lc '%d:%i:%s:%Y:%Z' -- "$MODEL") == "$MODEL_IDENTITY_BEFORE" ]] || {
  echo "GLM model identity changed after ON arm" >&2
  exit 2
}

python3 - "$OUT" "$REQUEST_SHA256" "$BINARY_SHA256" "$OBSERVED_MODEL_SHA256" \
    "$ENGINE_COMMIT" "$0" "$DRAND_ROUND" "$DRAND_RANDOMNESS" \
    "$DRAND_SIGNATURE" "$DRAND_FLOOR" "$FREEZE" \
    "$OBSERVED_REPOSITORY_HEAD" "$OBSERVED_FREEZE_SHA256" \
    "$OBSERVED_TOKENIZER_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
(request_sha, binary_sha, model_sha, engine_commit, harness_path, drand_round,
 drand_randomness, drand_signature, drand_floor, freeze_path, repository_head,
 freeze_sha, tokenizer_sha) = sys.argv[2:]
freeze = json.loads(Path(freeze_path).read_text())
arms = {name: json.loads((out / name / "arm.json").read_text())
        for name in ("off", "on")}
responses = {name: json.loads((out / name / "measured.json").read_text())
             for name in ("off", "on")}
warm_responses = {name: json.loads((out / name / "warm.json").read_text())
                  for name in ("off", "on")}

def generated(payload):
    message = payload["choices"][0]["message"]
    return {
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
    }

off_generated = generated(responses["off"])
on_generated = generated(responses["on"])
off_warm_generated = generated(warm_responses["off"])
on_warm_generated = generated(warm_responses["on"])
checks = {
    "same_frozen_binary": all(a["binary_sha256"] == binary_sha for a in arms.values()),
    "same_model": all(a["model_sha256"] == model_sha for a in arms.values()),
    "same_request": all(a["request_sha256"] == request_sha for a in arms.values()),
    "safe_returncodes_zero": all(a["safe_returncode"] == 0 for a in arms.values()),
    "http_200": all(a["warm_http_code"] == 200 and a["measured_http_code"] == 200
                    for a in arms.values()),
    "independent_exact_64_token_outputs": all(
        a["independent_completion_tokens"] == 64 and
        a["independent_warm_completion_tokens"] == 64
        for a in arms.values()
    ),
    "thinking_disabled_no_reasoning_channel": all(
        a["measured_reasoning_bytes"] == 0 and a["warm_reasoning_bytes"] == 0
        for a in arms.values()
    ),
    "all_generated_outputs_nonempty": all(
        generated(responses[name])["content"] and
        generated(warm_responses[name])["content"]
        for name in ("off", "on")
    ),
    "generated_output_byte_identical": off_generated == on_generated,
    "warm_generated_output_byte_identical": off_warm_generated == on_warm_generated,
    "off_path_not_mapped": arms["off"]["mapping_markers"] == 0,
    "off_path_has_no_direct_dispatches": arms["off"]["dispatch_markers"] == 0,
    "on_path_mapped": arms["on"]["mapping_markers"] >= 1,
    "on_path_dispatched_for_compared_warm_response":
        arms["on"]["warm_dispatch_delta"] >= 1,
    "on_path_dispatched_for_compared_measured_response":
        arms["on"]["measured_dispatch_delta"] >= 1,
    "clean_exit_attested": all(a["clean_exit_attestation"] for a in arms.values()),
    "no_fault_markers": all(a["fault_markers"] == 0 for a in arms.values()),
}
passed = all(checks.values())
summary = {
    "schema_version": 1,
    "gate": "W3-production-direct-slot-contained-probe-v3",
    "status": "PASS" if passed else "FAIL",
    "scope": "runtime eligibility only; no W3 completed-time performance credit",
    "acceptance_formula": "PASS iff every named boolean check is true",
    "checks": checks,
    "engine_commit": engine_commit,
    "binary_sha256": binary_sha,
    "model_sha256": model_sha,
    "tokenizer_sha256": tokenizer_sha,
    "repository_head": repository_head,
    "freeze_sha256": freeze_sha,
    "freeze_bindings": freeze,
    "environment_sha256": {
        name: arms[name]["environment_sha256"] for name in ("off", "on")
    },
    "request_sha256": request_sha,
    "public_randomness": {
        "round": int(drand_round),
        "randomness": drand_randomness,
        "signature": drand_signature,
        "freeze_floor_round": int(drand_floor),
    },
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
    "tokenizer_sha256": tokenizer_sha,
    "repository_head": repository_head,
    "freeze_sha256": freeze_sha,
    "freeze_bindings": freeze,
    "environment_sha256": {
        name: arms[name]["environment_sha256"] for name in ("off", "on")
    },
    "request_sha256": request_sha,
    "public_randomness_round": int(drand_round),
    "public_randomness": drand_randomness,
    "public_randomness_signature": drand_signature,
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
