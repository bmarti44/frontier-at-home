#!/usr/bin/env bash
# Detached worker: free display memory, run one guarded production baseline,
# stop the engine, verify kernel/resource safety, and restore the display.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly HOLD=/home/dsv4/.dsv4-start-hold
readonly HOLD_OVERRIDE=/home/dsv4/.glm52-headless-start-allow
readonly LAUNCHER=$REPO/scripts/21_serve_llamacpp.sh
readonly GUARD=$REPO/scripts/03_memory_guard.py
readonly FAULT_PATTERN='NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'

[[ $# == 4 ]] || { echo "usage: $0 OUT TAG SEED CANDIDATE_HASH" >&2; exit 2; }
OUT=$1
TAG=$2
SEED=$3
CANDIDATE_HASH=$4
(( EUID == 0 )) || { echo "headless worker must run as root" >&2; exit 2; }
[[ $OUT == /home/dsv4/ds4-project/headless-foundation-* && -d $OUT &&
        ! -L $OUT ]] || { echo "invalid evidence directory" >&2; exit 2; }
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || { echo "invalid tag" >&2; exit 2; }
[[ $SEED =~ ^[0-9]{1,10}$ ]] && (( 10#$SEED <= 4294967295 )) ||
    { echo "invalid seed" >&2; exit 2; }
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] ||
    { echo "invalid candidate hash" >&2; exit 2; }
[[ -e $HOLD && -f $HOLD && ! -L $HOLD ]] ||
    { echo "persistent maintenance hold is absent or unsafe" >&2; exit 2; }
[[ ! -e $HOLD_OVERRIDE ]] ||
    { echo "headless hold-override sentinel unexpectedly exists" >&2; exit 2; }
actual=$(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD
) || { echo "cannot resolve repository candidate" >&2; exit 2; }
[[ $actual == "$CANDIDATE_HASH" ]] ||
    { echo "candidate hash changed" >&2; exit 2; }
[[ -z $(
    /usr/bin/git -c safe.directory="$REPO" -C "$REPO" status --porcelain
) ]] || { echo "repository is not clean" >&2; exit 2; }

ENGINE_ACTIVE=false
DISPLAY_WAS_ACTIVE=false
KERNEL_CURSOR=
MEMWATCH_START_LINE=0
SWAP_START_KIB=0

dsv4_launcher() {
    /usr/sbin/runuser -u dsv4 -- env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
        DSV4_PORT=8013 \
        DSV4_START_HOLD_FILE="$HOLD_OVERRIDE" \
        DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
        DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json \
        DSV4_MEM_FLOOR_GIB=18 DSV4_WATCHDOG_FLOOR_GIB=18 \
        DSV4_UBATCH=512 DSV4_BATCH=2048 DSV4_UBATCH_LARGE=0 \
        CTX=8192 DSV4_PARALLEL=1 DSV4_NO_MMAP=1 \
        DSV4_SPEC_TYPE=ngram-map-k4v \
        "$LAUNCHER" "$@"
}

write_status() {
    local rc=$1 kernel_fault=$2 release_ok=$3 swap_end_kib=$4
    /usr/bin/python3 - "$OUT/status.json.tmp" "$rc" "$kernel_fault" \
        "$release_ok" "$SWAP_START_KIB" "$swap_end_kib" \
        "$CANDIDATE_HASH" "$TAG" <<'PY'
import json
import os
import sys

path, rc, kernel_fault, release_ok, swap_start, swap_end, candidate, tag = sys.argv[1:]
value = {
    "schema_version": 1,
    "tag": tag,
    "candidate_hash": candidate,
    "exit_status": int(rc),
    "kernel_fault": kernel_fault == "true",
    "memory_recovered": release_ok == "true",
    "swap_start_kib": int(swap_start),
    "swap_end_kib": int(swap_end),
    "swap_growth": int(swap_end) > int(swap_start),
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
    mv -f -- "$OUT/status.json.tmp" "$OUT/status.json"
}

write_evidence_triplet() {
    local rc=$1
    /usr/bin/python3 - "$OUT" "$rc" "$CANDIDATE_HASH" "$TAG" "$SEED" \
        "$REPO/scripts/55_headless_foundation_worker.sh" \
        /home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
        "$REPO/configs/build-manifests/llamacpp-fusion.json" \
        "$REPO/weights/unsloth-ud-q2_k_xl/manifest.json" \
        "$REPO/scripts/30_bench_speed.py" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
candidate, tag, seed = sys.argv[3:6]
worker, binary, build_manifest, model_manifest, scorer = map(
    pathlib.Path, sys.argv[6:]
)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def load(name):
    path = out / name
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

status = load("status.json")
admission = load("admission.json")
result = load("result.json")
configuration = {
    "batch": 2048,
    "ctx": 8192,
    "no_mmap": True,
    "parallel": 1,
    "port": 8013,
    "ubatch": 512,
    "watchdog_floor_gib": 18,
}
configuration_bytes = json.dumps(
    configuration, sort_keys=True, separators=(",", ":")
).encode()
fixture_sha256 = hashlib.sha256(
    f"{candidate}:{seed}:foundation-dsv4-headless".encode()
).hexdigest()
manifest = {
    "schema_version": 1,
    "gate": "foundation",
    "attempt_kind": "dsv4_headless_baseline",
    "candidate_hash": candidate,
    "tag": tag,
    "seed": int(seed),
    "source_sha256": digest(worker),
    "diff_sha256": hashlib.sha256(b"").hexdigest(),
    "binary_sha256": digest(binary),
    "scorer_sha256": digest(scorer),
    "model_sha256": digest(model_manifest),
    "tokenizer_sha256": digest(model_manifest),
    "fixture_sha256": fixture_sha256,
    "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
    "configuration": configuration,
}
raw = {
    "record_type": "headless_foundation_arm",
    "candidate_hash": candidate,
    "tag": tag,
    "seed": int(seed),
    "admission": admission,
    "benchmark": result,
    "status": status,
}
summary = {
    "formula_version": 1,
    "verdict": "FAIL" if rc else "NO_RESULT",
    "reason": (
        "headless DeepSeek arm failed before a valid baseline completed"
        if rc
        else "DeepSeek arm completed; matched GLM arm remains required"
    ),
    "admission": admission,
    "status": status,
}
for name, value in (
    ("manifest.json", manifest),
    ("raw.jsonl", raw),
    ("summary.json", summary),
):
    path = out / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
PY
}

cleanup() {
    local rc=$? kernel_fault=false release_ok=true swap_end_kib=0
    trap - EXIT
    set +e
    if "$ENGINE_ACTIVE"; then
        dsv4_launcher stop >>"$OUT/stop.log" 2>&1
        (( $? == 0 )) || rc=1
        ENGINE_ACTIVE=false
    fi
    /usr/bin/python3 "$GUARD" --required-gib 110 --stable-samples 3 \
        --interval-seconds 1 --timeout-seconds 180 \
        >>"$OUT/release.log" 2>&1
    if (( $? != 0 )); then release_ok=false; rc=1; fi
    if [[ -n $KERNEL_CURSOR ]]; then
        journalctl -k --after-cursor "$KERNEL_CURSOR" --no-pager \
            >"$OUT/kernel.log" 2>&1
        if grep -Eiq "$FAULT_PATTERN" "$OUT/kernel.log"; then
            kernel_fault=true
            rc=1
        fi
    else
        kernel_fault=true
        rc=1
    fi
    if [[ -r /home/dsv4/logs/memwatch-llamacpp.log ]]; then
        sed -n "$((MEMWATCH_START_LINE + 1)),\$p" \
            /home/dsv4/logs/memwatch-llamacpp.log >"$OUT/memwatch.log"
    fi
    if [[ -r /home/dsv4/logs/llamacpp-server.log ]]; then
        tail -n 500 /home/dsv4/logs/llamacpp-server.log >"$OUT/server.tail.log"
    fi
    swap_end_kib=$(awk '$1 == "SwapFree:" {free=$2}
        $1 == "SwapTotal:" {total=$2} END {print total-free}' /proc/meminfo)
    [[ -e $HOLD ]] || rc=1
    (( swap_end_kib <= SWAP_START_KIB )) || rc=1
    if "$DISPLAY_WAS_ACTIVE"; then
        systemctl start display-manager.service >>"$OUT/display.log" 2>&1
        (( $? == 0 )) || rc=1
    fi
    write_status "$rc" "$kernel_fault" "$release_ok" "$swap_end_kib" || rc=1
    write_evidence_triplet "$rc" || rc=1
    chmod -R a+rX "$OUT" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT

sleep 5
if journalctl -k -b --no-pager | grep -Eiq "$FAULT_PATTERN"; then
    echo "pre-existing kernel GPU/OOM fault on current boot" >&2
    exit 1
fi
KERNEL_CURSOR=$(
    journalctl -k -n 0 --show-cursor --no-pager |
        sed -n 's/^-- cursor: //p'
)
[[ -n $KERNEL_CURSOR ]] || { echo "cannot freeze kernel cursor" >&2; exit 1; }
SWAP_START_KIB=$(awk '$1 == "SwapFree:" {free=$2}
    $1 == "SwapTotal:" {total=$2} END {print total-free}' /proc/meminfo)
MEMWATCH_START_LINE=$(
    wc -l </home/dsv4/logs/memwatch-llamacpp.log 2>/dev/null || echo 0
)

if systemctl is-active --quiet display-manager.service; then
    DISPLAY_WAS_ACTIVE=true
    systemctl stop display-manager.service >>"$OUT/display.log" 2>&1
fi
/usr/bin/python3 "$GUARD" --required-gib 116 --stable-samples 3 \
    --interval-seconds 1 --timeout-seconds 180 >"$OUT/admission.json"
[[ -e $HOLD ]] || { echo "maintenance hold disappeared" >&2; exit 1; }

dsv4_launcher start >"$OUT/start.log" 2>&1
ENGINE_ACTIVE=true
dsv4_launcher status >"$OUT/engine-status.json"

/usr/sbin/runuser -u dsv4 -- env -i \
    HOME=/home/dsv4 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    "$REPO/.venv-harness/bin/python" "$REPO/scripts/30_bench_speed.py" \
    --base-url http://127.0.0.1:8013 \
    --out "$OUT/result.json" \
    --stack-label dsv4-production-headless \
    --model-id deepseek-v4-flash \
    --reps 2 --context-levels 0 --max-tokens 160 \
    --min-completion-tokens 128 --seed "$SEED" --ignore-eos-supported

dsv4_launcher stop >"$OUT/stop.log" 2>&1
ENGINE_ACTIVE=false
