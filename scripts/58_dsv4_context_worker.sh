#!/usr/bin/env bash
# Detached, watchdog-protected DeepSeek context qualification worker.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly LAUNCHER=$REPO/scripts/21_serve_llamacpp.sh
readonly PROBE=$REPO/scripts/57_dsv4_context_probe.py
readonly GUARD=$REPO/scripts/03_memory_guard.py
readonly FAULT_PATTERN='NV_ERR_NO_MEMORY|NVRM.*Xid|oom-kill|Out of memory: Killed process|Killed process .*total-vm'
readonly CAPS=(131072 262144 524288 1048576)
readonly TARGETS=(130000 260000 520000 1000000)
readonly SAFE_CTX=8192 # CTX=8192 is the proven production recovery profile.

[[ $# == 5 ]] || {
    echo "usage: $0 OUT TAG SEED_SHA256 CANDIDATE_HASH MODE" >&2
    exit 2
}
OUT=$1
TAG=$2
SEED_SHA256=$3
CANDIDATE_HASH=$4
MODE=$5
(( EUID == 0 )) || { echo "context worker must run as root" >&2; exit 2; }
[[ $OUT == /home/dsv4/ds4-project/dsv4-context-* && -d $OUT && ! -L $OUT ]] ||
    { echo "invalid evidence directory" >&2; exit 2; }
[[ $TAG =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]] || { echo "invalid tag" >&2; exit 2; }
[[ $SEED_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "invalid seed digest" >&2; exit 2; }
[[ $CANDIDATE_HASH =~ ^[0-9a-f]{40}$ ]] ||
    { echo "invalid candidate hash" >&2; exit 2; }
[[ $MODE == one-million || $MODE == graduated ]] ||
    { echo "mode must be one-million or graduated" >&2; exit 2; }

actual=$(/usr/bin/git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD)
[[ $actual == "$CANDIDATE_HASH" ]] || { echo "candidate hash changed" >&2; exit 2; }
[[ -z $(/usr/bin/git -c safe.directory="$REPO" -C "$REPO" status --porcelain) ]] ||
    { echo "repository is not clean" >&2; exit 2; }

ENGINE_ACTIVE=false
DISPLAY_WAS_ACTIVE=false
TELEMETRY_PID=
KERNEL_CURSOR=
ORIGINAL_RUNNING=false
RESTORE_ALLOWED=true

dsv4_launcher() {
    local context=$1 measured=$2
    shift 2
    # Normal/recovery contract: DSV4_MEM_FLOOR_GIB=18 and
    # DSV4_WATCHDOG_FLOOR_GIB=18. The exact 1M branch alone exports
    # DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB=15.
    local floor=18 qualification_floor=0
    if (( context == 1048576 && measured == 3 )); then
        floor=15
        qualification_floor=15
    fi
    /usr/sbin/runuser -u dsv4 -- env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
        DSV4_PORT=8013 \
        DSV4_SERVER_BINARY=/home/dsv4/llamacpp-project/src/llama.cpp-fusion/build/bin/llama-server \
        DSV4_BUILD_MANIFEST=$REPO/configs/build-manifests/llamacpp-fusion.json \
        DSV4_MEM_FLOOR_GIB="$floor" DSV4_WATCHDOG_FLOOR_GIB="$floor" \
        DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB="$qualification_floor" \
        DSV4_MEASURED_HEADLESS_OVERHEAD_GIB="$measured" \
        DSV4_UBATCH=512 DSV4_BATCH=2048 DSV4_UBATCH_LARGE=0 \
        CTX="$context" DSV4_PARALLEL=1 DSV4_NO_MMAP=1 \
        DSV4_SPEC_TYPE=none \
        "$LAUNCHER" "$@"
}

stop_telemetry() {
    [[ ${TELEMETRY_PID:-} =~ ^[0-9]+$ ]] || return 0
    kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=
}

restore_safe_profile() {
    "$ORIGINAL_RUNNING" || return 0
    systemctl start display-manager.service >>"$OUT/display.log" 2>&1 || return 1
    DISPLAY_WAS_ACTIVE=false
    "$RESTORE_ALLOWED" || {
        echo "safe-profile restore suppressed after a kernel/OOM/startup fault" \
            >>"$OUT/restore.log"
        return 1
    }
    /usr/bin/python3 "$GUARD" --required-gib 110 --stable-samples 3 \
        --interval-seconds 1 --timeout-seconds 180 >>"$OUT/restore.log" 2>&1 ||
        return 1
    dsv4_launcher "$SAFE_CTX" 0 start >>"$OUT/restore.log" 2>&1
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    stop_telemetry
    if "$ENGINE_ACTIVE"; then
        dsv4_launcher "$SAFE_CTX" 0 stop >>"$OUT/stop-cleanup.log" 2>&1
        ENGINE_ACTIVE=false
    fi
    if [[ -n $KERNEL_CURSOR ]]; then
        journalctl -k --after-cursor "$KERNEL_CURSOR" --no-pager \
            >"$OUT/kernel.log" 2>&1
        if grep -Eiq "$FAULT_PATTERN" "$OUT/kernel.log"; then
            RESTORE_ALLOWED=false
            rc=1
        fi
    fi
    if [[ -e /run/dsv4/llamacpp.start-failed ]]; then
        RESTORE_ALLOWED=false
        rc=1
    fi
    if "$DISPLAY_WAS_ACTIVE"; then
        if "$ORIGINAL_RUNNING"; then
            restore_safe_profile || rc=1
        else
            systemctl start display-manager.service >>"$OUT/display.log" 2>&1 || rc=1
            DISPLAY_WAS_ACTIVE=false
        fi
    fi
    /usr/bin/python3 - "$OUT" "$rc" "$CANDIDATE_HASH" "$SEED_SHA256" "$MODE" <<'PY'
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
candidate, seed, mode = sys.argv[3:]
stages = []
for path in sorted(out.glob("stage-*.json")):
    try:
        stages.append(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        stages.append({"pass": False, "error": f"invalid stage artifact: {path.name}"})
memory = []
path = out / "memory.jsonl"
if path.is_file():
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            memory.append(json.loads(line))
        except ValueError:
            memory.append({"invalid": line})
kernel = (out / "kernel.log").read_text(encoding="utf-8", errors="replace") \
    if (out / "kernel.log").is_file() else ""
summary = {
    "schema_version": 1,
    "gate": "dsv4_context",
    "candidate_hash": candidate,
    "seed_sha256": seed,
    "mode": mode,
    "stages_attempted": len(stages),
    "stage_passes": [stage.get("pass") is True for stage in stages],
    "minimum_available_memory_gib": min(
        (sample["available_gib"] for sample in memory
         if isinstance(sample, dict) and isinstance(sample.get("available_gib"), (int, float))),
        default=None,
    ),
    "kernel_fault": any(term in kernel.lower() for term in (
        "nv_err_no_memory", "xid", "oom-kill", "out of memory: killed process"
    )),
    "exit_status": rc,
}
summary["verdict"] = "PASS" if (
    rc == 0 and stages and all(summary["stage_passes"])
    and not summary["kernel_fault"]
) else "FAIL"
(out / "summary.json").write_text(
    json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
manifest = {
    "schema_version": 1,
    "candidate_hash": candidate,
    "seed_sha256": seed,
    "probe_sha256": hashlib.sha256(
        pathlib.Path("/home/bmarti44/spark-deepseek-v4-flash/scripts/57_dsv4_context_probe.py").read_bytes()
    ).hexdigest(),
    "worker_sha256": hashlib.sha256(
        pathlib.Path("/home/bmarti44/spark-deepseek-v4-flash/scripts/58_dsv4_context_worker.sh").read_bytes()
    ).hexdigest(),
}
(out / "manifest.json").write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
    chmod -R a+rX "$OUT" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT

KERNEL_CURSOR=$(
    journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p'
)
[[ -n $KERNEL_CURSOR ]] || { echo "cannot freeze kernel cursor" >&2; exit 1; }
if journalctl -k -b --no-pager | grep -Eiq "$FAULT_PATTERN"; then
    echo "pre-existing kernel GPU/OOM fault on current boot" >&2
    exit 1
fi

if dsv4_launcher "$SAFE_CTX" 0 status >/dev/null 2>&1; then
    ORIGINAL_RUNNING=true
    dsv4_launcher "$SAFE_CTX" 0 stop >"$OUT/original-stop.log" 2>&1
fi
if systemctl is-active --quiet display-manager.service; then
    DISPLAY_WAS_ACTIVE=true
    systemctl stop display-manager.service >>"$OUT/display.log" 2>&1
fi
/usr/bin/python3 "$GUARD" --required-gib 115.25 --stable-samples 3 \
    --interval-seconds 1 --timeout-seconds 180 >"$OUT/admission.json"

indices=(0 1 2 3)
[[ $MODE == graduated ]] || indices=(3)
for index in "${indices[@]}"; do
    cap=${CAPS[$index]}
    target=${TARGETS[$index]}
    dsv4_launcher "$cap" 3 start >"$OUT/start-$cap.log" 2>&1
    ENGINE_ACTIVE=true
    (
        trap 'exit 0' TERM INT
        while true; do
            awk -v now="$(date +%s.%N)" '
                $1 == "MemAvailable:" {
                    printf "{\"timestamp_seconds\":%s,\"available_gib\":%.6f}\n",
                           now, $2 / 1048576
                }' /proc/meminfo
            sleep 0.25
        done
    ) >>"$OUT/memory.jsonl" &
    TELEMETRY_PID=$!
    /usr/sbin/runuser -u dsv4 -- env -i \
        HOME=/home/dsv4 USER=dsv4 LOGNAME=dsv4 LANG=C.UTF-8 \
        PATH=/home/bmarti44/spark-deepseek-v4-flash/.venv-harness/bin:/usr/bin:/bin \
        "$REPO/.venv-harness/bin/python" "$PROBE" \
        --base-url http://127.0.0.1:8013 \
        --context-cap "$cap" --target-tokens "$target" \
        --seed-sha256 "$SEED_SHA256" --out "$OUT/stage-$cap.json"
    stop_telemetry
    dsv4_launcher "$cap" 3 stop >"$OUT/stop-$cap.log" 2>&1
    ENGINE_ACTIVE=false
    /usr/bin/python3 "$GUARD" --required-gib 110 --stable-samples 3 \
        --interval-seconds 1 --timeout-seconds 180 \
        >"$OUT/release-$cap.json"
done
