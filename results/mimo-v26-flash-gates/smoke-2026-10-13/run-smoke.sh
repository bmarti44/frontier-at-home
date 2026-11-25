#!/usr/bin/env bash
# M1 smoke + fidelity driver (DIAGNOSTIC; PREREG.md). Production qwen38fn is
# stopped through the switch first and restored at the end, also on failure.
# Start 1: probes + kit speed,toolcall; start 2: probes; then model_diff.
set -Eeuo pipefail
WT=/home/bmarti44/mimo26
PROD=/home/bmarti44/spark-deepseek-v4-flash
EV=$WT/results/mimo-v26-flash-gates/smoke-2026-10-13
PROFILE=mimo-v2.6-flash/cuda-spark-128g-262k
PORT=8019
HOST=$EV/host-worktree.json
RT=/home/bmarti44/.cache/mimo-v26-flash/native-runtime-001
PY=$RT/runtime/bin/python3.12
PACK=/home/bmarti44/models/mimo-v2.6-flash-serve/mimo-v2.6-flash
DRAFT=/home/bmarti44/models/mimo-v2.6-flash-draft/mimo-v2.6-flash-dflash
OFFICIAL=/home/bmarti44/models/mimo-v2.6-flash-mopd
SERVE=(bash "$WT/scripts/93_profile_serve.sh" --profile "$PROFILE" --host "$HOST" --port "$PORT")
SLUG=${PROFILE//\//-}
SERVER_LOG=$HOME/.frontier-at-home/logs/$SLUG.server.log
MEMWATCH_LOG=$HOME/.frontier-at-home/logs/$SLUG.memwatch.log
OBS=$EV/observe
RELEASE_ROOTS=(/home/bmarti44/models/qwen3.8-flash-next-nvfp4 /home/bmarti44/.cache/qwen38-flash-next
               "$PACK" "$DRAFT" "$OFFICIAL" /home/bmarti44/models/mimo-v2.6-flash-mopd-exl3-2.52-src
               /home/bmarti44/models/mimo-v2.6-flash-dflash-exl3-src "$RT")
UNITS=("mimo26-smoke-1-$$.scope" "mimo26-smoke-2-$$.scope" "mimo26-fidelity-$$.scope")
GIB=1048576
[[ ! -e $OBS ]] || { echo "evidence directory $OBS exists; use a fresh one for a rerun" >&2; exit 1; }
mkdir -p "$OBS"
cd "$WT"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$OBS/driver.log"; }
meminfo() { awk -v k="$1" '$1 == k":" {print $2}' /proc/meminfo; }

# Nothing may still be writing the trees this run reads.
if systemctl --user is-active -q mimo-download.scope; then
    echo "the MiMo download scope is still active; refusing" >&2
    exit 1
fi
if find "$OFFICIAL/.cache" /home/bmarti44/models/mimo-v2.6-flash-mopd-exl3-2.52-src/.cache \
        -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
    echo "an incomplete download remains; refusing" >&2
    exit 1
fi
python3 - "$WT/weights/mimo-v2.6-flash/manifest.json" "$OFFICIAL" <<'PY'
import json, os, sys
manifest, root = sys.argv[1:]
files = json.load(open(manifest))["artifacts"]["mopd-release"]["files"]
bad = [f["name"] for f in files
       if not os.path.isfile(os.path.join(root, f["name"]))
       or os.path.getsize(os.path.join(root, f["name"])) != f["bytes"]]
if bad:
    raise SystemExit(f"official tree does not match the manifest: {bad[:5]}")
print(f"official tree matches the manifest ({len(files)} files)")
PY

( set +e
  while :; do
      printf '%s ' "$(stamp)"
      awk '/^(MemFree|MemAvailable|SwapTotal|SwapFree|Cached|AnonPages|Mapped|Dirty):/{printf "%s=%s ", $1, $2}' /proc/meminfo
      printf '\n'
      sleep 1
  done >>"$OBS/meminfo.log" ) &
SAMPLER=$!
journalctl -k -f -o short-iso --since now >"$OBS/kernel-smoke.log" 2>&1 &
JOURNAL=$!
SERVER_OFFSET=$(( $(wc -l <"$SERVER_LOG" 2>/dev/null || echo 0) ))
MEMWATCH_OFFSET=$(( $(wc -l <"$MEMWATCH_LOG" 2>/dev/null || echo 0) ))

copy_server_log() {  # copy_server_log <dest>: the lines this run appended since the last copy
    local lines
    lines=$(( $(wc -l <"$SERVER_LOG" 2>/dev/null || echo 0) ))
    tail -n +"$((SERVER_OFFSET + 1))" "$SERVER_LOG" >"$1" 2>/dev/null || true
    SERVER_OFFSET=$lines
}

restore_production() {
    local attempt
    for attempt in 1 2; do
        for ((i = 0; i < 180; i++)); do
            (( $(meminfo MemAvailable) >= 110 * GIB )) && break
            sleep 1
        done
        log "restore attempt $attempt: MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
        if sudo -n "$PROD/scripts/52_engine_switch.sh" qwen38fn >>"$OBS/restore.log" 2>&1; then
            log "restore rc=0"
            return 0
        fi
        log "restore attempt $attempt failed"
        sleep 30
    done
    log "RESTORE_FAILED after 2 attempts"
    return 1
}
cleanup() {
    local rc=$?
    trap - EXIT
    "${SERVE[@]}" stop >>"$OBS/driver.log" 2>&1 || true
    for unit in "${UNITS[@]}"; do
        systemctl --user stop "$unit" >>"$OBS/driver.log" 2>&1 || true
    done
    [[ -n ${RELEASER:-} ]] && kill "$RELEASER" 2>/dev/null || true
    [[ -n ${WATCHDOG:-} ]] && kill "$WATCHDOG" 2>/dev/null || true
    [[ -e $OBS/server-start2.log ]] || copy_server_log "$OBS/server-last.log"
    tail -n +"$((MEMWATCH_OFFSET + 1))" "$MEMWATCH_LOG" >"$OBS/memwatch.log" 2>/dev/null || true
    restore_production || rc=1
    sleep 5
    kill "$SAMPLER" "$JOURNAL" 2>/dev/null || true
    log "driver exit rc=$rc"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP INT

release_and_gate() {  # release_and_gate <label> <failure marker>
    PYTHONPATH=$WT/scripts/lib python3 - "${RELEASE_ROOTS[@]}" <<'PY' | tee -a "$OBS/driver.log"
import sys
from checkpoint_cache import release_tree
for root in sys.argv[1:]:
    print(f"released {release_tree(root)} files under {root}")
PY
    local i run=0 free=0
    for ((i = 0; i < 120; i++)); do
        free=$(meminfo MemFree)
        if (( free >= 108 * GIB )); then
            (( ++run >= 3 )) && { log "$1: MemFree gate pass ${free} kB (3 samples)"; return 0; }
        else
            run=0
        fi
        sleep 1
    done
    log "$2 $1: MemFree ${free} kB < 108 GiB after release"
    return 1
}

check_no_survivors() {  # check_no_survivors <unit> <failure marker>
    local i
    for ((i = 0; i < 60; i++)); do
        systemctl --user is-active -q "$1" || { log "$1: scope empty after stop"; return 0; }
        sleep 1
    done
    log "$2 $1: surviving descendants; stopping scope"
    systemctl --user stop "$1" || true
    return 1
}

cgroup_snapshot() {  # cgroup_snapshot <unit> <label>
    local cg
    cg=$(systemctl --user show -P ControlGroup "$1" 2>/dev/null) || return 0
    [[ -n $cg ]] || return 0
    {
        printf '%s %s memory.peak=%s memory.current=%s\n' "$(stamp)" "$2" \
            "$(cat /sys/fs/cgroup$cg/memory.peak 2>/dev/null)" "$(cat /sys/fs/cgroup$cg/memory.current 2>/dev/null)"
        grep -E '^(anon|file|shmem|file_mapped) ' /sys/fs/cgroup$cg/memory.stat 2>/dev/null | sed "s/^/  /"
    } >>"$OBS/cgroup.log"
}

start_server() {  # start_server <n>
    local unit="mimo26-smoke-$1-$$" t0=$SECONDS
    release_and_gate "start $1" FAILED_ATTEMPT
    # Keep the pack's page cache from crowding MemFree while the weights load
    # (GB10 CUDA allocations come from MemFree and are not cgroup-charged).
    ( set +e
      while :; do
          sleep 5
          PYTHONPATH=$WT/scripts/lib python3 -c 'import sys; from checkpoint_cache import release_tree; [release_tree(r) for r in sys.argv[1:]]' "$PACK" "$DRAFT"
      done ) &
    RELEASER=$!
    log "start $1: launching in scope $unit"
    systemd-run --user --scope -q --unit="$unit" \
        -p MemoryHigh=16G -p MemoryMax=48G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=7200 \
        -- "${SERVE[@]}" start | tee -a "$OBS/driver.log"
    kill "$RELEASER" 2>/dev/null || true
    RELEASER=
    log "start $1: ready after $((SECONDS - t0)) s"
    log "start $1 idle MemAvailable=$(meminfo MemAvailable) MemFree=$(meminfo MemFree)"
    cgroup_snapshot "$unit.scope" "start $1 ready"
}

stop_server() {  # stop_server <n>
    cgroup_snapshot "mimo26-smoke-$1-$$.scope" "start $1 before stop"
    "${SERVE[@]}" stop | tee -a "$OBS/driver.log"
    check_no_survivors "mimo26-smoke-$1-$$.scope" FAILED_ATTEMPT
    copy_server_log "$OBS/server-start$1.log"
}

log "head $(git -C "$WT" rev-parse HEAD) dirty=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)"
log "prod-stop: sudo -n 52_engine_switch.sh stop"
sudo -n "$PROD/scripts/52_engine_switch.sh" stop 2>&1 | tee -a "$OBS/driver.log"
log "preload MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"

start_server 1
timeout 1800 python3 "$EV/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$EV/probe-start1.json" \
    | tee -a "$OBS/driver.log" || log "probe start 1 exit nonzero"
timeout 7200 python3 "$WT/scripts/94_qualify_profile.py" --profile "$PROFILE" --host "$HOST" --port "$PORT" \
    --no-launch --cells speed,toolcall --reasoning-effort '' --out "$EV/kit" 2>&1 \
    | tee -a "$OBS/driver.log" || log "kit exit nonzero"
stop_server 1

start_server 2
timeout 1800 python3 "$EV/probe.py" --base-url "http://127.0.0.1:$PORT" --out "$EV/probe-start2.json" \
    | tee -a "$OBS/driver.log" || log "probe start 2 exit nonzero"
stop_server 2
log "smoke window closed"
kill "$JOURNAL" 2>/dev/null || true
journalctl -k -f -o short-iso --since now >"$OBS/kernel-tail.log" 2>&1 &
JOURNAL=$!

# Fidelity: separate from the pack verdict (PREREG.md). Its own watchdog kills
# the scope below 10 GiB MemAvailable, since its CUDA use is not cgroup-charged.
if release_and_gate "fidelity" FIDELITY_FAILED; then
    unit="mimo26-fidelity-$$.scope"
    ( set +e
      for ((i = 0; i < 60; i++)); do systemctl --user is-active -q "$unit" && break; sleep 0.5; done
      while systemctl --user is-active -q "$unit"; do
          a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
          if (( a < 10 * 1048576 )); then
              printf '%s FIDELITY_FAILED watchdog: MemAvailable %s kB < 10 GiB; killing %s\n' \
                  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$a" "$unit" >>"$OBS/driver.log"
              systemctl --user kill -s KILL "$unit"
              break
          fi
          sleep 0.5
      done ) &
    WATCHDOG=$!
    log "fidelity: launching model_diff"
    if systemd-run --user --scope -q --unit="${unit%.scope}" \
        -p MemoryHigh=16G -p MemoryMax=64G -p MemorySwapMax=0 -p OOMPolicy=kill -p RuntimeMaxSec=14400 \
        --setenv=MIMO_WIKI2_ROWS=/home/bmarti44/.cache/mimo-v26-flash/fidelity/wiki2-test-rows.json \
        --setenv=PYTHONNOUSERSITE=1 --setenv=HF_HUB_OFFLINE=1 --setenv=HF_DATASETS_OFFLINE=1 \
        --setenv=HOME=/home/bmarti44/.cache/mimo-v26-flash/home \
        -- "$PY" -B "$EV/fidelity/run_model_diff.py" "$RT/src/exllamav3/eval/model_diff.py" \
           /home/bmarti44/.cache/mimo-v26-flash/fidelity/work -- \
           -ma "$PACK" -mb "$OFFICIAL" -r 100 -l 2048 -cq 8,8 >"$EV/fidelity/model_diff.log" 2>&1; then
        log "fidelity rc=0"
    else
        log "FIDELITY_FAILED rc=$?"
    fi
    kill "$WATCHDOG" 2>/dev/null || true
    WATCHDOG=
    check_no_survivors "$unit" FIDELITY_FAILED || true
fi
log "postrun MemFree=$(meminfo MemFree) MemAvailable=$(meminfo MemAvailable) SwapFree=$(meminfo SwapFree)"
