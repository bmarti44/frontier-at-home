#!/usr/bin/env bash
# Gate-window helpers for model-qualification scripts.
#
# Every gate window in the Qwen and Laguna campaigns hand-rolled the same
# boilerplate — stop production, memory-guard, serve a dev config, restore
# production on exit, capture evidence JSON — and two of the campaign's four
# gate failures lived in exactly that boilerplate. This library owns it once.
#
# Usage (from a gate script):
#   source /home/bmarti44/spark-deepseek-v4-flash/scripts/lib/gate_window.sh
#   gate_window_open                # stop production + arm the restore trap
#   gate_serve_cycle scripts/25_serve_laguna.sh LAGUNA_CTX=65536 ...
#   ... probes ...
#   capture_json "$OUT/probe.json" curl -s http://127.0.0.1:8016/...
#   gate_window_close               # explicit restore + health verify
#
# The EXIT trap restores production even if the gate script dies mid-probe.
# Requires: the switch's stop/restore verbs (sudoers NOPASSWD wildcard).

GATE_REPO=${GATE_REPO:-/home/bmarti44/spark-deepseek-v4-flash}
GATE_PRODUCTION_HEALTH_URL=${GATE_PRODUCTION_HEALTH_URL:-http://127.0.0.1:8013/health}
GATE_RESTORE_ATTEMPTS=${GATE_RESTORE_ATTEMPTS:-60}
_gate_serve_script=""
_gate_window_armed=false

gate_die() { printf 'GATE ERROR: %s\n' "$*" >&2; exit 1; }

# Stop the active production profile (active.json untouched) and arm the
# restore trap. Idempotent per script; refuses to double-open.
gate_window_open() {
    [[ $_gate_window_armed == false ]] || gate_die "gate window already open"
    echo "[gate] stopping production"
    sudo "$GATE_REPO/scripts/52_engine_switch.sh" stop \
        || gate_die "production stop failed"
    _gate_window_armed=true
    trap gate_restore_production EXIT
}

# Restore production and verify health. Used by the EXIT trap and by
# gate_window_close; safe to call twice.
gate_restore_production() {
    set +e
    [[ $_gate_window_armed == true ]] || return 0
    _gate_window_armed=false
    if [[ -n $_gate_serve_script ]]; then
        echo "[gate-restore] stopping dev server ($_gate_serve_script)"
        bash "$_gate_serve_script" stop >/dev/null 2>&1
    fi
    echo "[gate-restore] restoring production"
    sudo "$GATE_REPO/scripts/52_engine_switch.sh" restore
    local attempt
    for ((attempt = 0; attempt < GATE_RESTORE_ATTEMPTS; attempt++)); do
        if curl -s --max-time 3 "$GATE_PRODUCTION_HEALTH_URL" >/dev/null 2>&1; then
            echo "[gate-restore] production healthy"
            return 0
        fi
        sleep 5
    done
    echo "[gate-restore] WARNING: production did not report healthy" >&2
    return 1
}

# Close the window explicitly (clears the trap after restoring).
gate_window_close() {
    gate_restore_production
    trap - EXIT
}

# (Re)start a dev serve script with env overrides, running the 100 GiB
# release gate between cycles. First arg: serve script path; remaining
# args: NAME=VALUE env pairs passed to its start verb.
#   gate_serve_cycle scripts/25_serve_laguna.sh LAGUNA_DFLASH=1 LAGUNA_CTX=65536
gate_serve_cycle() {
    local serve=$1; shift
    [[ -f $serve ]] || serve=$GATE_REPO/$1
    [[ -f $serve ]] || gate_die "serve script not found: $1"
    _gate_serve_script=$serve
    bash "$serve" stop >/dev/null 2>&1 || true
    python3 "$GATE_REPO/scripts/03_memory_guard.py" --required-gib 100 \
        --stable-samples 3 --timeout-seconds 240 >/dev/null \
        || gate_die "memory release gate failed before serve cycle"
    env "$@" bash "$serve" start || gate_die "serve start failed: $serve $*"
}

# Run a command and write its stdout to an evidence file ONLY if the output
# is a single valid JSON document (guards against tee'd multi-line captures
# that the lint hook later rejects). On invalid JSON the raw output is kept
# at <path>.raw for debugging and the function fails.
capture_json() {
    local path=$1; shift
    local output
    output=$("$@") || gate_die "capture_json command failed: $*"
    if printf '%s' "$output" | python3 -c '
import json, sys
json.load(sys.stdin)
' 2>/dev/null; then
        printf '%s\n' "$output" >"$path"
    else
        printf '%s\n' "$output" >"$path.raw"
        gate_die "capture_json: output is not a single JSON document (kept at $path.raw)"
    fi
}

gate_mem_avail_gib() {
    awk '$1 == "MemAvailable:" {printf "%.1f", $2 / 1048576}' /proc/meminfo
}
